"""Presigned uploads, resolved into the same file parts as multipart requests."""

from contextlib import asynccontextmanager
import logging
import mimetypes
import os
from pathlib import PurePosixPath
import tempfile
import time
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import HTTPException
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import FormData, Headers, UploadFile

from server import print_jobs

NAMES = {"photo", "mesh", "depth", "depth_confidence", "face_vertices"}
EXPIRES_IN = 900
logger = logging.getLogger(__name__)


class MissingUploads(HTTPException):
    def __init__(self, missing):
        self.missing = missing
        super().__init__(409, "upload missing: " + ", ".join(missing))


def client():
    region = os.environ.get("AWS_REGION") or boto3.session.Session().region_name or "us-east-1"
    return boto3.client("s3", region_name=region, endpoint_url=f"https://s3.{region}.amazonaws.com",
                        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))


def bucket():
    value = os.environ.get("PRINT_S3_BUCKET")
    if not value:
        raise HTTPException(503, "S3 uploads are not configured")
    return value


def job_uuid(value):
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(400, "`job_id` must be a UUID")


def register(payload, limit):
    if not isinstance(payload, dict):
        raise HTTPException(400, "expected a JSON object")
    job_id = job_uuid(payload["job_id"]) if payload.get("job_id") is not None else str(uuid.uuid4())
    files = payload.get("files")
    if not isinstance(files, dict) or not files:
        raise HTTPException(400, "files must be a nonempty object")
    entries = {}
    for name, entry in files.items():
        if name not in NAMES:
            raise HTTPException(400, f"unknown upload name: {name}")
        if not isinstance(entry, dict):
            raise HTTPException(400, f"{name}: expected content_type and size")
        size, content_type = entry.get("size"), entry.get("content_type")
        if type(size) is not int or not 0 < size <= limit:
            raise HTTPException(400, f"{name}: size must be between 1 and {limit} bytes")
        if (not isinstance(content_type, str) or not content_type.isascii()
                or "/" not in content_type or len(content_type) > 255
                or any(ord(char) < 32 for char in content_type)):
            raise HTTPException(400, f"{name}: invalid content_type")
        filename = entry.get("filename")
        if filename is not None and (not isinstance(filename, str) or len(filename) > 255):
            raise HTTPException(400, f"{name}: invalid filename")
        defaults = {"photo": ".jpg", "mesh": ".obj", "depth": ".f32",
                    "depth_confidence": ".u8", "face_vertices": ".f32"}
        suffix = (defaults[name] if content_type == "application/octet-stream" else
                  ".jpg" if content_type == "image/jpeg" else mimetypes.guess_extension(content_type))
        filename = PurePosixPath((filename or name + (suffix or defaults[name])).replace("\\", "/")).name
        if filename in ("", ".", "..") or any(ord(char) < 32 for char in filename):
            raise HTTPException(400, f"{name}: invalid filename")
        entries[name] = dict(key=f"uploads/{job_id}/{name}/{filename}", filename=filename,
                             size=size, content_type=content_type)
    s3, target = client(), bucket()
    previous = print_jobs.store.get(f"{job_id}:uploads") or {"files": {}}
    uploads = {}
    for name, entry in entries.items():
        url = s3.generate_presigned_url("put_object", Params={"Bucket": target, "Key": entry["key"],
                                       "ContentType": entry["content_type"], "ContentLength": entry["size"]},
                                        ExpiresIn=EXPIRES_IN)
        uploads[name] = dict(url=url, method="PUT", headers={"Content-Type": entry["content_type"]})
    print_jobs.store.put(f"{job_id}:uploads", dict(created=time.time(), files={**previous["files"], **entries}))
    return dict(job_id=job_id, expires_in=EXPIRES_IN, uploads=uploads)


def open_files(job_id, names, limit):
    manifest = print_jobs.store.get(f"{job_id}:uploads") or {"files": {}}
    s3, target = client(), bucket()
    missing, entries = [], {}
    for name in names:
        entry = manifest["files"].get(name)
        if entry is None:
            missing.append(name)
            continue
        try:
            head = s3.head_object(Bucket=target, Key=entry["key"])
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                missing.append(name)
                continue
            raise
        entries[name] = entry, head
    if missing:
        raise MissingUploads(missing)
    uploads = {}
    try:
        for name, (entry, head) in entries.items():
            if head["ContentLength"] != entry["size"] or not 0 < head["ContentLength"] <= limit:
                raise HTTPException(400, f"{name}: uploaded size does not match the declared size")
            stream = tempfile.SpooledTemporaryFile(max_size=1 << 20)
            uploads[name] = UploadFile(stream, filename=entry["filename"],
                                       headers=Headers({"Content-Type": entry["content_type"]}))
            response = s3.get_object(Bucket=target, Key=entry["key"])
            total = 0
            try:
                for chunk in response["Body"].iter_chunks(1 << 20):
                    total += len(chunk)
                    if total > limit or total > entry["size"]:
                        raise HTTPException(400, f"{name}: uploaded size changed during download")
                    stream.write(chunk)
            finally:
                response["Body"].close()
            if total != entry["size"]:
                raise HTTPException(400, f"{name}: incomplete upload")
            stream.seek(0)
        return uploads
    except BaseException:
        for upload in uploads.values():
            upload.file.close()
        raise


def delete_files(job_id):
    s3, target = client(), bucket()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=target, Prefix=f"uploads/{job_id}/"):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if objects:
            response = s3.delete_objects(Bucket=target, Delete={"Objects": objects, "Quiet": True})
            if response.get("Errors"):
                raise RuntimeError("S3 did not delete all uploaded objects")


@asynccontextmanager
async def resolve(request, limit):
    if request.method != "POST" or request.url.path != "/print":
        yield
        return
    form = await request.form()
    uploaded = form.get("uploaded")
    if uploaded is None:
        yield
        return
    job_id = job_uuid(form.get("job_id"))
    files, cleanup = {}, True
    try:
        if not isinstance(uploaded, str):
            raise HTTPException(400, "uploaded must be a comma-separated list of file names")
        names = [name.strip() for name in uploaded.split(",")]
        if not names or any(name not in NAMES for name in names) or len(names) != len(set(names)):
            raise HTTPException(400, "uploaded contains an unknown or repeated file name")
        for name in names:
            if any(isinstance(value, UploadFile) for value in form.getlist(name)):
                raise HTTPException(400, f"{name}: send either a file part or uploaded, not both")
        if await run_in_threadpool(print_jobs.get_state, job_id) is not None:
            cleanup = False
            request.state.print_join_job_id = job_id
            yield
            return
        try:
            files = await run_in_threadpool(open_files, job_id, names, limit)
        except MissingUploads:
            cleanup = False
            raise
        request._form = FormData([(key, value) for key, value in form.multi_items() if key not in names]
                                 + list(files.items()))
        yield
    finally:
        for upload in files.values():
            upload.file.close()
        if cleanup:
            try:
                await run_in_threadpool(delete_files, job_id)
            except Exception as exc:
                code = getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)
                logger.warning("could not remove S3 uploads for print job %s error=%s", job_id, code)

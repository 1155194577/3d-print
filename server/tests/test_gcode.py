"""K1C PLA G-code: every print is sliced after its STL; GET /print/{id}/gcode serves it."""

import json
import stat
import sys
import textwrap

import pytest
import trimesh
from fastapi.testclient import TestClient

from server import print_api as api
from server import print_gcode, print_jobs
from server.app import Settings, create_app

NDJSON = {"Accept": "application/x-ndjson"}


def fake_slicer(tmp_path, moves="G1 X10 Y10 Z0.2 E1\nG1 X200 Y200 Z30 E1\n", exit_code=0):
    script = tmp_path / "prusa-slicer"
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import sys
        args = sys.argv[1:]
        if {exit_code}:
            print("Objects could not fit on the bed", file=sys.stderr)
            sys.exit({exit_code})
        with open(args[args.index("-o") + 1], "w") as f:
            f.write("START_PRINT EXTRUDER_TEMP=220 BED_TEMP=60\\n" + {moves!r} + "END_PRINT\\n")
            f.write("; filament used [mm] = 1234.5\\n; total filament used [g] = 3.71\\n")
            f.write("; estimated printing time (normal mode) = 1h 2m 3s\\n")
        """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.delenv("PRINT_API_TOKEN", raising=False)
    monkeypatch.delenv("PRINT_S3_BUCKET", raising=False)
    monkeypatch.setattr(api, "HEARTBEAT_S", 0.05)
    monkeypatch.setattr(api, "POLL_S", 0.01)
    monkeypatch.setattr(print_jobs, "store", print_jobs.MemoryStore())
    with TestClient(create_app(Settings(jobs_dir=tmp_path / "jobs"))) as c:
        yield c


def mesh_print(client, extents=(0.05, 0.05, 0.05), name="cube.obj"):
    data = trimesh.creation.box(extents=extents).export(file_type="obj")
    out = [json.loads(x) for x in client.post("/print", files={"mesh": (name, data)}, headers=NDJSON).text.splitlines()]
    assert out[-1]["type"] == "done", out[-1]
    return out


def test_every_print_is_sliced_for_the_k1c(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PRUSA_SLICER", fake_slicer(tmp_path))
    out = mesh_print(client)
    assert "slice" in [o.get("stage") for o in out]
    gcode = out[-1]["report"]["gcode"]
    assert gcode["printer"] == "Creality K1C (CUHK MakerSpace)" and gcode["filament"] == "PLA 1.75 mm"
    assert gcode["print_time"] == "1h 2m 3s" and gcode["print_time_s"] == 3723
    assert gcode["filament_g"] == 3.71 and gcode["filament_m"] == 1.23
    assert gcode["bounds_mm"] == [[10, 10, 0.2], [200, 200, 30]] and "skipped" not in gcode
    resp = client.get(f"/print/{out[0]['job_id']}/gcode")
    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/x-gcode")
    assert 'filename="cube.gcode"' in resp.headers["content-disposition"]
    assert resp.text.startswith("START_PRINT EXTRUDER_TEMP=220 BED_TEMP=60")


def test_toolpath_outside_the_bed_is_refused(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PRUSA_SLICER", fake_slicer(tmp_path, moves="G1 X10 Y10 Z0.2\nG0 X230 Y100\n"))
    out = mesh_print(client)
    assert "build volume" in out[-1]["report"]["gcode"]["skipped"]
    resp = client.get(f"/print/{out[0]['job_id']}/gcode")
    assert resp.status_code == 404 and "build volume" in resp.json()["detail"]


def test_slicer_failure_still_delivers_the_stl(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PRUSA_SLICER", fake_slicer(tmp_path, exit_code=1))
    out = mesh_print(client)
    assert out[-1]["report"]["gcode"]["skipped"] == "the slicer failed: Objects could not fit on the bed"
    assert out[-1]["stl_base64"]


def test_no_slicer_or_too_big_skips(monkeypatch):
    monkeypatch.setenv("PRUSA_SLICER", "/nonexistent/prusa-slicer")
    monkeypatch.setattr(print_gcode.shutil, "which", lambda _: None)
    gcode, report = print_gcode.slice_stl(b"", [50, 50, 50], "a.gcode")
    assert gcode is None and report["skipped"] == "no slicer on this server"
    gcode, report = print_gcode.slice_stl(b"", [221, 50, 50], "a.gcode")
    assert gcode is None and "220 × 220 × 250" in report["skipped"]
    assert print_gcode.slice_stl(b"", [100, 100, 251], "a.gcode")[0] is None


def test_unknown_job_has_no_gcode(client):
    assert client.get("/print/00000000-0000-0000-0000-000000000000/gcode").status_code == 404


@pytest.mark.skipif(print_gcode.slicer() is None, reason="PrusaSlicer CLI not installed")
def test_real_slice_fits_the_k1c():
    stl = trimesh.creation.box(extents=[40, 30, 20]).export(file_type="stl")
    gcode, report = print_gcode.slice_stl(stl, [40, 30, 20], "box.gcode")
    assert gcode is not None, report
    assert b"START_PRINT EXTRUDER_TEMP=220 BED_TEMP=60" in gcode and b"END_PRINT" in gcode
    lo, hi = report["bounds_mm"]
    assert min(lo) >= 0 and hi[0] <= 220 and hi[1] <= 220 and 19.5 < hi[2] <= 250
    assert report["print_time_s"] > 0 and report["filament_g"] > 0


def test_non_ascii_names_download_with_an_encoded_filename(client, monkeypatch, tmp_path):
    monkeypatch.setenv("PRUSA_SLICER", fake_slicer(tmp_path))
    out = mesh_print(client, name="杯子.obj")
    resp = client.get(f"/print/{out[0]['job_id']}/gcode")
    assert resp.status_code == 200
    assert "filename*=UTF-8''%E6%9D%AF%E5%AD%90.gcode" in resp.headers["content-disposition"]


def test_uploaded_gcode_is_served_from_storage_not_memory(monkeypatch, tmp_path):
    import boto3
    from moto import mock_aws

    for k, v in dict(AWS_ACCESS_KEY_ID="testing", AWS_SECRET_ACCESS_KEY="testing", AWS_REGION="us-east-1",
                     PRINT_S3_BUCKET="print-test-uploads", PRUSA_SLICER=fake_slicer(tmp_path)).items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("PRINT_API_TOKEN", raising=False)
    monkeypatch.setattr(print_jobs, "store", print_jobs.MemoryStore())
    monkeypatch.setattr(api, "POLL_S", 0.01)
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="print-test-uploads")
        with TestClient(create_app(Settings(jobs_dir=tmp_path / "jobs"))) as c:
            out = mesh_print(c)
            job_id = out[0]["job_id"]
            assert out[-1]["gcode_url"] and "gcode_chunks" not in print_jobs.store.get(job_id)
            resp = c.get(f"/print/{job_id}/gcode", follow_redirects=False)
            assert resp.status_code == 307 and resp.headers["location"] == print_uploads_url(resp)


def print_uploads_url(resp):
    assert "results/" in resp.headers["location"] and "model.gcode" in resp.headers["location"]
    assert "response-content-disposition=attachment" in resp.headers["location"]
    assert "cube.gcode" in resp.headers["location"]
    return resp.headers["location"]

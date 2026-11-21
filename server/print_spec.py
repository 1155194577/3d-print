"""Transport validated uploads and parameters without sharing filesystem paths."""

from pathlib import Path

import numpy as np


def _encode(value):
    if isinstance(value, np.ndarray):
        return {"data": value.tobytes(), "dtype": value.dtype.str, "shape": value.shape}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_encode(item) for item in value)
    return value


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"data", "dtype", "shape"}:
            return np.frombuffer(value["data"], dtype=value["dtype"]).reshape(value["shape"]).copy()
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_decode(item) for item in value)
    return value


def _pack_frame(frame):
    return _encode(vars(frame)) if frame is not None else None


def _unpack_frame(attributes):
    from depth_fusion import Frame

    if attributes is None:
        return None
    frame = Frame.__new__(Frame)
    vars(frame).update(_decode(attributes))
    return frame


def pack(job):
    source = Path(job["src"])
    files = {path.name: path.read_bytes() for path in Path(job["workdir"]).iterdir()
             if path.is_file() and path != source}
    files["source"] = source.read_bytes()
    return {
        "files": files,
        "suffix": source.suffix,
        "params": _encode({key: value for key, value in job.items()
                           if key not in ("src", "workdir", "frame", "views")}),
        "views": [_encode(dict(view, src=Path(view["src"]).name,
                               frame=_pack_frame(view["frame"])))
                  for view in job.get("views", [])],
        "frame": _pack_frame(job["frame"]),
    }


def unpack(spec, workdir):
    source = Path(workdir) / ("source" + spec["suffix"])
    for name, data in spec["files"].items():
        path = source if name == "source" else Path(workdir) / name
        path.write_bytes(data)
    views = [dict(_decode({key: value for key, value in view.items() if key != "frame"}),
                  src=str(Path(workdir) / view["src"]), frame=_unpack_frame(view["frame"]))
             for view in spec.get("views", [])]
    return dict(_decode(spec["params"]), src=str(source), workdir=str(workdir),
                frame=_unpack_frame(spec["frame"]), views=views)

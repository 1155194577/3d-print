"""Transport validated uploads and parameters without sharing filesystem paths."""

from pathlib import Path

import numpy as np


def pack(job):
    source = Path(job["src"])
    files = {path.name: path.read_bytes() for path in Path(job["workdir"]).iterdir()
             if path.is_file() and path != source}
    files["source"] = source.read_bytes()
    frame = job["frame"]
    attributes = None
    if frame is not None:
        attributes = {}
        for key, value in vars(frame).items():
            attributes[key] = ({"data": value.tobytes(), "dtype": value.dtype.str, "shape": value.shape}
                               if isinstance(value, np.ndarray) else value)
    return {
        "files": files,
        "suffix": source.suffix,
        "params": {key: value for key, value in job.items() if key not in ("src", "workdir", "frame")},
        "frame": attributes,
    }


def unpack(spec, workdir):
    from depth_fusion import Frame

    source = Path(workdir) / ("source" + spec["suffix"])
    for name, data in spec["files"].items():
        path = source if name == "source" else Path(workdir) / name
        path.write_bytes(data)
    frame = None
    if spec["frame"] is not None:
        frame = Frame.__new__(Frame)
        for key, value in spec["frame"].items():
            if isinstance(value, dict) and "data" in value:
                value = np.frombuffer(value["data"], dtype=value["dtype"]).reshape(value["shape"]).copy()
            setattr(frame, key, value)
    return dict(spec["params"], src=str(source), workdir=str(workdir), frame=frame)

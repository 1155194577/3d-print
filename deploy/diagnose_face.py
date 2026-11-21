"""Isolate face reconstruction, cutting and preparation in the deployment image.

modal run -m deploy.diagnose_face --photo face.jpg --mesh raw-face.obj
modal run -m deploy.diagnose_face --photo face.jpg --mesh raw-face.obj --operation object
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import modal

from deploy.modal_app import APP_DIR, image

app = modal.App("3d-print-face-diagnostic", image=image)


@app.function(gpu="L4", timeout=600)
def reconstruct(photo: bytes):
    sys.path.insert(0, APP_DIR)
    from server.print_api import reconstruct_photo

    with tempfile.TemporaryDirectory() as directory:
        source, output = Path(directory) / "face.jpg", Path(directory) / "raw.obj"
        source.write_bytes(photo)
        print("starting reconstruction, quarter_turns=0", flush=True)
        reconstruct_photo(str(source), str(output), quarter_turns=0)
        print("reconstruction complete", flush=True)
        return output.read_bytes()


@app.function(timeout=600)
def process(mesh: bytes, operation: str):
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "raw.obj"
        source.write_bytes(mesh)
        script = '''
import faulthandler, importlib.metadata, resource, sys
faulthandler.enable()
sys.path.insert(0, "/root/app/tools")
import trimesh, depth_fusion, img2mesh
for name in ("numpy", "trimesh", "shapely", "mapbox-earcut", "rtree", "pymeshfix"):
    print(name, importlib.metadata.version(name), flush=True)
mesh = trimesh.load(sys.argv[1], force="mesh")
print("loaded", len(mesh.vertices), len(mesh.faces), "watertight", mesh.is_watertight,
      "rss_kb", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
if sys.argv[2] != "object":
    print("starting cut", flush=True)
    mesh = depth_fusion.cut_bust(mesh, mesh.bounds[0, 2] + 0.2 * mesh.extents[2])
    print("cut complete", len(mesh.vertices), len(mesh.faces), "watertight", mesh.is_watertight,
          "rss_kb", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, flush=True)
if sys.argv[2] != "cut":
    mesh.export(sys.argv[1] + ".ply")
    print("starting prepare", flush=True)
    out, report = img2mesh.prepare_mesh(sys.argv[1] + ".ply", height=102.5)
    print("prepare complete", report, flush=True)
'''
        result = subprocess.run([sys.executable, "-u", "-c", script, str(source), operation],
                                capture_output=True, text=True)
        print(result.stdout, result.stderr, "returncode", result.returncode, flush=True)
        return result.returncode


@app.local_entrypoint()
def main(photo: str,
         mesh: str = "raw-face.obj", operation: str = "face"):
    if operation not in ("face", "cut", "object"):
        raise ValueError("operation must be face, cut or object")
    path = Path(mesh)
    if not path.exists():
        path.write_bytes(reconstruct.remote(Path(photo).read_bytes()))
    code = process.remote(path.read_bytes(), operation)
    if code:
        raise RuntimeError(f"native processing exited with code {code}")

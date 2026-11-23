import io

import numpy as np
from PIL import Image

from server.remote_triposr import ModalReconstructor


def _png(mask):
    buf = io.BytesIO()
    Image.fromarray(mask).save(buf, format="PNG")
    return buf.getvalue()


class FakeMethod:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def remote_gen(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.fn(*args, **kwargs)

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.fn(*args, **kwargs)


class FakeService:
    def __init__(self, mask):
        def reconstruct(image, suffix, **kwargs):
            yield "stage", "background"
            yield "stage", "reconstruct"
            yield "stage", "mesh"
            yield "result", {"mesh": b"o mesh\n", "mask": _png(mask)}

        self.reconstruct = FakeMethod(reconstruct)
        self.foreground_mask = FakeMethod(lambda image, suffix: _png(mask))


def test_reconstruct_streams_stages_and_caches_mask(tmp_path):
    mask = np.zeros((4, 6), np.uint8)
    mask[1:3, 2:5] = 255
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    remote = ModalReconstructor()
    remote._remote = service = FakeService(mask)
    stages = []

    remote(str(photo), str(tmp_path / "raw.obj"), quarter_turns=1, progress=stages.append)

    assert stages == ["background", "reconstruct", "mesh"]
    assert (tmp_path / "raw.obj").read_bytes() == b"o mesh\n"
    args, kwargs = service.reconstruct.calls[0]
    assert args == (b"jpeg", ".jpg") and kwargs["quarter_turns"] == 1
    np.testing.assert_array_equal(remote.foreground_mask(str(photo)), mask)
    assert service.foreground_mask.calls == []
    np.testing.assert_array_equal(remote.foreground_mask(str(photo)), mask)
    assert len(service.foreground_mask.calls) == 1


def test_unread_masks_are_bounded(tmp_path):
    from server import remote_triposr

    remote = ModalReconstructor()
    remote._remote = FakeService(np.zeros((2, 2), np.uint8))
    for i in range(remote_triposr.MASK_CACHE + 5):
        photo = tmp_path / f"p{i}.jpg"
        photo.write_bytes(b"jpeg")
        remote(str(photo), str(tmp_path / "raw.obj"))
    assert len(remote._masks) == remote_triposr.MASK_CACHE


def test_remote_returns_preprocessed_input_for_box_review(tmp_path):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"jpeg")
    png = _png(np.full((32, 32), 128, np.uint8))

    def reconstruct(*args, **kwargs):
        assert kwargs["with_input"]
        yield "result", {"mesh": b"o mesh\n", "input": png}

    service = FakeService(np.zeros((2, 2), np.uint8))
    service.reconstruct = FakeMethod(reconstruct)
    remote = ModalReconstructor()
    remote._remote = service
    destination = tmp_path / "input.png"
    remote(str(photo), str(tmp_path / "raw.obj"), input_path=str(destination))
    assert destination.read_bytes() == png

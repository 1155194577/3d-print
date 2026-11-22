"""Checks the on-device split against TripoSR itself. Needs the weights:

    TRIPOSR_MODEL_DIR=/path/with/config.yaml+model.ckpt python -m pytest edge
"""

import copy
import os

import pytest
import torch

import triposr_edge as te

MODEL_DIR = os.environ.get("TRIPOSR_MODEL_DIR")
pytestmark = pytest.mark.skipif(not MODEL_DIR, reason="set TRIPOSR_MODEL_DIR to the TripoSR weights")
torch.set_grad_enabled(False)


@pytest.fixture(scope="module")
def tsr():
    return te.load_tsr(MODEL_DIR)


@pytest.fixture(scope="module")
def image():
    torch.manual_seed(0)
    img = torch.full((1, 3, te.IMAGE_SIZE, te.IMAGE_SIZE), 0.5)
    img[:, :, 128:384, 160:352] = torch.rand(1, 3, 256, 192)
    return img


def test_encoder_matches_tsr(tsr, image):
    expected = tsr(image[0].permute(1, 2, 0), device="cpu")[0]
    got = te.Encoder(tsr)(image)
    assert got.shape == (3, 40, 64, 64)
    assert (got - expected).norm() / expected.norm() < 1e-4


def test_decoder_matches_renderer(tsr, image):
    planes = te.Encoder(tsr)(image)
    points = (torch.rand(4096, 3) * 2 - 1) * te.RADIUS
    expected = tsr.renderer.query_triplane(tsr.decoder, points, planes)["density_act"]
    got = torch.exp(te.Decoder(tsr)(planes, points))
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("scheme,max_err", [("fp16", 1e-3), ("int8", 0.03), ("int4-block32", 0.15)])
def test_quantized_triplanes_stay_close(tsr, image, scheme, max_err):
    reference = te.Encoder(tsr)(image)
    quantized = copy.deepcopy(tsr)
    te.fake_quantize(quantized, scheme)
    got = te.Encoder(quantized)(image)
    assert (got - reference).norm() / reference.norm() < max_err


def test_grid_is_x_major():
    pts = te.grid_points(3)
    assert pts[1].tolist() == pytest.approx([-te.RADIUS, -te.RADIUS, 0])
    assert pts[3].tolist() == pytest.approx([-te.RADIUS, 0, -te.RADIUS])

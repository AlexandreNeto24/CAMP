import torch

from camp.blend import SeamlessBlendingPipeline


def test_seamless_blending_pipeline_run_returns_final():
    size = 32
    lesion = torch.rand(size, size, 3)
    background = torch.rand(size, size, 3)
    mask = torch.zeros(size, size)
    mask[8:24, 8:24] = 1.0

    result = SeamlessBlendingPipeline().run(lesion, mask, background)

    assert isinstance(result, dict)
    assert "final" in result

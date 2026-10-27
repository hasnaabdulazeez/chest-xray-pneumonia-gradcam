import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xray.gradcam import GradCAM  # noqa: E402


class TinyCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU())
        self.head = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.head(self.features(x).mean(dim=(2, 3)))


def test_heatmap_shape_and_range():
    torch.manual_seed(0)
    model = TinyCNN()
    cam = GradCAM(model, model.features)
    heatmap, logits = cam(torch.rand(1, 3, 32, 32))
    cam.remove()
    assert heatmap.shape == (32, 32)
    assert logits.shape == (1, 1)
    assert np.all(heatmap >= 0) and heatmap.max() <= 1.0 + 1e-6

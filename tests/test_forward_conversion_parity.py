import importlib.util

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("jax") is None,
    reason="PyTorch and JAX optional dependencies are required",
)


def test_forward_surrogate_torch_to_jax_parity(tmp_path):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from pixelant_eggroll.checkpoints import forward_surrogate_from_torch_checkpoint
    from pixelant_eggroll.models_jax import forward_surrogate

    num_filters = [1, 64, 128, 256, 1000, 500, 81]
    kernels = [5, 5, 5, 3, 3, 3, 3]

    class NetBig(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(num_filters[0], num_filters[1], kernels[0], padding="same")
            self.conv2 = nn.Conv2d(num_filters[1], num_filters[1], kernels[1], padding="same")
            self.conv3 = nn.Conv2d(num_filters[1], num_filters[1], kernels[2], padding="same")
            self.conv4 = nn.Conv2d(num_filters[1], num_filters[2], kernels[3], padding="same")
            self.conv5 = nn.Conv2d(num_filters[2], num_filters[2], kernels[3], padding="same")
            self.conv6 = nn.Conv2d(num_filters[2], num_filters[2], kernels[3], padding="same")
            self.conv7 = nn.Conv2d(num_filters[2], num_filters[2], kernels[3], padding="same")
            self.conv8 = nn.Conv2d(num_filters[2], num_filters[2], kernels[3], padding="same")
            self.conv9 = nn.Conv2d(num_filters[2], num_filters[3], kernels[4], padding="same")
            self.conv10 = nn.Conv2d(num_filters[3], num_filters[3], kernels[4], padding="same")
            self.conv11 = nn.Conv2d(num_filters[3], num_filters[3], kernels[5], padding="same")
            self.conv12 = nn.Conv2d(num_filters[3], num_filters[3], kernels[5], padding="same")
            self.conv13 = nn.Conv2d(num_filters[3], num_filters[3], kernels[5], padding="same")
            self.conv14 = nn.Conv2d(num_filters[3], num_filters[3], kernels[5], padding="same")
            self.conv15 = nn.Conv2d(num_filters[3], num_filters[3], kernels[5], padding="same")
            self.conv16 = nn.Conv2d(num_filters[3], num_filters[3], kernels[5], padding="same")
            for idx, channels in enumerate([64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 256, 256], 1):
                setattr(self, f"bn{idx}", nn.BatchNorm2d(channels))
            self.fc17 = nn.Linear(num_filters[3] * 12 * 12, num_filters[4])
            self.fc18 = nn.Linear(num_filters[4], num_filters[5])
            self.fc19 = nn.Linear(num_filters[5], num_filters[6])
            self.bn17 = nn.BatchNorm1d(num_filters[4])
            self.bn18 = nn.BatchNorm1d(num_filters[5])

        def forward(self, x):
            for idx in range(1, 17):
                x = F.leaky_relu(getattr(self, f"bn{idx}")(getattr(self, f"conv{idx}")(x)))
            x = torch.flatten(x, 1)
            x = F.leaky_relu(self.bn17(self.fc17(x)))
            x = F.leaky_relu(self.bn18(self.fc18(x)))
            return self.fc19(x)

    torch.manual_seed(0)
    net = NetBig().eval()
    checkpoint = tmp_path / "forward.pth"
    torch.save({"state_dict": net.state_dict()}, checkpoint)
    params = forward_surrogate_from_torch_checkpoint(checkpoint)

    design = torch.randint(0, 2, (1, 1, 12, 12), dtype=torch.float32)
    with torch.no_grad():
        torch_out = net(design).numpy()
    jax_out = np.asarray(forward_surrogate(params, jnp.asarray(design.numpy())))

    np.testing.assert_allclose(jax_out, torch_out, rtol=1e-4, atol=1e-4)

from pathlib import Path
import torch
from torch import nn
from .ihgr_net import IHGRNet


class IHGRAdapter(nn.Module):
    """Frozen native IHGR-Net or exported TorchScript encoder."""
    def __init__(self, network, frozen=True):
        super().__init__()
        self.network = network
        self.frozen = frozen
        if frozen:
            self.network.requires_grad_(False)
            self.network.eval()

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        path = Path(path)
        if path.suffix in (".ts", ".torchscript"):
            network = torch.jit.load(str(path), map_location=device)
        else:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if checkpoint.get("kind") != "ihgr" or checkpoint.get("schema_version") != 1:
                raise ValueError("Expected a version-1 IHGR checkpoint produced by the pretrain command")
            network = IHGRNet(num_classes=checkpoint.get("num_classes", 74)).to(device)
            network.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
        return cls(network)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.network.eval()
        return self

    def forward(self, silhouettes):
        if self.frozen:
            with torch.no_grad():
                output = self.network(silhouettes)
        else:
            output = self.network(silhouettes)
        local = output["local"] if isinstance(output, dict) else output
        if local.shape[1:] != (16, 256):
            raise ValueError("IHGR adapter requires local output [N,16,256]")
        return local

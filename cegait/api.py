"""Public Python interface: silhouettes + active key -> packed template."""
import numpy as np
import torch
from .backbone import IHGRAdapter
from .training.alternate_trainer import load_protector
from .models.quantization import pack_codes, unpack_codes
from .protocols.domain_key_schedule import key_tensor


class CEGaitPipeline:
    def __init__(self, backbone, protector, device="cpu"):
        self.device = device
        self.backbone = backbone.to(device).eval()
        self.protector = protector.to(device).eval()

    @classmethod
    def from_checkpoints(cls, backbone_checkpoint, protector_checkpoint, device="cpu"):
        backbone = IHGRAdapter.from_checkpoint(backbone_checkpoint, device)
        protector, _ = load_protector(protector_checkpoint, device)
        return cls(backbone, protector, device)

    @torch.no_grad()
    def issue(self, silhouettes, domain_key):
        """domain_key: 16 bytes or 32 hexadecimal characters, shared per domain."""
        if isinstance(domain_key, str):
            domain_key = bytes.fromhex(domain_key)
        x = torch.as_tensor(silhouettes, dtype=torch.float32, device=self.device)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        local = self.backbone(x)
        result = self.protector(local, key_tensor([domain_key], self.device))
        return {"payload": pack_codes(result["codes"].cpu().numpy(), self.protector.bits),
                "codes": result["codes"].cpu().numpy(), "local_shape": list(local.shape),
                "attention": result["attention"].cpu().numpy()}

    @staticmethod
    def distance(first, second, bits=4):
        a = unpack_codes(np.asarray(first), bits).astype(float) * (2 / (2 ** bits - 1)) - 1
        b = unpack_codes(np.asarray(second), bits).astype(float) * (2 / (2 ** bits - 1)) - 1
        numerator = np.sum(a * b, -1)
        denominator = np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)
        return (1 - numerator / denominator) / 2

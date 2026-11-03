"""Domain keys are shared across identities; labels never enter key generation."""
import hashlib
import secrets
from functools import lru_cache
import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


@lru_cache(maxsize=8192)
def key_vector(key):
    if isinstance(key, str):
        key = bytes.fromhex(key)
    if len(key) != 16:
        raise ValueError("Domain keys must contain exactly 16 bytes")
    enc = Cipher(algorithms.AES(key), modes.CTR(bytes(16))).encryptor()
    stream = enc.update(bytes(32)) + enc.finalize()
    return np.unpackbits(np.frombuffer(stream, dtype=np.uint8), bitorder="big").astype(np.float32) * 2 - 1


def key_tensor(keys, device="cpu"):
    return torch.from_numpy(np.stack([key_vector(k) for k in keys])).to(device)


class DomainKeySchedule:
    """Reproducible experiment keys; production keys use new_domain_key()."""
    def __init__(self, seed=42, namespace="train"):
        self.seed, self.namespace = int(seed), str(namespace)

    def key(self, schedule, domain):
        message = f"CE-Gait/{self.seed}/{self.namespace}/{schedule}/{domain}"
        return hashlib.sha256(message.encode()).digest()[:16]

    def vectors(self, schedule, domains, device="cpu"):
        return key_tensor([self.key(schedule, d) for d in domains], device)


def new_domain_key():
    return secrets.token_bytes(16)

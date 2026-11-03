import numpy as np
import torch


def quantize(values, bits=4, straight_through=True):
    if not 1 <= bits <= 8:
        raise ValueError("bits must be between 1 and 8")
    levels = 2 ** bits - 1
    continuous = values.clamp(-1, 1)
    codes = torch.round((continuous + 1) * (levels / 2))
    discrete = codes * (2 / levels) - 1
    exposed = continuous + (discrete - continuous).detach() if straight_through else discrete
    return exposed, codes.to(torch.uint8)


def pack_codes(codes, bits=4):
    """MSB-first payload; two 4-bit values per byte."""
    a = np.asarray(codes, dtype=np.int64)
    if not 1 <= bits <= 8 or a.ndim == 0 or np.any((a < 0) | (a >= 2 ** bits)):
        raise ValueError("Invalid codes or bit width")
    if a.shape[-1] * bits % 8:
        raise ValueError("Payload must contain whole bytes")
    expanded = ((a[..., None] >> np.arange(bits - 1, -1, -1)) & 1).astype(np.uint8)
    return np.packbits(expanded.reshape(*a.shape[:-1], -1), axis=-1, bitorder="big")


def unpack_codes(payload, bits=4):
    a = np.asarray(payload, dtype=np.uint8)
    if not 1 <= bits <= 8 or a.shape[-1] * 8 % bits:
        raise ValueError("Invalid payload size")
    bits_array = np.unpackbits(a, axis=-1, bitorder="big").reshape(*a.shape[:-1], -1, bits)
    return (bits_array * (2 ** np.arange(bits - 1, -1, -1))).sum(-1).astype(np.uint8)

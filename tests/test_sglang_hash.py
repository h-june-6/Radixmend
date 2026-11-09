"""Hit-time hash baseline digests (radixmend.sglang_hash), CPU only, no SGLang."""
import types

import torch

from radixmend import sglang_hash as sh

L, N, H, D = 4, 64, 2, 16


def pool(seed=0):
    g = torch.Generator().manual_seed(seed)
    p = types.SimpleNamespace(
        k_buffer=[torch.randn(N, H, D, generator=g).to(torch.bfloat16) for _ in range(L)],
        v_buffer=[torch.randn(N, H, D, generator=g).to(torch.bfloat16) for _ in range(L)])
    sh._state["pools"].clear()
    sh._state["weights"] = None
    sh.attach_pool(p)
    return p


def write_all(p, loc):
    for lid in range(L):
        sh.store(p, lid, loc)


def test_store_matches_recompute_and_layer0_resets():
    p = pool()
    loc = torch.tensor([3, 7, 8, 40])
    write_all(p, loc)
    assert sh.verify(p, loc) == 0
    first = p.rm_digest[loc].clone()
    write_all(p, loc)  # slot reuse: layer 0 must reset, not accumulate
    assert torch.equal(p.rm_digest[loc], first)


def test_every_single_bit_flip_is_detected():
    p = pool(1)
    loc = torch.arange(10, 20)
    write_all(p, loc)
    for kind in ("k", "v"):
        for lid in (0, L - 1):
            for bit in range(16):
                buf = (p.k_buffer if kind == "k" else p.v_buffer)[lid].view(torch.int16)
                buf[13, 1, 5] ^= (1 << bit) if bit < 15 else -(1 << 15)
                assert sh.verify(p, loc) == 1, (kind, lid, bit)
                buf[13, 1, 5] ^= (1 << bit) if bit < 15 else -(1 << 15)
    assert sh.verify(p, loc) == 0


def test_flip_outside_checked_slots_is_not_seen():
    p = pool(2)
    loc = torch.arange(0, 8)
    write_all(p, torch.arange(0, 16))
    p.v_buffer[2].view(torch.int16)[12, 0, 0] ^= 1
    assert sh.verify(p, loc) == 0
    assert sh.verify(p, torch.arange(0, 16)) == 1

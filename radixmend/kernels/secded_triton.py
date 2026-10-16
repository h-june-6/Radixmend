"""Triton kernels for module A: encode-at-store, check-on-load, scrub.

These are standalone reference kernels with the same arithmetic the patched
attention kernels will use, so they can be validated on CPU with
``TRITON_INTERPRET=1`` before any GPU work:

* ``encode_kernel``   checkwords for freshly written K or V rows
                      (to be fused into the KV store kernel)
* ``check_load_kernel`` gathers vectors by slot like an attention kernel,
                      computes the syndrome, corrects one flipped bit in
                      registers, and appends non-clean vectors to an error log
* ``scrub_kernel``    re-verifies logged vectors and writes the correction back
                      to memory (data element or checkword)

KV elements are handled as raw 16-bit patterns (bf16/fp16 viewed as int16).
The checkword layout and decision rules are the ones in radixmend.secded_vec.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from ..secded_vec import VecCode

LOG_FIELDS = 4  # slot, head, status, syndrome


@triton.jit
def _xor(a, b):
    return a ^ b


@triton.jit
def _par16(v):
    v = v ^ (v >> 8)
    v = v ^ (v >> 4)
    v = v ^ (v >> 2)
    v = v ^ (v >> 1)
    return v & 1


@triton.jit
def _data_terms(x, e_lab, bmask_ptr, B: tl.constexpr):
    """P, S_e, S_b of one vector x[D] (int32 holding 16-bit patterns)."""
    par = _par16(x)
    p = tl.sum(par, axis=0) & 1
    se = tl.reduce(tl.where(par == 1, e_lab, 0), 0, _xor)
    sb = p * 0
    for k in tl.static_range(B):
        m = tl.load(bmask_ptr + k)
        sb = sb | ((tl.sum(_par16(x & m), axis=0) & 1) << k)
    return p, se, sb


@triton.jit
def encode_kernel(
    src_ptr, cw_ptr, elab_ptr, bmask_ptr,
    stride_n, stride_h,
    D: tl.constexpr, E: tl.constexpr, B: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    x = tl.load(src_ptr + n * stride_n + h * stride_h + d).to(tl.int32) & 0xFFFF
    e_lab = tl.load(elab_ptr + d)
    p, se, sb = _data_terms(x, e_lab, bmask_ptr, B)
    q = _par16(se) ^ _par16(sb)
    cw = p | (se << 1) | (sb << (1 + E)) | (q << (1 + E + B))
    tl.store(cw_ptr + n * tl.num_programs(1) + h, cw.to(tl.int16))


@triton.jit
def _classify(x, cw, e_lab, bmask_ptr, inv_e_ptr, inv_b_ptr, E: tl.constexpr, B: tl.constexpr):
    """Return (status, elem, bit, syndrome) for one vector and its stored checkword."""
    p1, se1, sb1 = _data_terms(x, e_lab, bmask_ptr, B)
    p0 = cw & 1
    se0 = (cw >> 1) & ((1 << E) - 1)
    sb0 = (cw >> (1 + E)) & ((1 << B) - 1)
    q0 = (cw >> (1 + E + B)) & 1
    dp = p1 ^ p0
    dse = se1 ^ se0
    dsb = sb1 ^ sb0
    dq = p1 ^ p0 ^ _par16(se0) ^ _par16(sb0) ^ q0
    ds = (dse << B) | dsb
    syn = dp | (dse << 1) | (dsb << (1 + E)) | (dq << (1 + E + B))
    elem = tl.load(inv_e_ptr + dse)
    bit = tl.load(inv_b_ptr + dsb)
    ds_unit = (ds != 0) & ((ds & (ds - 1)) == 0)
    is_data = (dq == 1) & (dp == 1) & (ds != 0) & (elem >= 0) & (bit >= 0)
    is_check = (dq == 1) & (((dp == 1) & (ds == 0)) | ((dp == 0) & ds_unit) | ((dp == 0) & (ds == 0)))
    status = tl.where(syn == 0, 0, tl.where(is_data, 1, tl.where(is_check, 2, 3)))
    return status, elem, bit, syn


@triton.jit
def check_load_kernel(
    buf_ptr, cw_ptr, loc_ptr, out_ptr, status_ptr,
    elab_ptr, bmask_ptr, inv_e_ptr, inv_b_ptr,
    log_ptr, log_count_ptr, log_cap,
    stride_s, stride_h, H,
    D: tl.constexpr, E: tl.constexpr, B: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    slot = tl.load(loc_ptr + n).to(tl.int64)
    raw = tl.load(buf_ptr + slot * stride_s + h * stride_h + d)
    x = raw.to(tl.int32) & 0xFFFF
    cw = tl.load(cw_ptr + slot * H + h).to(tl.int32) & 0xFFFF
    e_lab = tl.load(elab_ptr + d)
    status, elem, bit, syn = _classify(x, cw, e_lab, bmask_ptr, inv_e_ptr, inv_b_ptr, E, B)
    # Register-level correction: the attention math sees the fixed value,
    # memory is left for the scrub kernel.
    fix = tl.where((status == 1) & (d == elem), 1 << bit, 0)
    x = x ^ fix
    tl.store(out_ptr + (n * H + h) * D + d, x.to(tl.int16))
    tl.store(status_ptr + n * H + h, status.to(tl.int8))
    if status != 0:
        i = tl.atomic_add(log_count_ptr, 1)
        if i < log_cap:
            tl.store(log_ptr + i * 4 + 0, slot.to(tl.int32))
            tl.store(log_ptr + i * 4 + 1, h)
            tl.store(log_ptr + i * 4 + 2, status)
            tl.store(log_ptr + i * 4 + 3, syn)


@triton.jit
def scrub_kernel(
    buf_ptr, cw_ptr, log_ptr, result_ptr,
    elab_ptr, bmask_ptr, inv_e_ptr, inv_b_ptr,
    stride_s, stride_h, H,
    D: tl.constexpr, E: tl.constexpr, B: tl.constexpr,
):
    i = tl.program_id(0)
    slot = tl.load(log_ptr + i * 4 + 0).to(tl.int64)
    h = tl.load(log_ptr + i * 4 + 1)
    d = tl.arange(0, D)
    base = buf_ptr + slot * stride_s + h * stride_h
    x = tl.load(base + d).to(tl.int32) & 0xFFFF
    cw = tl.load(cw_ptr + slot * H + h).to(tl.int32) & 0xFFFF
    e_lab = tl.load(elab_ptr + d)
    # Re-verify: several readers may have logged the same vector, and a 3+ bit
    # pattern must not be written back unless it still decodes as a single.
    status, elem, bit, syn = _classify(x, cw, e_lab, bmask_ptr, inv_e_ptr, inv_b_ptr, E, B)
    x = x ^ tl.where((status == 1) & (d == elem), 1 << bit, 0)
    if status == 1:
        tl.store(base + d, x.to(tl.int16), mask=d == elem)
    if (status == 1) | (status == 2):
        p, se, sb = _data_terms(x, e_lab, bmask_ptr, B)
        q = _par16(se) ^ _par16(sb)
        new_cw = p | (se << 1) | (sb << (1 + E)) | (q << (1 + E + B))
        tl.store(cw_ptr + slot * H + h, new_cw.to(tl.int16))
    tl.store(result_ptr + i, status)


# ---- host side -------------------------------------------------------------------


class TritonVecCode:
    """Device-side tables for a VecCode, plus launch helpers."""

    def __init__(self, code: VecCode, device="cpu"):
        if code.elem_bits != 16:
            raise ValueError("kernels handle 16-bit elements")
        self.code = code
        self.D, self.E, self.B = code.n_elem, code.e_bits, code.b_bits
        inv_e = np.full(1 << self.E, -1, np.int32)
        inv_e[code.e_labels] = np.arange(code.n_elem)
        inv_b = np.full(1 << self.B, -1, np.int32)
        inv_b[code.b_labels] = np.arange(code.elem_bits)
        bmask = code.b_masks.astype(np.int64)
        self.elab = torch.tensor(code.e_labels, dtype=torch.int32, device=device)
        self.bmask = torch.tensor(bmask, dtype=torch.int32, device=device)
        self.inv_e = torch.tensor(inv_e, device=device)
        self.inv_b = torch.tensor(inv_b, device=device)

    @staticmethod
    def _i16(t):
        return t.view(torch.int16) if t.dtype != torch.int16 else t

    def encode(self, kv: torch.Tensor) -> torch.Tensor:
        """kv[N, H, D] (bf16/fp16/int16) -> checkwords[N, H] int16."""
        x = self._i16(kv)
        assert x.stride(-1) == 1 and x.shape[-1] == self.D
        n, h, _ = x.shape
        cw = torch.empty((n, h), dtype=torch.int16, device=x.device)
        encode_kernel[(n, h)](x, cw, self.elab, self.bmask, x.stride(0), x.stride(1), D=self.D, E=self.E, B=self.B)
        return cw

    def check_load(self, buf: torch.Tensor, cw: torch.Tensor, loc: torch.Tensor, log_cap: int = 1024):
        """Gather buf[loc] with check/correct. Returns (values, status, log, log_count)."""
        x = self._i16(buf)
        _, h, _ = x.shape
        n = loc.numel()
        out = torch.empty((n, h, self.D), dtype=torch.int16, device=x.device)
        status = torch.empty((n, h), dtype=torch.int8, device=x.device)
        log = torch.zeros((log_cap, 4), dtype=torch.int32, device=x.device)
        count = torch.zeros(1, dtype=torch.int32, device=x.device)
        check_load_kernel[(n, h)](
            x, cw, loc, out, status, self.elab, self.bmask, self.inv_e, self.inv_b,
            log, count, log_cap, x.stride(0), x.stride(1), h, D=self.D, E=self.E, B=self.B,
        )
        return out.view(buf.dtype), status, log, count

    def scrub(self, buf: torch.Tensor, cw: torch.Tensor, log: torch.Tensor, count: int) -> torch.Tensor:
        """Fix logged vectors in memory. Returns the re-verified status per entry."""
        x = self._i16(buf)
        _, h, _ = x.shape
        result = torch.zeros(count, dtype=torch.int32, device=x.device)
        if count:
            scrub_kernel[(count,)](
                x, cw, log, result, self.elab, self.bmask, self.inv_e, self.inv_b,
                x.stride(0), x.stride(1), h, D=self.D, E=self.E, B=self.B,
            )
        return result

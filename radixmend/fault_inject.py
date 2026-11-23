"""Software fault injector for KV buffers and slot-index tensors.

Works on plain torch tensors, so it runs the same on CPU (tests) and GPU
(experiments). The SGLang-facing helpers only rely on a few attributes:

* KV pool (MHATokenToKVPool, NHD layout): ``k_buffer`` / ``v_buffer`` are lists
  indexed by local layer, each ``[num_slots, head_num, head_dim]``.
* Radix tree: the default in SGLang v0.5.22 is UnifiedRadixCache, whose Python
  tree core (``SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python``) has nodes with
  ``children`` and ``component_data[FULL]`` holding ``value`` / ``lock_ref``.
  The legacy TreeNode with ``value`` / ``lock_ref`` fields is also accepted.

Every injected flip is returned as a record (and can be appended to a JSONL
log) holding the old and new raw bits, so a campaign can be replayed and its
effect on values analysed offline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch

_INT_VIEW = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def int_view(t: torch.Tensor) -> torch.Tensor:
    """Same-storage integer view with the element width of t."""
    if not t.dtype.is_floating_point and t.dtype not in (torch.bool,):
        return t
    return t.view(_INT_VIEW[t.element_size()])


def _bit_value(bit: int, dtype: torch.dtype) -> int:
    """1 << bit as a value representable in a signed dtype of that width."""
    width = torch.iinfo(dtype).bits
    v = 1 << bit
    if torch.iinfo(dtype).min < 0 and bit == width - 1:
        v -= 1 << width
    return v


def flip_bits_(t: torch.Tensor, flat_index: Sequence[int], bits: Sequence[int]) -> list[tuple[int, int]]:
    """Flip ``bits[i]`` of element ``flat_index[i]`` in place; return (old, new) raw ints."""
    iv = int_view(t).reshape(-1) if t.is_contiguous() else None
    if iv is None:
        raise ValueError("flip_bits_ needs a contiguous tensor")
    width = torch.iinfo(iv.dtype).bits
    out = []
    for idx, bit in zip(flat_index, bits):
        if not 0 <= bit < width:
            raise ValueError(f"bit {bit} out of range for {width}-bit elements")
        old = int(iv[idx].item())
        iv[idx] = old ^ _bit_value(bit, iv.dtype)
        out.append((old, int(iv[idx].item())))
    return out


# ---- KV faults ------------------------------------------------------------------


@dataclass(frozen=True)
class KVFault:
    layer: int  # local layer index into k_buffer / v_buffer
    kind: str  # "k" or "v"
    slot: int
    head: int
    elem: int
    bit: int  # 0 = mantissa LSB of bf16, 15 = sign


@dataclass
class FlipRecord:
    target: str
    fault: dict
    old_raw: int
    new_raw: int
    old_value: float | None = None
    new_value: float | None = None


def _as_float(raw: int, dtype: torch.dtype) -> float | None:
    if not dtype.is_floating_point:
        return None
    iv = torch.tensor([raw], dtype=_INT_VIEW[torch.empty((), dtype=dtype).element_size()])
    return float(iv.view(dtype).float().item())


def inject_kv(k_buffer: Sequence[torch.Tensor], v_buffer: Sequence[torch.Tensor], faults: Iterable[KVFault]) -> list[FlipRecord]:
    """Apply all flips of each buffer with one write, so a multi-bit fault lands at once.

    Flipping bit by bit (one read-modify-write per bit) let a decode step that ran in
    between (overlap scheduling) see, correct and write back the first flip as a single-bit
    error, then the second one a step later: a 2-bit fault measured as two 1-bit faults.
    """
    faults = list(faults)
    groups: dict[tuple[str, int], list[int]] = {}
    for i, f in enumerate(faults):
        groups.setdefault((f.kind, f.layer), []).append(i)
    recs: dict[int, tuple[int, int]] = {}
    for (kind, layer), ids in groups.items():
        buf = (k_buffer if kind == "k" else v_buffer)[layer]
        _, heads, dim = buf.shape
        iv = int_view(buf).reshape(-1)
        width = torch.iinfo(iv.dtype).bits
        flat_of = {}
        for i in ids:
            f = faults[i]
            if not 0 <= f.bit < width:
                raise ValueError(f"bit {f.bit} out of range for {width}-bit elements")
            flat_of[i] = (f.slot * heads + f.head) * dim + f.elem
        flats = sorted(set(flat_of.values()))
        idx = torch.tensor(flats, dtype=torch.int64, device=iv.device)
        cur = dict(zip(flats, iv[idx].tolist()))
        for i in ids:  # per-flip (old, new) in order, as if applied one by one
            x = flat_of[i]
            o = cur[x]
            n = _signed((o ^ (1 << faults[i].bit)) & ((1 << width) - 1), width)
            recs[i] = (o, n)
            cur[x] = n
        # one write for every flip of this buffer
        iv[idx] = torch.tensor([cur[x] for x in flats], dtype=iv.dtype, device=iv.device)
    records = []
    for i, f in enumerate(faults):
        buf = (k_buffer if f.kind == "k" else v_buffer)[f.layer]
        old, new = recs[i]
        records.append(
            FlipRecord("kv", asdict(f), old, new, _as_float(old, buf.dtype), _as_float(new, buf.dtype))
        )
    return records


def _signed(v: int, width: int) -> int:
    return v - (1 << width) if v >= 1 << (width - 1) else v


def sample_kv_faults(
    gen: torch.Generator,
    slots: Sequence[int],
    layers: Sequence[int],
    head_num: int,
    head_dim: int,
    *,
    kinds: Sequence[str] = ("k", "v"),
    bits: Sequence[int] | None = None,
    elem_bits: int = 16,
    flips_per_vector: int = 1,
    vectors: int = 1,
    bits_each: Sequence[int] | None = None,
) -> list[KVFault]:
    """Pick ``vectors`` random (layer, kind, slot, head) vectors and flip
    ``flips_per_vector`` distinct bits inside each one.

    With ``bits`` given, every flip uses a bit position from that list (e.g. a
    single position for a per-bit-position sweep). With ``bits_each`` given,
    flip i of each vector uses bit ``bits_each[i]`` on a distinct element
    (e.g. ``[14, 7]`` for a 2-bit error at different bit positions).
    """
    if bits_each is not None:
        flips_per_vector = len(bits_each)

    def pick(seq):
        return seq[int(torch.randint(len(seq), (1,), generator=gen))]

    faults = []
    for _ in range(vectors):
        layer, kind, slot = pick(list(layers)), pick(list(kinds)), pick(list(slots))
        head = int(torch.randint(head_num, (1,), generator=gen))
        chosen: set[tuple[int, int]] = set()
        if bits_each is not None:
            elems = torch.randperm(head_dim, generator=gen)[: len(bits_each)].tolist()
            chosen = {(int(e), int(b)) for e, b in zip(elems, bits_each)}
        while len(chosen) < flips_per_vector:
            elem = int(torch.randint(head_dim, (1,), generator=gen))
            bit = pick(list(bits)) if bits is not None else int(torch.randint(elem_bits, (1,), generator=gen))
            chosen.add((elem, bit))
        faults += [KVFault(layer, kind, slot, head, e, b) for e, b in sorted(chosen)]
    return faults


# ---- index faults -----------------------------------------------------------------


@dataclass(frozen=True)
class IndexFault:
    target: str  # e.g. "req_to_token", "node.value", "free_pages", "seq_lens"
    flat_index: int
    bit: int


def inject_index(tensors: dict[str, torch.Tensor], faults: Iterable[IndexFault]) -> list[FlipRecord]:
    records = []
    for f in faults:
        (old, new), = flip_bits_(tensors[f.target], [f.flat_index], [f.bit])
        records.append(FlipRecord("index", asdict(f), old, new))
    return records


def sample_index_faults(
    gen: torch.Generator, name: str, tensor: torch.Tensor, n: int, bits: Sequence[int] | None = None,
    valid_flat: Sequence[int] | None = None,
) -> list[IndexFault]:
    """Random index flips; ``valid_flat`` restricts to live entries (e.g. used req rows)."""
    width = torch.iinfo(int_view(tensor).dtype).bits
    numel = tensor.numel()
    out = []
    for _ in range(n):
        if valid_flat is not None:
            idx = valid_flat[int(torch.randint(len(valid_flat), (1,), generator=gen))]
        else:
            idx = int(torch.randint(numel, (1,), generator=gen))
        bit = bits[int(torch.randint(len(bits), (1,), generator=gen))] if bits else int(torch.randint(width, (1,), generator=gen))
        out.append(IndexFault(name, idx, bit))
    return out


# ---- radix tree targeting -----------------------------------------------------------

FULL_COMPONENT = 0  # ComponentType.FULL in sglang.srt.mem_cache.unified_cache


def node_value(node):
    if hasattr(node, "component_data"):
        return node.component_data[FULL_COMPONENT].value
    return node.value


def node_lock_ref(node) -> int:
    if hasattr(node, "component_data"):
        return node.component_data[FULL_COMPONENT].lock_ref
    return node.lock_ref



def iter_nodes(root) -> Iterator[tuple[object, int]]:
    """(node, depth) for every non-root node with KV on device, depth 1 = child of root."""
    stack = [(c, 1) for c in root.children.values()]
    while stack:
        node, d = stack.pop()
        if node_value(node) is not None:
            yield node, d
        stack.extend((c, d + 1) for c in node.children.values())


def count_descendants(node) -> int:
    return sum(1 + count_descendants(c) for c in node.children.values())


def first_level_nodes(root) -> list:
    """Children of the root. SGLang's root holds no KV, so these hold the most shared prefix."""
    return [n for n, d in iter_nodes(root) if d == 1]


def sink_slots(root, n_tokens: int = 4) -> list[int]:
    """KV slots of the first ``n_tokens`` tokens under each first-level node (BOS / sink)."""
    out = []
    for node in first_level_nodes(root):
        out += [int(s) for s in node_value(node)[:n_tokens].tolist()]
    return out


def node_slots(node) -> list[int]:
    # low 32 bits: module B keeps its code in the high bits of node.value
    return [int(s) & 0xFFFFFFFF for s in node_value(node).tolist()]


def nodes_by_sharing(root, top: int | None = None) -> list[tuple[object, int, int, int]]:
    """Nodes sorted by (lock_ref, descendants), most shared first: (node, depth, lock_ref, desc)."""
    rows = [(n, d, node_lock_ref(n), count_descendants(n)) for n, d in iter_nodes(root)]
    rows.sort(key=lambda r: (r[2], r[3]), reverse=True)
    return rows[:top] if top else rows


# ---- logging -------------------------------------------------------------------


def append_log(path: str | Path, records: Iterable[FlipRecord], **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for r in records:
            fh.write(json.dumps({**meta, **asdict(r)}) + "\n")


def _set_raw(t: torch.Tensor, flat: int, raw: int) -> None:
    # restore the saved value instead of flipping again: a protection module may
    # already have corrected the element in place (scrub), and a second flip
    # would then corrupt it
    int_view(t).reshape(-1)[flat] = raw


def revert(records: Sequence[FlipRecord], *, k_buffer=None, v_buffer=None, tensors=None) -> None:
    """Undo flips (in reverse order) so one model run can host many trials."""
    for r in reversed(records):
        f = r.fault
        if r.target == "kv":
            buf = (k_buffer if f["kind"] == "k" else v_buffer)[f["layer"]]
            _, heads, dim = buf.shape
            _set_raw(buf, (f["slot"] * heads + f["head"]) * dim + f["elem"], r.old_raw)
        else:
            _set_raw(tensors[f["target"]], f["flat_index"], r.old_raw)

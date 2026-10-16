"""Reference SECDED codec for KV head vectors (module A).

One codeword covers ``n_elem`` elements of ``elem_bits`` bits each (by default
one bf16 head vector: 128 x 16 = 2048 data bits). The checkword is

    bit 0               P    parity of all data bits
    bits 1 .. E         S_e  XOR of e-labels of elements with odd parity
    bits E+1 .. E+B     S_b  XOR of b-labels of every set data bit
    bit E+B+1           Q    parity over data, P, S_e and S_b

so 128 elements give E=8, B=5 and a 15-bit checkword stored in a uint16.

Decoding computes the syndrome (dP, dS_e, dS_b, dQ), where dQ is the overall
parity of the received codeword, and applies the rules from the design doc:

    dQ=1, dP=1, dS!=0, both labels valid   -> data bit (e, b), correct it
    dQ=1, dP=1, dS=0                        -> P bit error
    dQ=1, dP=0, dS is a unit vector         -> one S bit error
    dQ=1, dP=0, dS=0                        -> Q bit error
    dQ=0, syndrome != 0                     -> double error, detected
    anything else                           -> 3+ bits, detected

Check-bit errors leave the data untouched and only rewrite the checkword.
Everything is vectorised with numpy over arbitrary leading dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .labels import bits_needed, identity_labels, keyed_labels

CLEAN = 0
CORRECTED_DATA = 1
CORRECTED_CHECK = 2
DETECTED = 3

STATUS_NAMES = {
    CLEAN: "clean",
    CORRECTED_DATA: "corrected_data",
    CORRECTED_CHECK: "corrected_check",
    DETECTED: "detected",
}

DEFAULT_KEY = b"radixmend-default-key-not-secret"


def _parity(a: np.ndarray) -> np.ndarray:
    return (np.bitwise_count(a) & 1).astype(np.int64)


@dataclass
class VecCode:
    n_elem: int = 128
    elem_bits: int = 16
    key: bytes | None = DEFAULT_KEY
    # Optional wider label spaces. Unused labels turn more 3+ bit errors into
    # detections, at the cost of more check bits (the checkword stays <= 16).
    e_bits_override: int | None = None
    b_bits_override: int | None = None

    e_bits: int = field(init=False)
    b_bits: int = field(init=False)
    e_labels: np.ndarray = field(init=False, repr=False)
    b_labels: np.ndarray = field(init=False, repr=False)
    b_masks: np.ndarray = field(init=False, repr=False)
    lut_status: np.ndarray = field(init=False, repr=False)
    lut_pos: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.e_bits = self.e_bits_override or bits_needed(self.n_elem)
        self.b_bits = self.b_bits_override or bits_needed(self.elem_bits)
        if self.e_bits < bits_needed(self.n_elem) or self.b_bits < bits_needed(self.elem_bits):
            raise ValueError("label width too small for the number of labels")
        if self.check_bits > 16:
            raise ValueError(f"checkword needs {self.check_bits} bits (> 16)")
        if self.key is None:
            self.e_labels = identity_labels(self.n_elem)
            self.b_labels = identity_labels(self.elem_bits)
        else:
            self.e_labels = keyed_labels(self.n_elem, self.e_bits, self.key, b"elem")
            self.b_labels = keyed_labels(self.elem_bits, self.b_bits, self.key, b"bit")
        # b_masks[k] selects the in-element bit positions whose b-label has bit k
        # set, so S_b bit k = parity(sum_i popcount(x_i & b_masks[k])). This is
        # the popcount form the attention kernel uses.
        self.b_masks = np.array(
            [
                sum(1 << j for j in range(self.elem_bits) if (self.b_labels[j] >> k) & 1)
                for k in range(self.b_bits)
            ],
            dtype=np.int64,
        )
        self._build_lut()

    # ---- layout -------------------------------------------------------------
    @property
    def check_bits(self) -> int:
        return 2 + self.e_bits + self.b_bits

    @property
    def data_bits(self) -> int:
        return self.n_elem * self.elem_bits

    @property
    def codeword_bits(self) -> int:
        return self.data_bits + self.check_bits

    @property
    def se_shift(self) -> int:
        return 1

    @property
    def sb_shift(self) -> int:
        return 1 + self.e_bits

    @property
    def q_shift(self) -> int:
        return 1 + self.e_bits + self.b_bits

    def pack(self, p, se, sb, q):
        return (
            np.asarray(p, np.int64)
            | (np.asarray(se, np.int64) << self.se_shift)
            | (np.asarray(sb, np.int64) << self.sb_shift)
            | (np.asarray(q, np.int64) << self.q_shift)
        )

    def unpack(self, cw):
        cw = np.asarray(cw, np.int64)
        p = cw & 1
        se = (cw >> self.se_shift) & ((1 << self.e_bits) - 1)
        sb = (cw >> self.sb_shift) & ((1 << self.b_bits) - 1)
        q = (cw >> self.q_shift) & 1
        return p, se, sb, q

    # ---- encode ---------------------------------------------------------------
    def _data_terms(self, x: np.ndarray):
        """P, S_e, S_b of the data, popcount formulation (as in the kernel)."""
        x = np.asarray(x).astype(np.int64) & ((1 << self.elem_bits) - 1)
        if x.shape[-1] != self.n_elem:
            raise ValueError(f"last dim {x.shape[-1]} != n_elem {self.n_elem}")
        elem_par = _parity(x)
        p = elem_par.sum(-1) & 1
        se = np.bitwise_xor.reduce(np.where(elem_par == 1, self.e_labels, 0), axis=-1)
        sb = np.zeros(x.shape[:-1], np.int64)
        for k in range(self.b_bits):
            bit = np.bitwise_count(x & self.b_masks[k]).astype(np.int64).sum(-1) & 1
            sb |= bit << k
        return p, se, sb

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Checkword (uint16) for each codeword in x[..., n_elem]."""
        p, se, sb = self._data_terms(x)
        q = _parity(se) ^ _parity(sb)  # P cancels against the data parity
        return self.pack(p, se, sb, q).astype(np.uint16)

    def encode_bitloop(self, x: np.ndarray) -> np.ndarray:
        """Literal definition (XOR of labels over set bits); slow, for cross-checks."""
        x = np.asarray(x).astype(np.int64)
        flat = x.reshape(-1, self.n_elem)
        out = np.zeros(flat.shape[0], np.int64)
        for r, row in enumerate(flat):
            p = se = sb = 0
            for i, v in enumerate(row):
                for j in range(self.elem_bits):
                    if (int(v) >> j) & 1:
                        p ^= 1
                        se ^= int(self.e_labels[i])
                        sb ^= int(self.b_labels[j])
            q = (bin(se).count("1") + bin(sb).count("1") + p + p) & 1
            out[r] = int(self.pack(p, se, sb, q))
        return out.reshape(x.shape[:-1]).astype(np.uint16)

    # ---- syndrome & decode ----------------------------------------------------
    def syndrome(self, x: np.ndarray, cw: np.ndarray) -> np.ndarray:
        """Packed syndrome (dP, dS_e, dS_b, dQ) in the checkword layout."""
        p1, se1, sb1 = self._data_terms(x)
        p0, se0, sb0, q0 = self.unpack(cw)
        dq = p1 ^ p0 ^ _parity(se0) ^ _parity(sb0) ^ q0  # overall parity
        return self.pack(p1 ^ p0, se1 ^ se0, sb1 ^ sb0, dq)

    def classify(self, s: int):
        """Apply the design-doc decision rules to one packed syndrome.

        Returns (status, codeword_position or -1). Data bit (i, j) is position
        i * elem_bits + j; check bits follow the data in checkword bit order.
        """
        dp, dse, dsb, dq = (int(v) for v in self.unpack(s))
        ds = (dse << self.b_bits) | dsb
        if s == 0:
            return CLEAN, -1
        if dq == 0:
            return DETECTED, -1
        if dp == 1 and ds != 0:
            ei = np.flatnonzero(self.e_labels == dse)
            bj = np.flatnonzero(self.b_labels == dsb)
            if len(ei) == 1 and len(bj) == 1:
                return CORRECTED_DATA, int(ei[0]) * self.elem_bits + int(bj[0])
            return DETECTED, -1
        if dp == 1 and ds == 0:
            return CORRECTED_CHECK, self.data_bits + 0
        if dp == 0 and ds != 0 and (ds & (ds - 1)) == 0:
            if dse:
                k = dse.bit_length() - 1
                return CORRECTED_CHECK, self.data_bits + self.se_shift + k
            k = dsb.bit_length() - 1
            return CORRECTED_CHECK, self.data_bits + self.sb_shift + k
        if dp == 0 and ds == 0:
            return CORRECTED_CHECK, self.data_bits + self.q_shift
        return DETECTED, -1

    def _build_lut(self):
        n = 1 << self.check_bits
        self.lut_status = np.empty(n, np.int8)
        self.lut_pos = np.empty(n, np.int32)
        for s in range(n):
            st, pos = self.classify(s)
            self.lut_status[s] = st
            self.lut_pos[s] = pos

    def decode(self, x: np.ndarray, cw: np.ndarray):
        """Check and correct.

        Returns (x_fixed, cw_fixed, status, pos). Data are corrected only for
        CORRECTED_DATA; the checkword is re-encoded for both corrected states.
        For DETECTED the inputs are returned unchanged.
        """
        x = np.asarray(x).astype(np.uint16)
        cw = np.asarray(cw).astype(np.uint16)
        s = self.syndrome(x, cw)
        status = self.lut_status[s]
        pos = self.lut_pos[s]
        x_fixed = x.copy()
        fix = status == CORRECTED_DATA
        if fix.any():
            idx = np.nonzero(fix)
            p = pos[idx]
            elem, bit = p // self.elem_bits, p % self.elem_bits
            x_fixed[idx + (elem,)] ^= (1 << bit).astype(np.uint16)
        cw_fixed = cw.copy()
        recode = (status == CORRECTED_DATA) | (status == CORRECTED_CHECK)
        if recode.any():
            cw_fixed[recode] = self.encode(x_fixed[recode])
        return x_fixed, cw_fixed, status, pos

    # ---- parity-check matrix view ----------------------------------------------
    def columns(self) -> np.ndarray:
        """Syndrome of a single flip at each codeword position (the H matrix columns)."""
        cols = np.empty(self.codeword_bits, np.int64)
        e = np.repeat(self.e_labels, self.elem_bits)
        b = np.tile(self.b_labels, self.n_elem)
        cols[: self.data_bits] = self.pack(1, e, b, 1)
        for k in range(self.check_bits):
            # A flip of any stored check bit flips the overall parity too.
            cols[self.data_bits + k] = (1 << k) | (1 << self.q_shift)
        cols[self.data_bits + self.q_shift] = 1 << self.q_shift
        return cols


def split_code(head_dim: int, splits: int, elem_bits: int = 16, key=DEFAULT_KEY) -> VecCode:
    """Codec for one of ``splits`` equal codewords covering a head vector."""
    if head_dim % splits:
        raise ValueError("head_dim must be divisible by splits")
    return VecCode(n_elem=head_dim // splits, elem_bits=elem_bits, key=key)


def encode_heads(code: VecCode, x: np.ndarray, splits: int = 1) -> np.ndarray:
    """x[..., head_dim] -> checkwords[..., splits]."""
    return code.encode(x.reshape(*x.shape[:-1], splits, code.n_elem))


def decode_heads(code: VecCode, x: np.ndarray, cw: np.ndarray, splits: int = 1):
    xs = x.reshape(*x.shape[:-1], splits, code.n_elem)
    xf, cwf, st, pos = code.decode(xs, cw)
    return xf.reshape(x.shape), cwf, st, pos

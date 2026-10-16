import torch

from radixmend import fault_inject as fi


def make_kv(layers=2, slots=16, heads=4, dim=8):
    g = torch.Generator().manual_seed(0)
    k = [torch.randn(slots, heads, dim, generator=g).to(torch.bfloat16) for _ in range(layers)]
    v = [torch.randn(slots, heads, dim, generator=g).to(torch.bfloat16) for _ in range(layers)]
    return k, v


def test_flip_and_revert_kv():
    k, v = make_kv()
    k0 = [t.clone() for t in k]
    faults = [fi.KVFault(1, "k", 5, 2, 3, 15), fi.KVFault(1, "k", 5, 2, 3, 0)]
    recs = fi.inject_kv(k, v, faults)
    assert recs[0].new_value == -recs[0].old_value  # sign bit
    diff = (k[1].view(torch.int16) ^ k0[1].view(torch.int16))
    assert diff[5, 2, 3].item() == (1 | -32768)
    assert int((diff != 0).sum()) == 1
    fi.revert(recs, k_buffer=k, v_buffer=v)
    assert all(torch.equal(a, b) for a, b in zip(k, k0))


def test_sampler_distinct_bits_in_one_vector():
    g = torch.Generator().manual_seed(1)
    f = fi.sample_kv_faults(g, slots=[3, 4], layers=[0], head_num=4, head_dim=128, flips_per_vector=3, vectors=10)
    assert len(f) == 30
    for i in range(0, 30, 3):
        grp = f[i : i + 3]
        assert len({(x.layer, x.kind, x.slot, x.head) for x in grp}) == 1
        assert len({(x.elem, x.bit) for x in grp}) == 3


def test_index_flip_int32_sign_bit():
    t = torch.arange(10, dtype=torch.int32)
    recs = fi.inject_index({"req_to_token": t}, [fi.IndexFault("req_to_token", 4, 31), fi.IndexFault("req_to_token", 7, 22)])
    assert t[4].item() == 4 - 2**31 and t[7].item() == 7 + 2**22
    fi.revert(recs, tensors={"req_to_token": t})
    assert torch.equal(t, torch.arange(10, dtype=torch.int32))


class Node:
    def __init__(self, value=None, lock_ref=0):
        self.children, self.value, self.lock_ref = {}, value, lock_ref


def test_tree_targeting():
    root = Node(value=[], lock_ref=1)
    sys_prompt = Node(torch.tensor([10, 11, 12, 13, 14, 15]), lock_ref=3)
    other = Node(torch.tensor([20, 21]), lock_ref=0)
    root.children = {1: sys_prompt, 2: other}
    a, b = Node(torch.tensor([30]), 1), Node(torch.tensor([40]), 2)
    sys_prompt.children = {3: a, 4: b}
    evicted = Node(None)
    other.children = {5: evicted}
    assert {id(n) for n in fi.first_level_nodes(root)} == {id(sys_prompt), id(other)}
    assert sorted(fi.sink_slots(root, 4)) == [10, 11, 12, 13, 20, 21]
    top = fi.nodes_by_sharing(root, 2)
    assert top[0][0] is sys_prompt and top[0][3] == 2 and top[1][0] is b


class Comp:
    def __init__(self, value=None, lock_ref=0):
        self.value, self.lock_ref = value, lock_ref


class UNode:
    def __init__(self, value=None, lock_ref=0):
        self.children = {}
        self.component_data = [Comp(value, lock_ref), Comp(), Comp()]


def test_unified_tree_nodes():
    root = UNode([], 1)
    a = UNode(torch.tensor([5, 6, 7, 8, 9]), 2)
    root.children = {1: a}
    assert fi.sink_slots(root, 2) == [5, 6]
    assert fi.nodes_by_sharing(root)[0][2] == 2

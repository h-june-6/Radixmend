"""Loaded at interpreter start when this folder is on PYTHONPATH.

Installs the RadixMend fault-injection hook in every SGLang process
(including the spawned scheduler) when RADIXMEND_CMD_DIR is set, and the
module A prototype (SECDED check inside triton decode attention) when
RADIXMEND_SECDED=1, module C recovery (retract + recompute on uncorrectable
errors) when RADIXMEND_RECOVER=1 as well, module D replicas of shared prefixes when
RADIXMEND_REPLICA=1, module B index protection when RADIXMEND_INDEX=1, and the hit-time hash baseline when
RADIXMEND_HASH=1.
Does nothing otherwise.
"""
import os
import sys

if os.environ.get("RADIXMEND_CMD_DIR"):
    try:
        from radixmend.sglang_hook import install

        install()
    except Exception as e:  # never block interpreter start-up
        print(f"[radixmend] hook not installed: {e!r}", file=sys.stderr)

if os.environ.get("RADIXMEND_INDEX") == "1":  # module B, before C and D so they see plain indices
    try:
        from radixmend.sglang_index import install as index_install

        index_install()
    except Exception as e:
        print(f"[radixmend] index protection not installed: {e!r}", file=sys.stderr)

if os.environ.get("RADIXMEND_SECDED") == "1":
    try:
        from radixmend.sglang_secded import install as secded_install

        secded_install()
    except Exception as e:
        print(f"[radixmend] secded not installed: {e!r}", file=sys.stderr)

    if os.environ.get("RADIXMEND_RECOVER") == "1":
        try:
            from radixmend.sglang_recover import install as recover_install

            recover_install()
        except Exception as e:
            print(f"[radixmend] recovery not installed: {e!r}", file=sys.stderr)

    if os.environ.get("RADIXMEND_REPLICA") == "1":  # module D, after C (its match wrapper runs first)
        try:
            from radixmend.sglang_replica import install as replica_install

            replica_install()
        except Exception as e:
            print(f"[radixmend] replicas not installed: {e!r}", file=sys.stderr)

if os.environ.get("RADIXMEND_HASH") == "1":
    try:
        from radixmend.sglang_hash import install as hash_install

        hash_install()
    except Exception as e:
        print(f"[radixmend] hash baseline not installed: {e!r}", file=sys.stderr)

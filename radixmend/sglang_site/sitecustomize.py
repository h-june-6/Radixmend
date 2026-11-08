"""Loaded at interpreter start when this folder is on PYTHONPATH.

Installs the RadixMend fault-injection hook in every SGLang process
(including the spawned scheduler) when RADIXMEND_CMD_DIR is set, and the
module A prototype (SECDED check inside triton decode attention) when
RADIXMEND_SECDED=1. Does nothing otherwise.
"""
import os
import sys

if os.environ.get("RADIXMEND_CMD_DIR"):
    try:
        from radixmend.sglang_hook import install

        install()
    except Exception as e:  # never block interpreter start-up
        print(f"[radixmend] hook not installed: {e!r}", file=sys.stderr)

if os.environ.get("RADIXMEND_SECDED") == "1":
    try:
        from radixmend.sglang_secded import install as secded_install

        secded_install()
    except Exception as e:
        print(f"[radixmend] secded not installed: {e!r}", file=sys.stderr)

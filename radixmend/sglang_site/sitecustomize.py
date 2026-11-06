"""Loaded at interpreter start when this folder is on PYTHONPATH.

Installs the RadixMend fault-injection hook in every SGLang process
(including the spawned scheduler) when RADIXMEND_CMD_DIR is set. Does nothing
otherwise.
"""
import os
import sys

if os.environ.get("RADIXMEND_CMD_DIR"):
    try:
        from radixmend.sglang_hook import install

        install()
    except Exception as e:  # never block interpreter start-up
        print(f"[radixmend] hook not installed: {e!r}", file=sys.stderr)

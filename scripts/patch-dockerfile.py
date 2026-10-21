#!/usr/bin/env python3
"""Patch docker/Dockerfile on llmhost2: pre-stage the pinned CPython archive.

The host's network drops every connection at ~60 MiB, and the pinned CPython is 66 MiB,
so package_terminal.py died with ContentTooShortError. We hand it a pre-staged copy
through a build context directory instead. package_terminal.py still verifies the file
against its own sha256 pin, so nothing is trusted blindly.
"""
from pathlib import Path

P = Path("/opt/strata-780m/strata/docker/Dockerfile")
s = P.read_text()

ANCHOR = 'RUN python3 /src/packaging/package_terminal.py --out /out --cache /cache \\'

REPLACEMENT = (
    "# The pinned CPython archive is pre-staged in /pb-cache: this network drops every connection\n"
    "# at ~60 MiB and this file is 66 MiB, so package_terminal.py used to die with\n"
    "# ContentTooShortError. It still verifies the sha256 against its own pin below.\n"
    "COPY pb-cache/ /pb-cache/\n"
    "RUN python3 /src/packaging/package_terminal.py --out /out --cache /pb-cache \\"
)

if ANCHOR not in s:
    raise SystemExit("anchor not found in Dockerfile")

s = s.replace(ANCHOR, REPLACEMENT, 1)
P.write_text(s)
print("patched: terminal stage now uses /pb-cache")
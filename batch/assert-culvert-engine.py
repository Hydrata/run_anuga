#!/usr/bin/env python3
"""assert-culvert-engine.py — fail the BUILD if the culvert engine is missing or wrong.

Usage:  python assert-culvert-engine.py <anuga-drainage-sha>

Run in batch/Dockerfile right after the culvert-engine pip line of EACH target
(cpu and gpu), and after the substitution guard's rationale: a culvert package
on an image without the engine must never reach a customer (TASK-3643, epic 3585).

Proves, from installed metadata and a real import, that:
* pyswmm 2.2.0 and swmm-toolkit 0.17.0 are installed (exact pins);
* anuga_drainage was built from the pinned commit (PEP 610 direct_url.json
  vcs_info.commit_id for a VCS install; for the Dockerfile's local-dir install from the
  src stage, the HYDRATA_SOURCE_SHA stamp written at checkout);
* ``from anuga_drainage.culverts import HeadwallCulvert, attach_culverts`` works
  (this imports pyswmm and the bundled SWMM 5.2 library).
"""

from __future__ import annotations

import json
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

EXPECTED_VERSIONS = {"pyswmm": "2.2.0", "swmm-toolkit": "0.17.0"}


def main(argv) -> int:
    if len(argv) != 2 or len(argv[1]) < 7:
        print("usage: assert-culvert-engine.py <anuga-drainage-sha>", file=sys.stderr)
        return 2
    sha = argv[1].lower()
    problems = []
    for name, expected in EXPECTED_VERSIONS.items():
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            problems.append(f"{name} is not installed (need {expected})")
            continue
        if found != expected:
            problems.append(f"{name} is {found}, need {expected}")
    try:
        direct_url = json.loads(metadata.distribution("anuga_drainage").read_text("direct_url.json") or "{}")
    except metadata.PackageNotFoundError:
        problems.append("anuga_drainage is not installed")
        direct_url = None
    if direct_url is not None:
        commit = (direct_url.get("vcs_info") or {}).get("commit_id", "")
        if not commit and "dir_info" in direct_url:  # the Dockerfile's src-stage clone, stamped at checkout
            stamp = Path(unquote(urlparse(direct_url.get("url", "")).path)) / "HYDRATA_SOURCE_SHA"
            commit = stamp.read_text().strip() if stamp.is_file() else ""
        if not commit or not (commit.lower().startswith(sha) or sha.startswith(commit.lower())):
            problems.append(f"anuga_drainage was not built from commit {sha} (direct_url commit_id={commit!r})")
    try:
        from anuga_drainage.culverts import HeadwallCulvert, attach_culverts  # noqa: F401
        import pyswmm  # noqa: F401
        from swmm import toolkit  # noqa: F401
    except Exception as error:  # noqa: BLE001 - any import failure is the finding
        problems.append(f"the culvert engine does not import: {error!r}")
    if problems:
        print("CULVERT ENGINE GUARD FAILED:\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    print(f"culvert engine OK: pyswmm {EXPECTED_VERSIONS['pyswmm']}, swmm-toolkit "
          f"{EXPECTED_VERSIONS['swmm-toolkit']}, anuga_drainage @ {sha[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

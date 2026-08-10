#!/usr/bin/env python3
"""assert-anuga-provenance.py — fail the BUILD if the importable ``anuga`` is
not the engine this image just built from the pinned anuga_core checkout.

Usage:
    python assert-anuga-provenance.py <source-dir> <anuga-core-sha>
    python assert-anuga-provenance.py /app/anuga_core 7f1a4847df34...

Run it in batch/Dockerfile AFTER the last pip install of a stage — the whole
point is to catch a LATER pip line silently replacing the engine.

WHY THIS EXISTS (TASK-2700)
---------------------------
The ``src`` stage deletes ``/src/anuga_core/.git``, so anuga_core's
``git describe``-derived version used to resolve to the literal
``0.0.0+unknown``. Any ``anuga>=`` constraint reaching a pip line in this image
— the obvious one being ``run_anuga_src[full]``, which declares
``anuga>=3.3.0`` — therefore resolves the floor, downloads PyPI anuga and
installs it OVER the purpose-built engine. Nothing fails; the image just
quietly runs different code. In the ``gpu`` target that swaps an nvc +
OpenMP-target-offload build for a generic CPU wheel, and the existing baked
offload detector cannot see it: that detector inspects whichever engine it
finds, so a WHOLESALE package swap is exactly the shape it misses.

This is not hypothetical. A cp312 manylinux wheel for anuga 3.3.x is on PyPI
today, so the substitution is a fast, silent success. The web side paid ~19h of
hydrata.com ANUGA outage for this same bug before TASK-2695 removed ``[full]``
from install-anuga.yml; the container was protected only by the accident that
nobody had typed ``[full]`` into the Dockerfile.

WHAT PROVES IDENTITY — AND WHAT DELIBERATELY DOES NOT
-----------------------------------------------------
* THE PROOF is PEP 610 build provenance. pip writes ``direct_url.json`` into
  the ``.dist-info`` of any distribution installed from a local path or VCS,
  and writes NOTHING for one resolved from an index. So "was this dist built
  from /app/anuga_core?" is answerable from the installed metadata itself, with
  no reference to a version number at all. A PyPI substitute has no
  ``direct_url.json`` whatsoever — a categorical difference, not a comparison.

* NOT a version ORDERING compare. ``anuga.__version__ >= '3.3.0'``, in any
  spelling including the tuple-of-ints one, is exactly backwards here: the
  legitimate fork build reports a ``0.0.0`` base, so an ordering test REJECTS
  the correct engine and ACCEPTS the PyPI substitute. Epic 2702 forbids it
  outright, and the sibling deploy-side gate
  (deploy: roles/ansible-geonode/tasks/anuga-engine-verify.yml) makes the same
  argument at length.

* The version string is a CORROBORATOR only, never the identity proof — and it
  is checked as SHA CONTAINMENT, never ordering. It is deliberately weak
  evidence because this image now sets the version itself (ANUGA_VERSION ->
  _git_version.py's documented override), so all it can prove is that the pin
  reached meson. That is still worth asserting: it is what keeps the image's
  self-reported version honest, and it fails loudly if a future edit drops the
  ANUGA_VERSION plumbing.
"""

import json
import re
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

CHECKS = []


def _record(ok, headline, detail):
    CHECKS.append((ok, headline, detail))
    return ok


def _die():
    bad = [c for c in CHECKS if not c[0]]
    if not bad:
        return
    out = sys.stderr
    print("=" * 78, file=out)
    print("ANUGA PROVENANCE GATE — BUILD REFUSED", file=out)
    print("=" * 78, file=out)
    print(
        "The `anuga` this image would import is NOT provably the engine built\n"
        "from the pinned anuga_core checkout in this build. The usual cause is a\n"
        "pip line that introduced an `anuga>=` constraint (e.g. the `[full]`\n"
        "extra), letting pip install PyPI anuga OVER the purpose-built engine.\n"
        "In the gpu target that silently discards the nvc OpenMP-offload build.\n"
        "See batch/assert-anuga-provenance.py's module docstring (TASK-2700).\n",
        file=out,
    )
    # `detail` is written as the FAILURE explanation for its check, so print it
    # only where the check actually failed — attached to a PASS line it reads as
    # a flat contradiction ("[PASS] ... direct_url.json is ABSENT").
    for ok, headline, detail in CHECKS:
        print("  [%s] %s" % ("PASS" if ok else "FAIL", headline), file=out)
        if not ok and detail:
            for line in str(detail).splitlines():
                print("         %s" % line, file=out)
    print("=" * 78, file=out)
    sys.exit(1)


def _anuga_dists():
    return [
        d
        for d in importlib_metadata.distributions()
        if (d.metadata["Name"] or "").strip().lower() == "anuga"
    ]


def main(argv):
    if len(argv) != 3:
        sys.exit("usage: assert-anuga-provenance.py <source-dir> <anuga-core-sha>")
    source_dir, pinned_sha = argv[1], argv[2].strip().lower()

    if not re.fullmatch(r"[0-9a-f]{7,40}", pinned_sha):
        sys.exit(
            "assert-anuga-provenance.py: <anuga-core-sha> must be a hex git sha, "
            "got %r. An empty value usually means the ANUGA_CORE_SHA build-arg "
            "was not in ARG scope at this point in the Dockerfile stage." % pinned_sha
        )

    expected_url = Path(source_dir).resolve().as_uri()

    # ---- 1. exactly one installed anuga distribution ----------------------
    # Two would mean a shadowing install where import order, not the build,
    # decides which engine runs.
    dists = _anuga_dists()
    if not _record(
        len(dists) == 1,
        "exactly one installed `anuga` distribution",
        "found %d: %s" % (len(dists), [str(getattr(d, "_path", "?")) for d in dists]),
    ):
        _die()
    dist = dists[0]

    # ---- 2. THE identity proof: PEP 610 direct_url.json -------------------
    raw = dist.read_text("direct_url.json")
    if not _record(
        raw is not None,
        "dist-info carries PEP 610 direct_url.json (built from a local path)",
        "direct_url.json is ABSENT from %s.\n"
        "That is the signature of an INDEX install: pip records this file only\n"
        "for a local-path/VCS install. This anuga came from PyPI, not from %s."
        % (getattr(dist, "_path", "?"), source_dir),
    ):
        _die()

    try:
        info = json.loads(raw)
    except ValueError as exc:
        _record(False, "direct_url.json parses as JSON", "%s: %r" % (exc, raw))
        _die()

    recorded_url = str(info.get("url", "")).rstrip("/")
    _record(
        recorded_url == expected_url,
        "direct_url.json url == the build's anuga_core source tree",
        "recorded=%r expected=%r (full record: %s)" % (recorded_url, expected_url, info),
    )

    # ---- 3. the imported module is the one we just vetted -----------------
    # Guards against a stray source tree or .pth entry winning on sys.path
    # ahead of the distribution whose metadata we just checked.
    import anuga  # noqa: E402  (deliberately after the metadata checks)

    imported = Path(anuga.__file__).resolve().parent
    vetted = Path(str(dist.locate_file("anuga"))).resolve()
    _record(
        imported == vetted,
        "`import anuga` resolves to the vetted distribution",
        "imported from %s but the checked distribution installs to %s" % (imported, vetted),
    )

    # ---- 4. the C extensions were actually built --------------------------
    # A metadata-correct but hollow install (pure-python, extensions missing)
    # would pass every check above and die at simulation time instead.
    so_files = sorted(p.name for p in imported.rglob("*.so"))
    _record(
        len(so_files) > 0,
        "compiled extension modules present in the installed package",
        "no .so files under %s — the engine was not built" % imported,
    )

    # ---- 5. CORROBORATOR: the version carries the pinned SHA --------------
    # SHA CONTAINMENT, never an ordering compare (see the module docstring).
    # Prefix-tolerant in both directions because git chooses the abbreviation
    # width; same matching rule as the deploy-side anuga-engine-verify gate, so
    # the web and the container speak one provenance dialect.
    version = str(getattr(anuga, "__version__", ""))
    tokens = re.findall(r"g([0-9a-f]{7,40})", version)
    _record(
        any(t.startswith(pinned_sha) or pinned_sha.startswith(t) for t in tokens),
        "anuga.__version__ carries a build id for the pinned anuga_core commit",
        "version=%r carries no g-token matching %s (found: %s).\n"
        "If the identity checks above PASSED, this means the ANUGA_VERSION\n"
        "build-arg stopped reaching meson (_git_version.py's override) — the\n"
        "engine is right but the image can no longer say so."
        % (version, pinned_sha, tokens or "none"),
    )

    _die()

    print("=" * 78)
    print("ANUGA PROVENANCE GATE — PASS")
    for _, headline, _detail in CHECKS:
        print("  [PASS] %s" % headline)
    print("  engine source : %s" % recorded_url)
    print("  anuga_core pin: %s" % pinned_sha)
    print("  __version__   : %s" % version)
    print("  extensions    : %d compiled .so modules" % len(so_files))
    print("=" * 78)


if __name__ == "__main__":
    main(sys.argv)

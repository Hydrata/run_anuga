#!/usr/bin/env python3
# VENDORED from Hydrata/deploy scripts/marker_lint.py (TASK-2329). Canonical source lives there; keep in sync. Sync-seam tracked as an IMPROVEMENT.
"""Fleet marker & quarantine taxonomy linter — TASK-2293 (epic 2290, W0).

Classifies EVERY test marker in a repo checkout into the fleet taxonomy and, in
``--check`` mode, enforces each class's governance rule. It is a STATIC scanner
(ast for Python, regex for JS) so it needs no pytest/karma, no DB and no repo
install — that is what lets W3 wire the SAME script into every fleet repo's CI:

    python scripts/marker_lint.py --repo /opt/hydrata            # report
    python scripts/marker_lint.py --repo /opt/geonode --check    # gate

Taxonomy (full spec: docs/testing/marker-quarantine-taxonomy.md):

  xfail            @pytest.mark.xfail / xfail_bug / xfail_env / xfail_mock and
                   pytest.xfail(...).  RATCHET TO ZERO (hard). Governance:
                   reason + sunset (+ ticket) — a still-failing test hides a bug.
  conditional-skip @pytest.mark.skipif(...) and pytest.importorskip(...).
                   LEGITIMATE environment carve-out (Linux-only, optional dep,
                   GeoServer-up, ERA5 opt-in). Auditable + COUNT-CAPPED allowlist.
                   Governance: reason (the condition itself documents *when*).
  bare-skip        @pytest.mark.skip(...) / pytest.skip(...) with NO condition.
                   RATCHET TO ZERO. Governance: reason + sunset + ticket — an
                   unconditional skip is a deleted test in disguise; migrate it to
                   xfail (known bug), skipif (real env gate), quarantine (flake),
                   or fix it.
  quarantine       @pytest.mark.quarantine(...) — a GENUINE flake with a sunset.
                   The one legal home for "passes-usually" tests. COUNT-CAPPED.
                   Governance: reason + sunset + ticket.
  js-focus         it.only / describe.only / fit / fdescribe (JS). RATCHET TO
                   ZERO, hard — a focus silently drops the rest of the suite.
  js-skip          xit / xdescribe / it.skip / describe.skip (JS). RATCHET TO
                   ZERO. Governance: a trailing // TICKET+sunset comment.

Exit codes (``--check``): 0 = clean, 1 = governance violation or over cap, 2 = a
file failed to parse. Without ``--check`` it always exits 0 (report only).
"""
from __future__ import annotations

import argparse
import ast
import datetime
import os
import re
import sys

# ---- class constants --------------------------------------------------------
XFAIL_MARKS = {"xfail", "xfail_bug", "xfail_env", "xfail_mock"}
COND_SKIP_MARKS = {"skipif", "importorskip"}
BARE_SKIP_MARKS = {"skip"}
QUARANTINE_MARKS = {"quarantine", "flaky"}
ALL_PY_MARKS = XFAIL_MARKS | COND_SKIP_MARKS | BARE_SKIP_MARKS | QUARANTINE_MARKS

CLASS_XFAIL = "xfail"
CLASS_COND = "conditional-skip"
CLASS_BARE = "bare-skip"
CLASS_QUAR = "quarantine"
CLASS_JS_FOCUS = "js-focus"
CLASS_JS_SKIP = "js-skip"

# Ratchet target per class: 0 => must burn to zero; None => count-capped allowlist.
RATCHET = {
    CLASS_XFAIL: 0,
    CLASS_COND: None,
    CLASS_BARE: 0,
    CLASS_QUAR: None,
    CLASS_JS_FOCUS: 0,
    CLASS_JS_SKIP: 0,
}
# Required governance kwargs/annotations per class. conditional-skip carries NO
# per-marker requirement — the skipif condition / importorskip module name / the
# guarding `if` around an imperative pytest.skip() IS the audit; that class is
# governed by the count-cap + human allowlist review instead.
REQUIRED = {
    CLASS_XFAIL: ("reason", "sunset"),
    CLASS_COND: (),
    CLASS_BARE: ("reason", "sunset", "ticket"),
    CLASS_QUAR: ("reason", "sunset", "ticket"),
    CLASS_JS_FOCUS: (),
    CLASS_JS_SKIP: ("ticket",),
}

TEST_PY_RE = re.compile(r"(^|/)(test_[^/]*\.py|[^/]*_test\.py|conftest\.py)$")
TEST_JS_RE = re.compile(r"(-test\.js|\.test\.jsx?)$|/__tests__/")
SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "_archived",
             "dist", "build", ".pytest_cache", "site-packages"}
TICKET_RE = re.compile(r"TASK-\d+|#\d+")


class Finding:
    __slots__ = ("cls", "name", "file", "line", "reason", "sunset", "ticket")

    def __init__(self, cls, name, file, line, reason=None, sunset=None, ticket=None):
        self.cls, self.name, self.file, self.line = cls, name, file, line
        self.reason, self.sunset, self.ticket = reason, sunset, ticket

    def missing(self):
        need = REQUIRED[self.cls]
        miss = []
        for k in need:
            v = getattr(self, k)
            if not v:
                miss.append(k)
        # a valid sunset must parse as a date
        if "sunset" in need and self.sunset:
            try:
                datetime.date.fromisoformat(self.sunset)
            except ValueError:
                miss.append("sunset(unparseable)")
        return miss


# ---- python scanning --------------------------------------------------------
def _attr_tail(node):
    """Return the trailing attribute name of a decorator/attribute expr, and
    whether the chain contains a `.mark.` segment (a pytest marker)."""
    # node may be Call(func=Attribute...) or Attribute directly
    if isinstance(node, ast.Call):
        node = node.func
    names = []
    cur = node
    while isinstance(cur, ast.Attribute):
        names.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        names.append(cur.id)
    names.reverse()  # e.g. ['pytest','mark','xfail']
    if not names:
        return None, False
    is_mark = "mark" in names
    return names[-1], is_mark


def _kwargs_of(call):
    out = {}
    if isinstance(call, ast.Call):
        for kw in call.keywords:
            if kw.arg and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                out[kw.arg] = kw.value.value
    return out


def _mk_finding(cls, name, path, lineno, call):
    kw = _kwargs_of(call)
    reason = kw.get("reason")
    sunset = kw.get("sunset")
    ticket = kw.get("ticket")
    # ticket may be embedded in the reason text (TASK-N / #N)
    if not ticket and reason and TICKET_RE.search(reason):
        ticket = TICKET_RE.search(reason).group(0)
    return Finding(cls, name, path, lineno, reason, sunset, ticket)


def _classify_py_mark(name):
    if name in XFAIL_MARKS:
        return CLASS_XFAIL
    if name in COND_SKIP_MARKS:
        return CLASS_COND
    if name in BARE_SKIP_MARKS:
        return CLASS_BARE
    if name in QUARANTINE_MARKS:
        return CLASS_QUAR
    return None


def scan_python(path, findings, parse_errors):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except SyntaxError as exc:
        parse_errors.append(f"{path}:{exc.lineno}: {exc.msg}")
        return

    def handle_marker_expr(expr, lineno):
        name, is_mark = _attr_tail(expr)
        if not name:
            return
        # decorator/pytestmark form: pytest.mark.<name>
        if is_mark and name in ALL_PY_MARKS:
            cls = _classify_py_mark(name)
            if cls:
                findings.append(_mk_finding(cls, name, path, lineno, expr))
            return
        # imperative form: pytest.importorskip(...) / pytest.skip(...) / pytest.xfail(...).
        # Imperative pytest.skip() runs the test up to a runtime guard and skips on
        # a condition -> conditional-skip (auditable), NOT a collection-time bare skip
        # (that dangerous form is the @pytest.mark.skip DECORATOR, handled above).
        if isinstance(expr, ast.Call) and not is_mark and name in {"importorskip", "skip", "xfail"}:
            cls = {"importorskip": CLASS_COND, "skip": CLASS_COND, "xfail": CLASS_XFAIL}[name]
            findings.append(_mk_finding(cls, name, path, lineno, expr))

    for node in ast.walk(tree):
        # decorators on defs/classes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                handle_marker_expr(dec, getattr(dec, "lineno", node.lineno))
        # module/class-level `pytestmark = pytest.mark.skip(...)` (or a list of them)
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "pytestmark":
                    vals = node.value.elts if isinstance(node.value, (ast.List, ast.Tuple)) else [node.value]
                    for v in vals:
                        handle_marker_expr(v, getattr(v, "lineno", node.lineno))
        # imperative calls anywhere
        if isinstance(node, ast.Call):
            name, is_mark = _attr_tail(node)
            if not is_mark and name in {"importorskip", "skip", "xfail"}:
                # only pytest.<name>(...), not unrelated .skip()
                base = node.func.value if isinstance(node.func, ast.Attribute) else None
                if isinstance(base, ast.Name) and base.id == "pytest":
                    handle_marker_expr(node, node.lineno)


# ---- js scanning ------------------------------------------------------------
JS_FOCUS_RE = re.compile(r"\b(?:fit|fdescribe)\s*\(|\b(?:it|describe)\.only\s*\(")
JS_SKIP_RE = re.compile(r"\b(?:xit|xdescribe)\s*\(|\b(?:it|describe)\.skip\s*\(")


def scan_js(path, findings, parse_errors):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        parse_errors.append(f"{path}: {exc}")
        return
    for i, line in enumerate(lines, 1):
        s = line.split("//", 1)[0]
        if JS_FOCUS_RE.search(s):
            findings.append(Finding(CLASS_JS_FOCUS, "focus", path, i))
        if JS_SKIP_RE.search(s):
            ticket = None
            m = TICKET_RE.search(line)
            if m:
                ticket = m.group(0)
            findings.append(Finding(CLASS_JS_SKIP, "skip", path, i, ticket=ticket))


# ---- driver -----------------------------------------------------------------
def walk_repo(repo):
    for root, dirs, files in os.walk(repo):
        # Prune SKIP_DIRS, _archived*, and NESTED git repos (a subdir with its own
        # .git is a separate fleet clone — e.g. /opt/hydrata/run_anuga — whose
        # markers belong to that repo's own lint run, not this one).
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRS and not d.startswith("_archived")
            and not os.path.exists(os.path.join(root, d, ".git"))
        ]
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, repo)
            if f.endswith(".py") and TEST_PY_RE.search("/" + rel):
                yield full, "py"
            elif f.endswith((".js", ".jsx")) and TEST_JS_RE.search("/" + rel):
                yield full, "js"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="repo checkout root to scan")
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero on any governance violation or over-cap class")
    ap.add_argument("--cap", action="append", default=[], metavar="CLASS=N",
                    help="cap for a count-capped class, e.g. --cap conditional-skip=40 "
                         "(repeatable). RATCHET-TO-ZERO classes are always capped at 0.")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(repo):
        print(f"ERROR: --repo {repo} is not a directory", file=sys.stderr)
        return 2

    caps = {}
    for spec in args.cap:
        k, _, v = spec.partition("=")
        caps[k.strip()] = int(v)

    findings, parse_errors = [], []
    for full, kind in walk_repo(repo):
        (scan_python if kind == "py" else scan_js)(full, findings, parse_errors)

    by_class = {}
    for f in findings:
        by_class.setdefault(f.cls, []).append(f)

    # governance: missing kwargs
    violations = []
    for f in findings:
        miss = f.missing()
        if miss:
            violations.append((f, miss))
    # caps: an explicit --cap always wins (W3 pins the CURRENT baseline as the
    # ceiling and ratchets it DOWN over time); otherwise a ratchet-to-zero class
    # defaults to 0 and a count-capped class is unbounded. The taxonomy TARGET for
    # ratchet classes stays 0 regardless of the interim ceiling.
    over_cap = []
    for cls, items in by_class.items():
        target = RATCHET.get(cls)
        cap = caps.get(cls, 0 if target == 0 else None)
        if cap is not None and len(items) > cap:
            over_cap.append((cls, len(items), cap))

    if args.json:
        import json as _json
        print(_json.dumps({
            "repo": repo,
            "counts": {c: len(v) for c, v in sorted(by_class.items())},
            "findings": [
                {"class": f.cls, "marker": f.name, "file": os.path.relpath(f.file, repo),
                 "line": f.line, "reason": f.reason, "sunset": f.sunset, "ticket": f.ticket}
                for f in findings
            ],
            "violations": [
                {"file": os.path.relpath(f.file, repo), "line": f.line,
                 "class": f.cls, "missing": miss} for f, miss in violations
            ],
            "over_cap": [{"class": c, "count": n, "cap": cap} for c, n, cap in over_cap],
            "parse_errors": parse_errors,
        }, indent=2))
    else:
        print(f"\nMARKER TAXONOMY LINT — {repo}")
        print("=" * 68)
        if not by_class:
            print("  (no markers found)")
        for cls in (CLASS_XFAIL, CLASS_BARE, CLASS_JS_FOCUS, CLASS_JS_SKIP, CLASS_COND, CLASS_QUAR):
            items = by_class.get(cls)
            if not items:
                continue
            target = RATCHET[cls]
            tgt_str = "ratchet->0" if target == 0 else f"cap={caps.get(cls, 'uncapped')}"
            print(f"\n  {cls:<16} {len(items):>3}   [{tgt_str}]")
            for f in sorted(items, key=lambda x: (x.file, x.line)):
                rel = os.path.relpath(f.file, repo)
                meta = []
                if f.sunset:
                    meta.append(f"sunset={f.sunset}")
                if f.ticket:
                    meta.append(f"ticket={f.ticket}")
                flag = ""
                miss = f.missing()
                if miss:
                    flag = f"  <-- MISSING {','.join(miss)}"
                print(f"      {rel}:{f.line}  {f.name}  {' '.join(meta)}{flag}")
        if parse_errors:
            print(f"\n  PARSE ERRORS ({len(parse_errors)}):")
            for e in parse_errors:
                print(f"      {e}")
        print("\n  " + "-" * 64)
        print(f"  totals: " + ", ".join(f"{c}={len(v)}" for c, v in sorted(by_class.items())) or "  totals: 0")
        if violations:
            print(f"  GOVERNANCE VIOLATIONS: {len(violations)} marker(s) missing required metadata")
        for cls, n, cap in over_cap:
            print(f"  OVER CAP: {cls} has {n} (cap {cap})")

    if args.check:
        if parse_errors:
            return 2
        if violations or over_cap:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

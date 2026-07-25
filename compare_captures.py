#!/usr/bin/env python3
"""
compare_captures.py — n-way comparison of system prompt captures.

Usage:
    python3 compare_captures.py capture1.md capture2.md capture3.md ...
    python3 compare_captures.py --outdir results/ *.md

Outputs to stdout (or --outdir):
    presence_matrix.md   which blocks appear in which captures
    agreement.md         per-block similarity + agreement counts
    divergences.md       word-level diffs for blocks that differ

No dependencies beyond the Python standard library. Tested on 3.10+.

Method notes live in COMPARISON_METHODOLOGY.md. This script covers
Phases 2-4 (normalize, segment, diff). Phases 1, 5 and 6 -- metadata
capture, divergence classification, and publication -- are human work.
"""

import argparse
import difflib
import hashlib
import os
import re
import sys
import unicodedata
from collections import OrderedDict

# ---------------------------------------------------------------- normalize

# Typographic substitutions. Copy-paste through a browser mangles these
# inconsistently between captures and produces enormous false diff signal.
TYPO_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2013": "-", "\u2014": "--", "\u2015": "--", "\u2212": "-",
    "\u2026": "...",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u200a": " ",
    "\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",
    "\u00ad": "",
}


def normalize(text):
    """Normalize a capture for comparison.

    Deliberately does NOT lowercase: case is meaningful in these documents
    (NEVER, CRITICAL, ALWAYS are load-bearing formatting).
    """
    text = unicodedata.normalize("NFKC", text)
    for src, dst in TYPO_MAP.items():
        text = text.replace(src, dst)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Unwrap hard line breaks inside paragraphs: a single newline between two
    # non-empty, non-tag lines is treated as soft wrapping. Blank lines and
    # tag boundaries are preserved as real structure.
    text = re.sub(r"(?<=[^\n>])\n(?=[^\n<\-*#|\d])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ------------------------------------------------------------------ segment

# Matches <block_name> ... </block_name>, non-greedy, dotall.
BLOCK_RE = re.compile(r"<([a-zA-Z_][\w:-]*)\s*>(.*?)</\1\s*>", re.DOTALL)


def segment(text):
    """Split a normalized capture into named blocks.

    Returns an OrderedDict of block_name -> dict(content, ordinal, chars, hash).
    Repeated block names are suffixed #2, #3, ... to keep keys unique.
    Text outside any block is collected under the key '(unwrapped)'.
    """
    blocks = OrderedDict()
    seen = {}
    ordinal = 0
    covered = []

    for m in BLOCK_RE.finditer(text):
        name, body = m.group(1), m.group(2).strip()
        seen[name] = seen.get(name, 0) + 1
        key = name if seen[name] == 1 else f"{name}#{seen[name]}"
        ordinal += 1
        blocks[key] = _entry(body, ordinal)
        covered.append((m.start(), m.end()))

    # Anything not inside a matched block, concatenated.
    leftover, cursor = [], 0
    for start, end in covered:
        if start > cursor:
            leftover.append(text[cursor:start])
        cursor = max(cursor, end)
    if cursor < len(text):
        leftover.append(text[cursor:])
    stray = "\n".join(s.strip() for s in leftover if s.strip())
    if stray:
        blocks["(unwrapped)"] = _entry(stray, 0)

    return blocks


def _entry(body, ordinal):
    return {
        "content": body,
        "ordinal": ordinal,
        "chars": len(body),
        "hash": hashlib.sha256(body.encode("utf-8")).hexdigest()[:12],
    }


# --------------------------------------------------------------------- diff

def similarity(a, b):
    """Token-level similarity ratio, 0.0-1.0."""
    return difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def word_diff(a, b, label_a, label_b, context=6):
    """Readable word-level diff between two block bodies."""
    sm = difflib.SequenceMatcher(None, a.split(), b.split())
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            chunk = a.split()[i1:i2]
            if len(chunk) > context * 2:
                out.append(" ".join(chunk[:context]) + " [...] " +
                           " ".join(chunk[-context:]))
            else:
                out.append(" ".join(chunk))
        elif tag == "delete":
            out.append(f"\n\n  [-{label_a} ONLY-] {' '.join(a.split()[i1:i2])}\n\n")
        elif tag == "insert":
            out.append(f"\n\n  [+{label_b} ONLY+] {' '.join(b.split()[j1:j2])}\n\n")
        elif tag == "replace":
            out.append(f"\n\n  [-{label_a}-] {' '.join(a.split()[i1:i2])}"
                       f"\n  [+{label_b}+] {' '.join(b.split()[j1:j2])}\n\n")
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()


# ------------------------------------------------------------------ reports

def presence_matrix(captures):
    """captures: OrderedDict of label -> blocks dict."""
    labels = list(captures)
    all_blocks = OrderedDict()
    for blocks in captures.values():
        for key, entry in blocks.items():
            all_blocks.setdefault(key, []).append(entry["ordinal"])

    lines = ["# Presence matrix", "",
             "`+` present, `.` absent. Number in parentheses is ordinal "
             "position within that capture.", "",
             "| block | " + " | ".join(labels) + " | n |",
             "|---|" + "---|" * (len(labels) + 1)]

    ragged = []
    for key in all_blocks:
        cells, count = [], 0
        for label in labels:
            entry = captures[label].get(key)
            if entry:
                count += 1
                cells.append(f"+ ({entry['ordinal']})")
            else:
                cells.append(".")
        if count < len(labels):
            ragged.append((key, count))
        lines.append(f"| `{key}` | " + " | ".join(cells) + f" | {count} |")

    lines += ["", f"**{len(all_blocks)} distinct blocks across "
                  f"{len(labels)} captures.**", ""]
    if ragged:
        lines += ["## Ragged rows -- candidates for account- or "
                  "surface-conditional content", "",
                  "Present in some captures but not all. Check these against "
                  "your metadata table (Phase 1) before concluding the prompt "
                  "changed.", ""]
        for key, count in sorted(ragged, key=lambda x: x[1]):
            lines.append(f"- `{key}` -- {count}/{len(labels)} captures")
    else:
        lines.append("No ragged rows: all captures contain the same block set.")
    return "\n".join(lines) + "\n"


def agreement_report(captures, threshold=0.98):
    labels = list(captures)
    all_keys = OrderedDict()
    for blocks in captures.values():
        for key in blocks:
            all_keys.setdefault(key, None)

    lines = ["# Per-block agreement", "",
             f"Similarity is mean pairwise token-level ratio. Blocks at "
             f">= {threshold:g} are treated as identical.", "",
             "| block | captures | mean similarity | verdict |",
             "|---|---|---|---|"]

    stable, volatile, single = [], [], []
    for key in all_keys:
        present = [(lab, captures[lab][key]["content"])
                   for lab in labels if key in captures[lab]]
        if len(present) == 1:
            single.append(key)
            lines.append(f"| `{key}` | 1/{len(labels)} | -- | "
                         f"SINGLE SOURCE -- unconfirmed |")
            continue
        ratios = [similarity(a[1], b[1])
                  for i, a in enumerate(present)
                  for b in present[i + 1:]]
        mean = sum(ratios) / len(ratios)
        if mean >= threshold:
            verdict = "identical"
            stable.append(key)
        elif mean >= 0.85:
            verdict = "minor divergence"
            volatile.append(key)
        else:
            verdict = "**MAJOR divergence**"
            volatile.append(key)
        lines.append(f"| `{key}` | {len(present)}/{len(labels)} | "
                     f"{mean:.3f} | {verdict} |")

    lines += ["", "## Stable-vs-volatile map", "",
              f"**Stable ({len(stable)})** -- identical wherever present. "
              f"Likely core deployment config.", ""]
    lines += [f"- `{k}`" for k in stable] or ["- (none)"]
    lines += ["", f"**Volatile ({len(volatile)})** -- content differs. "
                  f"Classify each in the divergence register.", ""]
    lines += [f"- `{k}`" for k in volatile] or ["- (none)"]
    if single:
        lines += ["", f"**Single-source ({len(single)})** -- appears in only "
                      f"one capture. Treat as hypothesis, not finding.", ""]
        lines += [f"- `{k}`" for k in single]
    return "\n".join(lines) + "\n"


def divergence_report(captures, threshold=0.98):
    labels = list(captures)
    lines = ["# Divergences (word-level)", "",
             "Only blocks present in 2+ captures and below the identity "
             "threshold are shown.", ""]
    any_found = False
    all_keys = OrderedDict()
    for blocks in captures.values():
        for key in blocks:
            all_keys.setdefault(key, None)

    for key in all_keys:
        present = [(lab, captures[lab][key]["content"])
                   for lab in labels if key in captures[lab]]
        if len(present) < 2:
            continue
        pairs = [(a, b) for i, a in enumerate(present) for b in present[i + 1:]]
        shown = [(a, b) for a, b in pairs
                 if similarity(a[1], b[1]) < threshold]
        if not shown:
            continue
        any_found = True
        lines += [f"## `{key}`", ""]
        for (la, ta), (lb, tb) in shown:
            ratio = similarity(ta, tb)
            lines += [f"### {la} vs {lb} (similarity {ratio:.3f})", "",
                      "```diff", word_diff(ta, tb, la, lb), "```", "",
                      "**Classification:** _A account / B surface / "
                      "C temporal / D capture artifact / E rollout — "
                      "fill in_", ""]
    if not any_found:
        lines.append("No divergences above threshold.")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="capture files (2 or more)")
    ap.add_argument("--outdir", help="write reports here instead of stdout")
    ap.add_argument("--threshold", type=float, default=0.98,
                    help="similarity at or above which blocks count as "
                         "identical (default 0.98)")
    args = ap.parse_args()

    if len(args.files) < 2:
        ap.error("need at least two capture files to compare")

    captures = OrderedDict()
    for path in args.files:
        label = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
        blocks = segment(normalize(raw))
        captures[label] = blocks
        print(f"[{label}] {len(blocks)} blocks, {len(raw):,} chars",
              file=sys.stderr)

    reports = {
        "presence_matrix.md": presence_matrix(captures),
        "agreement.md": agreement_report(captures, args.threshold),
        "divergences.md": divergence_report(captures, args.threshold),
    }

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        for name, body in reports.items():
            with open(os.path.join(args.outdir, name), "w",
                      encoding="utf-8") as fh:
                fh.write(body)
            print(f"wrote {os.path.join(args.outdir, name)}", file=sys.stderr)
    else:
        for name, body in reports.items():
            print(f"\n{'=' * 70}\n{name}\n{'=' * 70}\n")
            print(body)


if __name__ == "__main__":
    main()

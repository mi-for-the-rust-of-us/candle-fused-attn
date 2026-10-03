# SPDX-License-Identifier: MIT OR Apache-2.0
"""Fail unless every Rust code block of `docs/tutorial.md` appears verbatim in the tutorial example.

The tutorial is only worth reading if its code runs. `examples/tutorial.rs` is run by CI, so this
check ties the prose to it: each ```rust block of the tutorial must appear in the example as a run
of consecutive lines (compared line by line with surrounding whitespace stripped, so a block may be
quoted at a different indentation). A block that drifts fails CI.

Run (CPU, instant), from the crate root:
  python3 scripts/check-tutorial.py
Doctests:
  python3 -m doctest scripts/check-tutorial.py
"""

from __future__ import annotations

import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]


def rust_blocks(markdown: str) -> list[list[str]]:
  """The ```rust code blocks of a Markdown text, each as its list of lines.

  >>> rust_blocks("text\\n```rust\\nlet a = 1;\\nlet b = 2;\\n```\\nmore\\n```bash\\nls\\n```\\n")
  [['let a = 1;', 'let b = 2;']]
  """
  blocks, current = [], None
  for line in markdown.splitlines():
    if current is None and line.strip() == "```rust":
      current = []
    elif current is not None and line.strip() == "```":
      blocks.append(current)
      current = None
    elif current is not None:
      current.append(line)
  return blocks


def appears_in(block: list[str], source: str) -> bool:
  """Whether `block` occurs in `source` as consecutive lines, whitespace-stripped, blank lines
  ignored.

  >>> appears_in(["  let a = 1;", "let b = 2;"], "fn f() {\\n    let a = 1;\\n    let b = 2;\\n}")
  True
  >>> appears_in(["let a = 1;", "let c = 3;"], "let a = 1;\\nlet b = 2;\\nlet c = 3;")
  False
  """
  want = [line.strip() for line in block if line.strip()]
  have = [line.strip() for line in source.splitlines() if line.strip()]
  return any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1))


def main() -> None:
  """Check every tutorial block against the example; exit 1 on the first that is missing."""
  tutorial = (CRATE / "docs" / "tutorial.md").read_text(encoding="utf-8")
  example = (CRATE / "examples" / "tutorial.rs").read_text(encoding="utf-8")
  blocks = rust_blocks(tutorial)
  missing = [block for block in blocks if not appears_in(block, example)]
  for block in missing:
    print("docs/tutorial.md block not found in examples/tutorial.rs:\n  " + "\n  ".join(block))
  print(f"{len(blocks) - len(missing)} of {len(blocks)} tutorial blocks found in the example")
  sys.exit(1 if missing or not blocks else 0)


if __name__ == "__main__":
  main()

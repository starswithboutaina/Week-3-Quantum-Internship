"""README lint + image link checker for Week 3.

Checks:
  (a) every local ](...) path in README.md exists on disk,
  (b) even parity of $$ delimiters and of remaining $ delimiters,
  (c) no raw | inside math spans that sit in GitHub table rows,
  (d) no forbidden \\ket / \\bra macros,
  (e) no $$ blocks inside tables, only-PNG assets, no http image links.

Exit 0 + PASS line on success, exit 1 with FAIL lines otherwise.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def read_readme() -> list[str]:
    """Read README lines or exit with a FAIL message."""
    if not README.exists():
        print("FAIL: README.md not found")
        sys.exit(1)
    return README.read_text(encoding="utf-8").splitlines()


def check_images(lines: list[str]) -> list[str]:
    """Assert every local ](...) reference exists and is a PNG in assets/."""
    errors: list[str] = []
    for i, line in enumerate(lines, start=1):
        for m in re.finditer(r"\]\(([^)]+)\)", line):
            path = m.group(1).strip()
            if path.startswith(("http://", "https://", "#", "mailto:")):
                if path.startswith(("http://", "https://")) and "assets" in path:
                    errors.append(f"line {i}: absolute image link {path}")
                continue
            full = ROOT / path
            if not full.exists():
                errors.append(f"line {i}: missing file {path}")
            elif full.suffix.lower() != ".png":
                errors.append(f"line {i}: non-PNG asset {path}")
            elif full.parent.resolve() != (ROOT / "assets").resolve():
                errors.append(f"line {i}: asset outside assets/ {path}")
    return errors


def check_math_parity(text: str) -> list[str]:
    """Assert even parity of $$ blocks and of leftover inline $ spans."""
    errors: list[str] = []
    dbl = text.count("$$")
    if dbl % 2 != 0:
        errors.append(f"odd $$ count: {dbl}")
    stripped = text.replace("$$", "")
    single = stripped.count("$")
    if single % 2 != 0:
        errors.append(f"odd inline $ count: {single}")
    return errors


def check_display_placement(lines: list[str]) -> list[str]:
    """Display $$ must sit on their own lines; never inside tables."""
    errors: list[str] = []
    for i, line in enumerate(lines, start=1):
        if "$$" not in line:
            continue
        s = line.strip()
        if "|" in line and s.startswith("|"):
            errors.append(f"line {i}: $$ block inside table")
        if not (s.startswith("$$") and s.endswith("$$")) and s != "$$":
            # Allow a $$ line only if the whole stripped line is wrapped in $$.
            if s.count("$$") not in (1, 2):
                errors.append(f"line {i}: $$ not on its own line")
            elif not (s.startswith("$$") and s.endswith("$$")):
                errors.append(f"line {i}: $$ not on its own line")
    return errors


def check_table_math_pipes(lines: list[str]) -> list[str]:
    """No raw | inside $...$ spans within table rows."""
    errors: list[str] = []
    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        spans = re.findall(r"\$[^$]*?\$", line)
        for span in spans:
            if "|" in span:
                errors.append(f"line {i}: raw | inside math in table")
    return errors


def check_macros(text: str) -> list[str]:
    """Forbid \\ket and \\bra macros."""
    errors: list[str] = []
    if r"\ket" in text or r"\bra" in text:
        errors.append("forbidden \\ket or \\bra macro found")
    return errors


def check_underscores(lines: list[str]) -> list[str]:
    """Flag unescaped underscores in prose outside math and code spans."""
    errors: list[str] = []
    in_fence = False
    for i, line in enumerate(lines, start=1):
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if "$$" in line:
            continue
        if not s or s.startswith(("|", "#", "-", "*", ">", "!")):
            continue
        if line.startswith("    "):
            continue
        no_code = re.sub(r"`[^`]*`", "", line)
        no_math = re.sub(r"\$[^$]*?\$", "", no_code.replace("$$", ""))
        for m in re.finditer(r"[A-Za-z0-9]_", no_math):
            errors.append(f"line {i}: unescaped underscore near '{m.group(0)}'")
            break
    return errors


def main() -> int:
    """Run all checks and report PASS or FAIL lines."""
    lines = read_readme()
    text = "\n".join(lines)
    errors: list[str] = []
    errors += check_images(lines)
    errors += check_math_parity(text)
    errors += check_display_placement(lines)
    errors += check_table_math_pipes(lines)
    errors += check_macros(text)
    errors += check_underscores(lines)
    n_img = len(re.findall(r"!\[[^\]]*\]\(assets/[^)]+\)", text))
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print(f"PASS: README lint ok ({n_img} local images, math parity even, "
          f"no table-pipe math, assets exist)")
    print("PASS: os.path.exists() asserted for every local ](...) path")
    return 0


if __name__ == "__main__":
    sys.exit(main())

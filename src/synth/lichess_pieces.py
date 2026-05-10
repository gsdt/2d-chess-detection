"""Download and load Lichess open-source piece SVG sets.

Lichess publishes ~30 piece styles under github.com/lichess-org/lila/tree/master/public/piece.
Each style has 12 SVG files (wK..bP), each with a 45x45 viewBox.

Usage:
    python -m src.synth.lichess_pieces --out data/piece_sets

After download, `available_styles(root)` returns the set of styles that have a
complete 12-piece set; `load_piece_set(root, style)` returns dict[symbol -> inner SVG].
"""
from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.request
from pathlib import Path

# Curated list — every style on Lichess that ships a full 12-piece SVG set.
# Some entries (letter, mono, pixel, shapes, xkcd) are deliberately stylised
# and good for augmentation diversity; the network learns invariance.
LICHESS_STYLES: list[str] = [
    "alpha",
    "anarcandy",
    "california",
    "caliente",
    "cardinal",
    "cburnett",
    "chess7",
    "chessnut",
    "companion",
    "cooke",
    "disguised",
    "dubrovny",
    "fantasy",
    "fresca",
    "gioco",
    "governor",
    "horsey",
    "icpieces",
    "kiwen-suwi",
    "kosal",
    "leipzig",
    "letter",
    "libra",
    "maestro",
    "merida",
    "monarchy",
    "mono",
    "mpchess",
    "pirouetti",
    "pixel",
    "reillycraig",
    "rhosgfx",
    "riohacha",
    "shapes",
    "spatial",
    "staunty",
    "tatiana",
]

PIECE_FILES = [f"{c}{p}" for c in ("w", "b") for p in ("K", "Q", "R", "B", "N", "P")]
SYMBOLS = "KQRBNPkqrbnp"

BASE_URL = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece"


def _download(url: str, dest: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            data = r.read()
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  ! {url}: {e}")
        return False


def download_styles(out_dir: Path, styles: list[str] | None = None) -> dict[str, int]:
    """Download all SVGs for each style. Returns {style: num_pieces_downloaded}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    styles = styles or LICHESS_STYLES
    counts: dict[str, int] = {}
    for style in styles:
        sdir = out_dir / style
        sdir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for piece in PIECE_FILES:
            dest = sdir / f"{piece}.svg"
            if dest.exists() and dest.stat().st_size > 0:
                ok += 1
                continue
            url = f"{BASE_URL}/{style}/{piece}.svg"
            if _download(url, dest):
                ok += 1
        counts[style] = ok
        print(f"  {style}: {ok}/12")
    return counts


_INNER_RE = re.compile(r"<svg[^>]*>(.*)</svg>", re.DOTALL | re.IGNORECASE)


def _extract_inner(svg_text: str) -> str:
    """Strip the root <svg> wrapper, returning everything inside."""
    m = _INNER_RE.search(svg_text)
    if not m:
        # Defensive: some SVGs are just a <g> already.
        return svg_text
    return m.group(1)


def available_styles(root: Path) -> list[str]:
    """Return styles under `root` that have all 12 pieces."""
    if not root.exists():
        return []
    out = []
    for style_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if all((style_dir / f"{p}.svg").exists() for p in PIECE_FILES):
            out.append(style_dir.name)
    return out


def load_piece_set(root: Path, style: str) -> dict[str, str]:
    """Return {symbol -> inner SVG} for a style.

    Symbol uses python-chess convention: uppercase=white, lowercase=black.
    """
    sdir = root / style
    pieces: dict[str, str] = {}
    for piece in PIECE_FILES:
        fp = sdir / f"{piece}.svg"
        text = fp.read_text(encoding="utf-8", errors="ignore")
        sym = piece[1] if piece[0] == "w" else piece[1].lower()
        pieces[sym] = _extract_inner(text)
    return pieces


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/piece_sets"))
    ap.add_argument(
        "--styles",
        nargs="*",
        default=None,
        help="subset of styles to fetch (default: all)",
    )
    args = ap.parse_args()
    counts = download_styles(args.out, args.styles)
    full = [s for s, n in counts.items() if n == 12]
    print(f"\n{len(full)}/{len(counts)} styles complete: {full}")


if __name__ == "__main__":
    main()

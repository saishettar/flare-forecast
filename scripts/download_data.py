"""Download HMP2/IBDMDB source data into data/raw/.

Pulls the merged (whole-cohort) product tables from ibdmdb.org's public
Globus-backed file store: clinical/sample metadata, MetaPhlAn taxonomic
profiles, and HUMAnN pathway + EC functional profiles (bioBakery 3.0
release), plus the metabolomics BIOM table. These are the "Merged Table"
/ "Merged Tables" links on https://ibdmdb.org/results (products_MGX and
products_MBX pages), verified by hand since the site has no API or
manifest file.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --only metadata taxonomy
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

BASE = "https://g-227ca.190ebd.75bc.data.globus.org/ibdmdb"

FILES = {
    "metadata": f"{BASE}/metadata/hmp2_metadata_2018-08-20.csv",
    "taxonomy": f"{BASE}/products/HMP2/MGX/2018-05-04/taxonomic_profiles_3.tsv.gz",
    "pathways": f"{BASE}/products/HMP2/MGX/2018-05-04/pathabundances_3.tsv.gz",
    "ecs": f"{BASE}/products/HMP2/MGX/2018-05-04/ecs_3.tsv.gz",
    "metabolomics": f"{BASE}/products/HMP2/MBX/HMP2_metabolomics_w_metadata.biom.gz",
}


def download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(FILES),
        help="Download only these named products (default: all).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if the file already exists."
    )
    args = parser.parse_args()

    names = args.only or sorted(FILES)
    for name in names:
        url = FILES[name]
        dest = RAW_DIR / Path(url).name
        if dest.exists() and not args.force:
            print(f"[skip] {name}: {dest.name} already present ({dest.stat().st_size:,} bytes)")
            continue
        print(f"[fetch] {name}: {url}")
        try:
            download(url, dest)
        except requests.RequestException as exc:
            print(f"[error] {name}: {exc}", file=sys.stderr)
            continue
        print(f"[done] {name}: {dest.name} ({dest.stat().st_size:,} bytes) "
              f"sha256={sha256sum(dest)[:12]}...")


if __name__ == "__main__":
    main()

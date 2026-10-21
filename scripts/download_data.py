"""Download the public datasets into data/ (gitignored). Safe to re-run: finished files are skipped.

    python scripts/download_data.py               # both datasets
    python scripts/download_data.py track3        # BCI Competition 2020 Track 3 only (~0.6 GB)
    python scripts/download_data.py nieto         # Nieto et al. 2022 inner speech only (~6.6 GB)

Sources
- track3: NEMAR nm000113, a BIDS/EDF conversion (by P. Guetschel) of BCI Competition 2020 Track 3
  (https://osf.io/pq7vb/). The OSF Track 3 folder itself holds only a readme, so NEMAR is the
  public copy. Includes the test-set labels from the organisers' answer sheet.
- nieto: OpenNeuro ds003626 (Nieto et al. 2022, Sci. Data 9:52), the authors' preprocessed epochs
  (derivatives/*_eeg-epo.fif + *_events.dat). Raw BDF, EXG channels and .pkl reports are skipped.
"""

from __future__ import annotations

import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
OPENNEURO_S3 = "https://s3.amazonaws.com/openneuro.org"


def download_track3() -> Path:
    import nemar  # installed with moabb

    target = DATA / "bci2020_track3"
    nemar.download(dataset="nm000113", target_dir=target, trust_existing=True)
    return target


def _s3_list(prefix: str) -> list[tuple[str, int]]:
    keys, token = [], None
    while True:
        url = f"{OPENNEURO_S3}?list-type=2&prefix={prefix}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token)
        xml = urllib.request.urlopen(url, timeout=60).read().decode()
        keys += [(k, int(s)) for k, s in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", xml)]
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def download_nieto() -> Path:
    target = DATA / "nieto2022"
    wanted = [("ds003626/dataset_description.json", None), ("ds003626/README", None)]
    wanted += [(k, s) for k, s in _s3_list("ds003626/derivatives/") if k.endswith(("_eeg-epo.fif", "_events.dat"))]
    # Small files first so the events/labels are available early; S3 throttles each connection,
    # so fetch several files at once.
    wanted.sort(key=lambda ks: ks[1] or 0)
    total = sum(s or 0 for _, s in wanted)
    print(f"nieto: {len(wanted)} files, {total / 1e9:.2f} GB", flush=True)

    def fetch(item: tuple[str, int | None]) -> None:
        key, size = item
        dest = target / key.removeprefix("ds003626/")
        if dest.exists() and (size is None or dest.stat().st_size == size):
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(1, 6):  # S3 sometimes drops long connections; retry with backoff
            try:
                urllib.request.urlretrieve(f"{OPENNEURO_S3}/{key}", tmp)
                if size is not None and tmp.stat().st_size != size:
                    raise OSError(f"size mismatch for {key}: got {tmp.stat().st_size}, expected {size}")
                break
            except OSError as e:  # URLError, RemoteDisconnected and size mismatches are all OSErrors
                if attempt == 5:
                    raise
                print(f"  retry {attempt} for {key}: {e}", flush=True)
                time.sleep(10 * attempt)
        tmp.replace(dest)
        print(f"  done {key} ({(size or 0) / 1e6:.0f} MB)", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(fetch, wanted):
            pass
    return target


def main(argv: list[str]) -> int:
    which = argv or ["track3", "nieto"]
    for name in which:
        fn = {"track3": download_track3, "nieto": download_nieto}.get(name)
        if fn is None:
            print(f"unknown dataset {name!r}; choose track3 or nieto", file=sys.stderr)
            return 2
        print(f"{name}: done -> {fn()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

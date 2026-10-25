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
import shutil
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
OPENNEURO_S3 = "https://s3.amazonaws.com/openneuro.org"  # used for listing files and sizes
OPENNEURO_FILES = "https://openneuro.org/crn/datasets/ds003626/files"  # faster per connection, supports resume


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
    # Small files first so the events/labels are available early. The servers throttle each
    # connection, so fetch several files at once, and resume partial files after interruptions.
    wanted.sort(key=lambda ks: ks[1] or 0)
    total = sum(s or 0 for _, s in wanted)
    print(f"nieto: {len(wanted)} files, {total / 1e9:.2f} GB", flush=True)

    def fetch(item: tuple[str, int | None]) -> None:
        key, size = item
        rel = key.removeprefix("ds003626/")
        dest = target / rel
        if dest.exists() and (size is None or dest.stat().st_size == size):
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        url = f"{OPENNEURO_FILES}/{rel.replace('/', ':')}"
        for attempt in range(1, 11):  # connections drop on long transfers; resume with backoff
            try:
                have = tmp.stat().st_size if tmp.exists() else 0
                if size is not None and have > size:
                    tmp.unlink()
                    have = 0
                if size is None or have < size:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
                    with urllib.request.urlopen(req, timeout=120) as resp:
                        if have and resp.status != 206:  # server ignored the range: start over
                            have = 0
                        with open(tmp, "ab" if have else "wb") as f:
                            shutil.copyfileobj(resp, f, length=1 << 20)
                if size is not None and tmp.stat().st_size != size:
                    raise OSError(f"incomplete {rel}: {tmp.stat().st_size} of {size} bytes")
                break
            except OSError as e:  # URLError, RemoteDisconnected, timeouts and short files are all OSErrors
                if attempt == 10:
                    raise
                print(f"  retry {attempt} for {rel}: {e}", flush=True)
                time.sleep(min(60, 10 * attempt))
        tmp.replace(dest)
        print(f"  done {rel} ({(size or 0) / 1e6:.0f} MB)", flush=True)

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

"""Download the Olist raw CSVs into data/olist/.

Primary source: the canonical Kaggle dataset via kagglehub, which works
for public datasets without a Kaggle account:
    https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Fallback: a public GitHub mirror of the same files, in case kagglehub
or Kaggle is unavailable.

Dataset licence: CC BY-NC-SA 4.0.
"""

import glob
import io
import os
import shutil
import tarfile
import urllib.request

DEST = "data/olist"
MIRROR = ("https://codeload.github.com/mara/mara-olist-ecommerce-data"
          "/tar.gz/refs/heads/master")


def via_kagglehub() -> bool:
    try:
        import kagglehub
    except ImportError:
        print("kagglehub not installed (pip install kagglehub), trying mirror...")
        return False
    try:
        path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
    except Exception as e:
        print(f"kagglehub download failed ({e}), trying mirror...")
        return False
    csvs = glob.glob(os.path.join(path, "*.csv"))
    if not csvs:
        return False
    for f in csvs:
        shutil.copy(f, DEST)
        print("  ", os.path.basename(f))
    return True


def via_mirror():
    print("downloading GitHub mirror (~43 MB)...")
    buf = io.BytesIO(urllib.request.urlopen(MIRROR).read())
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for m in tar.getmembers():
            if "/data/olist-ecommerce/" in m.name and m.name.endswith(".csv"):
                m.name = os.path.basename(m.name)
                tar.extract(m, DEST)
                print("  ", m.name)


if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    if not via_kagglehub():
        via_mirror()
    print("done ->", DEST)

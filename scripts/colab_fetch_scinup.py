from pathlib import Path
from urllib.request import urlretrieve
import tarfile

root = Path("/content/data/SciNUP")
root.mkdir(parents=True, exist_ok=True)
for name in ("dataset", "sampled_users"):
    archive = root / f"{name}.tgz"
    urlretrieve(f"https://gustav1.ux.uis.no/downloads/SciNUP/{name}.tgz", archive)
    with tarfile.open(archive) as bundle:
        bundle.extractall(root, filter="data")
print((root / "dataset.jsonl").stat().st_size)

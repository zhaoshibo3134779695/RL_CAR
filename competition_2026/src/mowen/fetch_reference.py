#!/usr/bin/env python3
"""Materialize the two reference files from their published upstream source."""
import hashlib
from pathlib import Path
import tempfile
from urllib.request import urlopen

PACKAGE = Path(__file__).resolve().parent
SOURCE = ("https://raw.githubusercontent.com/EnochxynL/RAICOM_edu/"
          "62253df73839c9212f6f60932cdd7b65946e4413/1_robot_description/src/robot_description/mowen/")
FILES = {
    "urdf/mowen.urdf": "66b5f29c3eff416f4aa6fc06f178b531122d947e1aaeed6accb183609e5186ea",
    "meshes/base_link.STL": "eca4584c04c6995f0e2dc1cc4cfc5caeca50209f2cca2648004bad1f97299e4f",
}


def main():
    for relative, expected in FILES.items():
        target = PACKAGE / relative
        if target.is_file():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise SystemExit(f"Existing file has a different checksum: {target}")
            continue
        with urlopen(SOURCE + relative, timeout=60) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise SystemExit(f"Upstream checksum differs for {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
            tmp.write(payload)
            staged = Path(tmp.name)
        staged.chmod(0o644)
        staged.replace(target)
        print(f"Verified {relative}")


if __name__ == "__main__":
    main()

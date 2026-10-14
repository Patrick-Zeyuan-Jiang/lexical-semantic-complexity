"""Check the frozen source bytes, without importing the WSD or ML dependencies."""
import argparse
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
freeze = json.loads((root / "FROZEN_SOURCE.json").read_bytes())
expected = freeze["files"]
actual_names = {"ode_wsd/" + p.name for p in (root / "ode_wsd").glob("*.py")}
expected_names = {name for name in expected if name.startswith("ode_wsd/")}
if actual_names != expected_names:
    raise SystemExit("FAILED: package Python file set changed")
for name, expected_hash in expected.items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise SystemExit("FAILED: source bytes changed: " + name)
parser = argparse.ArgumentParser()
parser.add_argument("--bank", help="Optional: compare the existing bank source fingerprint")
args = parser.parse_args()
if args.bank:
    manifest = json.loads((Path(args.bank) / "manifest.json").read_bytes())
    if manifest.get("implementation_sha256") != freeze["implementation_sha256"]:
        raise SystemExit("FAILED: source does not match the bank")
print("PASS: original WSD source bytes preserved")
print("implementation_sha256:", freeze["implementation_sha256"])

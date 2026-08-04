#!/usr/bin/env python3
from pathlib import Path
import sys

root = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else Path.cwd()
src = root / "src" / "PTDiracNHNRG.jl"
test = root / "test" / "biorth_operator_regression.jl"

if not src.exists():
    raise SystemExit(f"ERROR: missing {src}")
text = src.read_text()

required = [
    "last_cre::NTuple{2,Matrix{ComplexF64}}",
    "function coupling_term(old_ann::NTuple{2,Matrix{ComplexF64}},",
    "old_cre::NTuple{2,Matrix{ComplexF64}}",
    "old_cre[a] * Pold",
    "coupling_term(old.last_ann, old.last_cre, old.charges, T)",
    "last_cre_full = ntuple",
    "last_cre_new = ntuple",
]
missing = [item for item in required if item not in text]
if missing:
    print("ERROR: missing required fixed-source markers:")
    for item in missing:
        print("  -", item)
    raise SystemExit(1)

forbidden = [
    "adjoint(old_last[a]) * Pold",
    "coupling_term(old.last_ann, old.charges, T)",
]
present = [item for item in forbidden if item in text]
if present:
    print("ERROR: obsolete shortcut remains:")
    for item in present:
        print("  -", item)
    raise SystemExit(1)

if not test.exists():
    raise SystemExit(f"ERROR: missing {test}")

test_text = test.read_text()
for marker in ["old_shortcut_error", "Hhop_exact", "CAR error", "last_cre"]:
    if marker not in test_text:
        raise SystemExit(f"ERROR: regression test missing marker: {marker}")

print("PASS: independent creation/annihilation propagation is installed")
print(f"source: {src}")
print(f"test:   {test}")

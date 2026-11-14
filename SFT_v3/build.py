import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "SFT"))
import build_sft  

SOURCES = [
    ("seeds.txt", None),
    ("external/global_medical.txt", "explain"),
    (HERE / "new_samples.txt", None),   
]

if __name__ == "__main__":
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    build_sft.main(sources=SOURCES, out_dir=out, fuzzy_cutoff=0.98)
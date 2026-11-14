import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "SFT"))
import build_sft

QUOTA = {
    "harmful": None, "cheating": None, "unknown": None,
    "exam_prep": None, "chat": None, "list": None,
    "factual": None, "explain": 150, "translate": 100
}

rng = random.Random(7)


def main():
    print("[1/3] SFT v3 datasetlari yuklanmoqda...")
    rows = [
        json.loads(line)
        for name in ("train.jsonl", "val.jsonl")
        for line in (ROOT / "SFT_v3" / "data" / name)
        .read_text(encoding="utf-8").splitlines()
    ]

    by_task = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)

    print("[2/3] SFT promptlari tanlanmoqda...")
    prompts = []

    for task, quota in QUOTA.items():
        items = by_task.get(task, [])
        if quota is not None and len(items) > quota:
            items = rng.sample(items, quota)

        for r in items:
            q, a = r["messages"][0]["content"], r["messages"][1]["content"]
            p = {"task": task, "prompt": q, "reference": a}

            if task == "list":
                import re
                p["items"] = int(re.search(r"\d+", q).group())

            prompts.append(p)

    print(f"  SFT promptlari: {len(prompts)}")

    rng.shuffle(prompts)
    for i, p in enumerate(prompts):
        p["id"] = f"p{i:04d}"

    print("[3/3] Natija saqlanmoqda...")
    out = HERE / "data"
    out.mkdir(exist_ok=True)

    (out / "prompts.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts),
        encoding="utf-8"
    )

    for task, n in Counter(p["task"] for p in prompts).most_common():
        print(f"  {task:<12} {n}")

    print(f"  Jami: {len(prompts)} -> {out / 'prompts.jsonl'}")


if __name__ == "__main__":
    main()
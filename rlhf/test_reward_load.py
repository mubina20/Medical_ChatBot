import torch
import rlhf.scalar_model as scalar_model
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
device = "cuda"

folder = ROOT / "rlhf" / "models" / "reward-sft-qlora"

assert torch.cuda.is_available(), "CUDA mavjud emas!"

print("GPU:", torch.cuda.get_device_name(0))
print("Reward Model yuklanmoqda...")

model, meta = scalar_model.load(
    folder,
    device,
    trainable=False,
    dtype=torch.float16
)

print("Reward Model muvaffaqiyatli yuklandi!")
print("Base:", meta["base"])
print("SFT adapter:", meta["sft_adapter"])
print("Device:", next(model.parameters()).device)
print("Dtype:", next(model.parameters()).dtype)
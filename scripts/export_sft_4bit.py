from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import unsloth  # noqa: F401

from lab22 import config as C
from lab22 import modeling as MD


def main() -> None:
    C.ensure_dirs()
    model, tokenizer = MD.load_model(C.SFT_ADAPTER)
    model.save_pretrained_merged(str(C.SFT_MERGED), tokenizer, save_method="merged_4bit_forced")
    print(f"saved {C.SFT_MERGED}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export Persian ALBERT MLM to ONNX.
Tokenizer is loaded from tokenizer.json.
SentencePiece file is NOT required.
"""

import os
import sys
from pathlib import Path

import torch
import onnx

from transformers import AutoTokenizer, AutoModelForMaskedLM


MODEL_NAME = "shekar-ai/albert-base-v2-persian-zwnj-naab-mlm"

ROOT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT_DIR
    / "app"
    / "src"
    / "main"
    / "assets"
    / "semantic"
)

ONNX_FILE = OUTPUT_DIR / "albert_persian_mlm.onnx"

MAX_LENGTH = 32
OPSET_VERSION = 17


class AlbertONNXWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )
        return outputs[0]


def main():

    print("=" * 60)
    print("Persian ALBERT ONNX Export")
    print("Model:", MODEL_NAME)
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Load tokenizer
    # --------------------------------------------------

    print("[1/6] Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise RuntimeError(
            "Fast tokenizer is required for this export."
        )

    print("Tokenizer:", type(tokenizer).__name__)

    # --------------------------------------------------
    # 2. Save tokenizer files
    # --------------------------------------------------

    print("[2/6] Saving tokenizer...")

    tokenizer.save_pretrained(OUTPUT_DIR)

    tokenizer_json = OUTPUT_DIR / "tokenizer.json"

    if not tokenizer_json.exists():
        raise FileNotFoundError(
            "tokenizer.json was not generated."
        )

    print("Tokenizer saved:", tokenizer_json)

    # --------------------------------------------------
    # 3. Load actual pretrained model
    # --------------------------------------------------

    print("[3/6] Loading pretrained ALBERT model...")

    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_NAME,
    )

    model.eval()
    model.cpu()

    wrapper = AlbertONNXWrapper(model)
    wrapper.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------
    # 4. Prepare sample inputs
    # --------------------------------------------------

    print("[4/6] Preparing ONNX inputs...")

    sample = tokenizer(
        "این یک آزمایش برای مدل زبان فارسی است.",
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    )

    input_ids = sample["input_ids"]
    attention_mask = sample["attention_mask"]

    token_type_ids = sample.get(
        "token_type_ids",
        torch.zeros_like(input_ids),
    )

    # --------------------------------------------------
    # 5. Export ONNX
    # --------------------------------------------------

    print("[5/6] Exporting ONNX model...")

    with torch.no_grad():

        torch.onnx.export(
            wrapper,
            (
                input_ids,
                attention_mask,
                token_type_ids,
            ),
            str(ONNX_FILE),

            input_names=[
                "input_ids",
                "attention_mask",
                "token_type_ids",
            ],

            output_names=[
                "logits",
            ],

            dynamic_axes={
                "input_ids": {
                    0: "batch_size",
                    1: "sequence_length",
                },
                "attention_mask": {
                    0: "batch_size",
                    1: "sequence_length",
                },
                "token_type_ids": {
                    0: "batch_size",
                    1: "sequence_length",
                },
                "logits": {
                    0: "batch_size",
                    1: "sequence_length",
                },
            },

            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            export_params=True,
            dynamo=False,
        )

    # --------------------------------------------------
    # 6. Validate ONNX
    # --------------------------------------------------

    print("[6/6] Validating ONNX model...")

    onnx_model = onnx.load(str(ONNX_FILE))

    onnx.checker.check_model(onnx_model)

    if not ONNX_FILE.exists() or ONNX_FILE.stat().st_size == 0:
        raise RuntimeError("ONNX output is empty.")

    print("=" * 60)
    print("EXPORT COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("ONNX:", ONNX_FILE)
    print(
        "ONNX size:",
        round(ONNX_FILE.stat().st_size / (1024 * 1024), 2),
        "MB",
    )

    print("Tokenizer:", tokenizer_json)
    print("Vocabulary size:", len(tokenizer))
    print("Maximum sequence length:", MAX_LENGTH)

    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nEXPORT FAILED:")
        print(str(error))
        sys.exit(1)

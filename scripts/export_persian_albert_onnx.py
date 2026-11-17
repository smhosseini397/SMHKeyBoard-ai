
#!/usr/bin/env python3
"""Download and export Persian ALBERT to ONNX for Android."""

from pathlib import Path
import json
import shutil
import tempfile

import torch
import onnx
import sentencepiece as spm

from transformers import AutoModelForMaskedLM, AutoTokenizer


REPO = "shekar-ai/albert-base-v2-persian-zwnj-naab-mlm"

OUT = Path("app/src/main/assets/semantic")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Persian ALBERT ONNX Export")
print(f"Model: {REPO}")
print("=" * 60)


# --------------------------------------------------
# 1. Load tokenizer
# --------------------------------------------------

print("\n[1/6] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    REPO,
    use_fast=False,
)

print("Tokenizer:", tokenizer.__class__.__name__)


# --------------------------------------------------
# 2. Extract SentencePiece model safely
# --------------------------------------------------

print("\n[2/6] Extracting SentencePiece model...")

with tempfile.TemporaryDirectory() as temp_dir:

    temp_path = Path(temp_dir)

    tokenizer.save_pretrained(temp_path)

    vocab_file = getattr(tokenizer, "vocab_file", None)

    candidates = []

    if vocab_file:
        candidates.append(Path(vocab_file))

    candidates.extend([
        temp_path / "spiece.model",
        temp_path / "sentencepiece.bpe.model",
        temp_path / "tokenizer.model",
    ])

    source_spiece = next(
        (p for p in candidates if p.is_file()),
        None,
    )

    if source_spiece is None:
        raise FileNotFoundError(
            "SentencePiece model could not be located. "
            "Tokenizer files: "
            + ", ".join(p.name for p in temp_path.iterdir())
        )

    destination_spiece = OUT / "spiece.model"

    shutil.copy2(
        source_spiece,
        destination_spiece,
    )

# Validate actual SentencePiece model.

processor = spm.SentencePieceProcessor(
    model_file=str(OUT / "spiece.model")
)

if processor.get_piece_size() <= 0:
    raise RuntimeError("Invalid SentencePiece vocabulary.")

print("SentencePiece vocabulary size:", processor.get_piece_size())


# --------------------------------------------------
# 3. Save exact special token IDs
# --------------------------------------------------

print("\n[3/6] Saving special tokens...")

required_tokens = [
    "cls_token_id",
    "sep_token_id",
    "mask_token_id",
    "unk_token_id",
    "pad_token_id",
]

special = {}

for name in required_tokens:
    value = getattr(tokenizer, name, None)

    if value is None:
        raise RuntimeError(
            f"Missing required tokenizer token: {name}"
        )

    special[name] = int(value)

(OUT / "special_tokens.json").write_text(
    json.dumps(special, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(special, indent=2))


# --------------------------------------------------
# 4. Load pretrained ALBERT
# --------------------------------------------------

print("\n[4/6] Loading pretrained ALBERT model...")

model = AutoModelForMaskedLM.from_pretrained(
    REPO,
    use_safetensors=True,
).cpu().eval()

torch.set_num_threads(1)


class ExportWrapper(torch.nn.Module):

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
    ):

        output = self.wrapped(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        return output[0]


wrapper = ExportWrapper(model).eval()


# --------------------------------------------------
# 5. Export ONNX
# --------------------------------------------------

print("\n[5/6] Exporting ONNX model...")

ids = torch.tensor(
    [[
        special["cls_token_id"],
        special["mask_token_id"],
        special["sep_token_id"],
    ]],
    dtype=torch.long,
)

mask = torch.ones_like(ids)
types = torch.zeros_like(ids)

onnx_path = OUT / "albert_persian_mlm.onnx"

# Remove any stale output before exporting.
if onnx_path.exists():
    onnx_path.unlink()

with torch.inference_mode():

    torch.onnx.export(
        wrapper,
        (ids, mask, types),
        str(onnx_path),

        input_names=[
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ],

        output_names=["logits"],

        dynamic_axes={
            "input_ids": {
                0: "batch",
                1: "sequence",
            },
            "attention_mask": {
                0: "batch",
                1: "sequence",
            },
            "token_type_ids": {
                0: "batch",
                1: "sequence",
            },
            "logits": {
                0: "batch",
                1: "sequence",
            },
        },

        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


# --------------------------------------------------
# 6. Validate exported ONNX
# --------------------------------------------------

print("\n[6/6] Validating exported model...")

if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
    raise RuntimeError("ONNX export failed: empty model file.")

onnx_model = onnx.load(str(onnx_path))

onnx.checker.check_model(onnx_model)

expected_inputs = {
    "input_ids",
    "attention_mask",
    "token_type_ids",
}

actual_inputs = {
    item.name for item in onnx_model.graph.input
}

if not expected_inputs.issubset(actual_inputs):
    raise RuntimeError(
        f"Missing ONNX inputs: {expected_inputs - actual_inputs}"
    )

print("\n" + "=" * 60)
print("PERSIAN ALBERT EXPORT SUCCESSFUL")
print("=" * 60)

for filename in [
    "albert_persian_mlm.onnx",
    "spiece.model",
    "special_tokens.json",
]:

    path = OUT / filename

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty asset: {path}")

    print(f"{path}: {path.stat().st_size:,} bytes")

print("\nAll model assets successfully generated.")

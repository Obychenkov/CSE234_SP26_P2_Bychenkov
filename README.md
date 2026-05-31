# CSE/DSC 234 Project 2

## Inference

Run from the repo root:

```bash
python3 main.py --input input_filename --output output_filename
```

`main.py` expects schemas at `./schemas/` and writes a JSON list of
`question_id` / `schema_links` predictions.

## Model Artifact

The submitted model uses:

```text
Base model: Qwen/Qwen2.5-1.5B-Instruct
Adapter type: LoRA
Adapter path: ./adapter/
```

The base model is loaded from HuggingFace with `transformers`. The local LoRA
adapter is loaded from `./adapter/` with PEFT. No retraining is required for
inference.

## Dependencies

Inference dependencies:

```text
torch
transformers
peft
accelerate
```

Training dependencies:

```text
rapidfireai
datasets
trl
```

## Training And Logs

RapidFire training presets are in `train_sft.py`.

Generated training data scripts:

```text
make_augmented_train.py
make_hardneg_train.py
```

RapidFire logs and experiment artifacts are stored in:

```text
logs/
```

Generated training data used by augmentation configs is stored in:

```text
training_data/
```

Validation predictions and per-question metrics are stored in:

```text
validation_outputs/
```

Non-final adapters and other bulky local artifacts are stored in the ignored
local folder:

```text
artifacts/
```

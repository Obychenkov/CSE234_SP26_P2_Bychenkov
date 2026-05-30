import argparse
import json
import os

from datasets import Dataset
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig

def load_dataset_from_json(path):
    with open(path) as f:
        return Dataset.from_list(json.load(f))


def formatting_function(row):
    from schema_linking_utils import make_messages

    return make_messages(row, schemas_dir="schemas", include_completion=True)


def create_model(model_config):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = model_config["model_name"]
    model_kwargs = dict(model_config["model_kwargs"])
    if torch.cuda.is_available():
        model_kwargs.setdefault("torch_dtype", torch.bfloat16)
        model_kwargs.setdefault("device_map", "auto")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def supports_bf16():
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def build_config(args):
    peft_configs = List([
        RFLoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
    ])

    return List([
        RFModelConfig(
            model_name=args.model_name,
            peft_config=peft_configs,
            training_args=RFSFTConfig(
                output_dir=args.output_dir,
                learning_rate=args.learning_rate,
                lr_scheduler_type="linear",
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.train_batch_size,
                per_device_eval_batch_size=args.eval_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_length=args.max_seq_length,
                logging_steps=5,
                eval_strategy="epoch",
                save_strategy="epoch",
                bf16=supports_bf16(),
                gradient_checkpointing=True,
                packing=False,
            ),
            model_type="causal_lm",
            model_kwargs={"use_cache": False},
            formatting_func=formatting_function,
        )
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.json")
    parser.add_argument("--validation", default="validation.json")
    parser.add_argument("--experiment_name", default="p2-qwen25-05b-lora-json-1epoch")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", default="rapidfire_outputs/qwen25_05b_lora_json_1epoch")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_dataset = load_dataset_from_json(args.train).shuffle(seed=args.seed)
    eval_dataset = load_dataset_from_json(args.validation).shuffle(seed=args.seed)

    experiment = Experiment(experiment_name=args.experiment_name, mode="fit")
    try:
        config_group = RFGridSearch(configs=build_config(args), trainer_type="SFT")
        experiment.run_fit(
            config_group,
            create_model,
            train_dataset,
            eval_dataset,
            num_chunks=args.num_chunks,
            seed=args.seed,
        )
    finally:
        experiment.end()


if __name__ == "__main__":
    main()

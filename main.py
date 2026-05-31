import argparse
import json
import os

from schema_linking_utils import extract_first_json_object, make_messages, sanitize_links


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class SchemaLinker:
    def __init__(
        self,
        model_name=DEFAULT_MODEL,
        adapter_path="./adapter",
        schemas_dir="./schemas",
        max_new_tokens=256,
        schema_mode="candidate",
        max_candidate_tables=40,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.schemas_dir = schemas_dir
        self.max_new_tokens = max_new_tokens
        self.schema_mode = schema_mode
        self.max_candidate_tables = max_candidate_tables
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if torch.cuda.is_available():
            model_kwargs.update({"torch_dtype": torch.bfloat16, "device_map": "auto"})
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        if adapter_path and os.path.isdir(adapter_path):
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        if not torch.cuda.is_available():
            self.model.to("cpu")
        self.model.eval()

    def predict_one(self, item):
        import torch

        messages = make_messages(
            item,
            self.schemas_dir,
            include_completion=False,
            schema_mode=self.schema_mode,
            max_candidate_tables=self.max_candidate_tables,
        )["prompt"]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion_ids = generated[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        raw_links = extract_first_json_object(text)
        return sanitize_links(raw_links, item["db_id"], self.schemas_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--schemas_dir", default="./schemas")
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--adapter_path", default="./adapter")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--schema_mode", choices=["full", "candidate", "candidate_fk"], default="candidate")
    parser.add_argument("--max_candidate_tables", type=int, default=40)
    args = parser.parse_args()

    with open(args.input) as f:
        items = json.load(f)

    linker = SchemaLinker(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        schemas_dir=args.schemas_dir,
        max_new_tokens=args.max_new_tokens,
        schema_mode=args.schema_mode,
        max_candidate_tables=args.max_candidate_tables,
    )

    preds = []
    for item in items:
        preds.append({
            "question_id": item["question_id"],
            "schema_links": linker.predict_one(item),
        })

    with open(args.output, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"Wrote {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()

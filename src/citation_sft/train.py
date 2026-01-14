from __future__ import annotations

import json
import random

from pathlib import Path

import torch

from datasets import Dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer
from transformers import TrainingArguments

from citation_sft.settings import settings
from citation_sft.sft_data.sft_data import SFTExample


random.seed(42)


PROMPT_DIR = Path(__file__).parent / "prompts"
PROMPT_TEMPLATE = (PROMPT_DIR / "citation_sft.txt").read_text()
DOC_TEMPLATE = (PROMPT_DIR / "document.txt").read_text()


def format_prompt(ex: SFTExample) -> str:
    docs_text = "\n\n".join(
        DOC_TEMPLATE.format(index=d.index, title=d.title, text=d.text) for d in ex.docs
    )
    return PROMPT_TEMPLATE.format(documents=docs_text, question=ex.question)


def load_sft_dataset(
    path: str, test_questions: int = 5
) -> tuple[list[SFTExample], list[SFTExample]]:
    with open(path) as f:
        examples = [SFTExample(**ex) for ex in json.load(f)]

    grouped: dict[str, list[SFTExample]] = {}
    for ex in examples:
        base_id = ex.id.split("_aug")[0]
        grouped.setdefault(base_id, []).append(ex)

    base_ids = list(grouped.keys())
    random.shuffle(base_ids)
    test_ids = set(base_ids[:test_questions])
    train_ids = set(base_ids[test_questions:])

    train = [ex for qid in train_ids for ex in grouped[qid]]
    test = [ex for qid in test_ids for ex in grouped[qid]]

    return train, test


def preprocess(examples: list[SFTExample], tokenizer) -> Dataset:
    formatted = []

    for ex in examples:
        prompt = format_prompt(ex)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ex.label},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        # Remove Qwen3 thinking tags (enable_thinking=False doesn't work reliably)
        text = text.replace("<think>\n\n</think>\n\n", "")
        
        # Find where label starts to get accurate prompt_len
        label_start = text.find(ex.label)
        if label_start == -1:
            raise ValueError(f"Label not found in text for example {ex.id}")
        
        prompt_text = text[:label_start]
        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        formatted.append({"text": text, "prompt_len": prompt_len, "id": ex.id})

    return Dataset.from_list(formatted)


def tokenize(example, tokenizer):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=settings.max_length,
        padding=False,
    )

    labels = tokens["input_ids"].copy()
    prompt_len = min(example["prompt_len"], len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    tokens["labels"] = labels

    return tokens


def test_forward_pass(model, tokenizer, dataset):
    """Quick sanity check: run a single forward pass to verify everything works."""
    sample = dataset[0]

    inputs = {
        "input_ids": torch.tensor([sample["input_ids"]]),
        "attention_mask": torch.tensor([sample["attention_mask"]]),
        "labels": torch.tensor([sample["labels"]]),
    }

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    prompt_len = int(sample.get("prompt_len", 0))

    prompt_text = tokenizer.decode(
        sample["input_ids"][:prompt_len],
        skip_special_tokens=False
    )
    completion_text = tokenizer.decode(
        sample["input_ids"][prompt_len:],
        skip_special_tokens=False
    )

    trained_token_ids = [
        tid for tid, lab in zip(sample["input_ids"], sample["labels"], strict=False) if lab != -100
    ]
    trained_text = tokenizer.decode(trained_token_ids, skip_special_tokens=False)

    print("Forward pass OK!")
    print(f"  Input length: {len(sample['input_ids'])}")
    print(f"  Loss: {outputs.loss.item():.4f}")
    print(f"  prompt_len: {prompt_len}")
    print(f"  Masked tokens: {sample['labels'].count(-100)}/{len(sample['labels'])}")
    print(f"\n{'='*50}\nPROMPT:\n{prompt_text}")
    print(f"\n{'='*50}\nCOMPLETION (raw tail of input_ids):\n{completion_text}")
    print(f"\n{'='*50}\nTRAINED TOKENS (labels != -100):\n{trained_text}")
    
    
def main():
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    train_examples, test_examples = load_sft_dataset(
        f"{settings.data_dir}/sft_dataset.json"
    )
    print(f"Train: {len(train_examples)}, Test: {len(test_examples)}")

    train_dataset = preprocess(train_examples, tokenizer)
    test_dataset = preprocess(test_examples, tokenizer)

    train_dataset = train_dataset.map(
        lambda x: tokenize(x, tokenizer), remove_columns=["text", "id"]
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize(x, tokenizer), remove_columns=["text", "id"]
    )

    # test_forward_pass(model, tokenizer, train_dataset)
    # return

    training_args = TrainingArguments(
        output_dir=f"{settings.data_dir}/checkpoints",
        logging_dir=f"{settings.data_dir}/runs",
        num_train_epochs=settings.num_train_epochs,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        per_device_eval_batch_size=settings.per_device_eval_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        warmup_ratio=settings.warmup_ratio,
        logging_steps=settings.logging_steps,
        eval_strategy=settings.eval_strategy,
        save_strategy=settings.save_strategy,
        save_total_limit=settings.save_total_limit,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    trainer.train()
    trainer.save_model(f"{settings.data_dir}/final_model")
    tokenizer.save_pretrained(f"{settings.data_dir}/final_model")


if __name__ == "__main__":
    main()

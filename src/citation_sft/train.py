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


PROMPT_DIR = Path(__file__).parent / "prompts"
PROMPT_TEMPLATE = (PROMPT_DIR / "citation_sft.txt").read_text()
DOC_TEMPLATE = (PROMPT_DIR / "document.txt").read_text()


def format_prompt(ex: SFTExample) -> str:
    docs_text = "\n\n".join(
        DOC_TEMPLATE.format(index=d.index, title=d.title, text=d.text) for d in ex.docs
    )
    return PROMPT_TEMPLATE.format(documents=docs_text, question=ex.question)


def load_sft_dataset(path: str, test_questions: int = 5, seed: int = 42) -> tuple[list[SFTExample], list[SFTExample]]:
    with open(path) as f:
        examples = [SFTExample(**ex) for ex in json.load(f)]
    
    grouped: dict[str, list[SFTExample]] = {}
    for ex in examples:
        base_id = ex.id.split("_aug")[0]
        grouped.setdefault(base_id, []).append(ex)
    
    base_ids = list(grouped.keys())
    random.seed(seed)
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
        
        if tokenizer.chat_template:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ex.label},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # Fallback for models without chat template -> I tested with sshleifer/tiny-gpt2
            text = f"{prompt}{ex.label}{tokenizer.eos_token}"
        
        formatted.append({"text": text, "id": ex.id})
    return Dataset.from_list(formatted)

def tokenize(example, tokenizer):
    tokens = tokenizer(
        example["text"],
        max_length=settings.max_length,
        truncation=True,
        padding=False,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

def main():
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    train_examples, test_examples = load_sft_dataset(f"{settings.data_dir}/sft_dataset.json")
    print(f"Train: {len(train_examples)}, Test: {len(test_examples)}")

    train_dataset = preprocess(train_examples, tokenizer)
    test_dataset = preprocess(test_examples, tokenizer)

    train_dataset = train_dataset.map(
        lambda x: tokenize(x, tokenizer), remove_columns=["text", "id"]
    )
    test_dataset = test_dataset.map(
        lambda x: tokenize(x, tokenizer), remove_columns=["text", "id"]
    )
    
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
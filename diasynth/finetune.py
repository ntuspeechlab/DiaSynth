from transformers import AutoTokenizer, LEDForConditionalGeneration, T5ForConditionalGeneration, BartForConditionalGeneration, DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
import pandas as pd
import os
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from argparse import ArgumentParser
from functools import partial
import torch
import random
from .summarization_metrics import score_summary
from .relevance_redund import final_evaluation_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'running on {device}')

def generate_summaries(dialogues_summaries, model, tokenizer, input_max_length, output_max_length):
    """Generates summaries for a list of dialogues."""
    for i in range(len(dialogues_summaries)):
        inputs = tokenizer(dialogues_summaries[i]["dialogue"], return_tensors="pt", max_length=input_max_length, truncation=True).to(device)
        summary_ids = model.generate(inputs['input_ids'], max_length=output_max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        dialogues_summaries[i]["predicted_summary"] = summary


def convert_examples_to_features(example_batch, tokenizer, input_max_length: int, output_max_length: int):
    input_encodings = tokenizer(example_batch["dialogue"], max_length=input_max_length, padding=True, truncation=True, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch["summary"], max_length=output_max_length, padding=True, truncation=True, return_tensors="pt")
    return {"input_ids": input_encodings["input_ids"].to(device),
            "attention_mask": input_encodings["attention_mask"].to(device),
            "labels": target_encodings["input_ids"].to(device)
        }


def finetune(csv_path: str, input_max_length: int, output_max_length: int, dialogue_base: str, model_id: str):

    base_model = csv_path.split('/')[-1].split('_')[0]
    root_folder = model_id.replace('/', '_').replace('-', '_') + f'_{base_model}'
    log_filename = f"{root_folder}/logs_{dialogue_base}.txt"
    if not os.path.exists(root_folder): os.mkdir(root_folder)
    dialogue_ds = load_dataset("knkarthick/dialogsum" if dialogue_base=="dialoguesum" else "Samsung/samsum", trust_remote_code=True)
    dialogue_ds = dialogue_ds.remove_columns([col for col in dialogue_ds['train'].features if col not in ["id", "dialogue", "summary"]])
    dialogue_ds_test = Dataset.from_pandas(pd.read_csv(f"data/{dialogue_base}_test.csv"))

    model_map = {
            "allenai/led-base-16384": LEDForConditionalGeneration,
            "sshleifer/distilbart-cnn-12-6": BartForConditionalGeneration,
            "facebook/bart-base": BartForConditionalGeneration,
            "google-t5/t5-base": T5ForConditionalGeneration
     }
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = model_map[model_id].from_pretrained(model_id).to(device)
    ds = Dataset.from_pandas(pd.read_csv(csv_path))
    ds = ds.remove_columns([col for col in ds.features if col not in ["dialogue", "summary"]])
    sd_dict = DatasetDict({"train": ds, "test": dialogue_ds_test})
    train_sample = dialogue_ds['train'].select(random.sample(range(len(dialogue_ds['train'])), len(ds['dialogue'])))
    dialogue_ds_dict = DatasetDict({"train": train_sample, "test": dialogue_ds_test})
    print(sd_dict)

    dialogues_summaries = dialogue_ds_test.to_pandas().to_dict(orient="records") # contains id, dialogue, summary

    generate_summaries(dialogues_summaries=dialogues_summaries, model=model, tokenizer=tokenizer, input_max_length=input_max_length, output_max_length=output_max_length)
    score_before_finetuning = final_evaluation_score(documents=[d["dialogue"] for d in dialogues_summaries],
                                                     summaries=[d["predicted_summary"] for d in dialogues_summaries])
    with open(log_filename, "a") as f:
        f.write(f"score before finetuning: {score_before_finetuning}\n")

    fn = partial(convert_examples_to_features, tokenizer=tokenizer, input_max_length=input_max_length, output_max_length=output_max_length)
    sd_dict_pt = sd_dict.map(fn, batched=True)
    dialogue_ds_dict_pt = dialogue_ds_dict.map(fn, batched=True)
    columns = ["input_ids", "labels", "attention_mask"]
    sd_dict_pt = sd_dict_pt.remove_columns([col for col in sd_dict_pt.column_names['train'] if col not in columns])
    dialogue_ds_dict_pt = dialogue_ds_dict_pt.remove_columns([col for col in dialogue_ds_dict_pt.column_names['train'] if col not in columns])

    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=f"{root_folder}/sdg_{dialogue_base}",
        num_train_epochs=2,
        warmup_steps=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        logging_steps=0,
        evaluation_strategy='steps',
        eval_steps=50,
        save_steps=0,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        push_to_hub=False,
        logging_dir=None
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dialogue_ds_dict_pt['train'],
        eval_dataset=dialogue_ds_dict_pt['test'],
        data_collator=seq2seq_data_collator
    )
    trainer.train()
    print(f'training done on in-domain data, generating summaries now..')
    generate_summaries(dialogues_summaries=dialogues_summaries, model=model, tokenizer=tokenizer, input_max_length=input_max_length, output_max_length=output_max_length)

    score_after_finetuning_indomain = final_evaluation_score(documents=[d["dialogue"] for d in dialogues_summaries],
                                                     summaries=[d["predicted_summary"] for d in dialogues_summaries])
    with open(log_filename, "a") as f:
        f.write(f"score after finetuning on in-domain data: {score_after_finetuning_indomain}\n")


    del model
    model = model_map[model_id].from_pretrained(model_id).to(device)


    training_args = TrainingArguments(
        output_dir=f"{root_folder}/sdg_{dialogue_base}",
        num_train_epochs=2,
        warmup_steps=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        logging_steps=0,
        evaluation_strategy='steps',
        eval_steps=50,
        save_steps=0,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        push_to_hub=False,
        logging_dir=None
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=sd_dict_pt['train'],
        eval_dataset=sd_dict_pt['test'],
        data_collator=seq2seq_data_collator
    )
    trainer.train()
    print(f'training done on sdg, generating summaries now..')
    generate_summaries(dialogues_summaries=dialogues_summaries, model=model, tokenizer=tokenizer, input_max_length=input_max_length, output_max_length=output_max_length)

    score_after_finetuning_sdg = final_evaluation_score(documents=[d["dialogue"] for d in dialogues_summaries],
                                                     summaries=[d["predicted_summary"] for d in dialogues_summaries])
    with open(log_filename, "a") as f:
        f.write(f"score after finetuning on sdg: {score_after_finetuning_sdg}\n")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--input_max_length", type=int, required=True)
    parser.add_argument("--output_max_length", type=int, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dialogue_base", type=str, default="dialoguesum")

    args = parser.parse_args()
    finetune(csv_path=args.csv_path,
             input_max_length=args.input_max_length,
             output_max_length=args.output_max_length,
             model_id=args.model_id,
             dialogue_base=args.dialogue_base)
from transformers import AutoTokenizer, LEDForConditionalGeneration, T5ForConditionalGeneration, BartForConditionalGeneration, DataCollatorForSeq2Seq
from transformers import TrainingArguments, Trainer
import pandas as pd
import os
from datasets import load_dataset, DatasetDict, Dataset 
from argparse import ArgumentParser
from functools import partial
import torch
from rouge_score import rouge_scorer
from bert_score import score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'running on {device}')

def generate_responses(contexts_responses, model, tokenizer, input_max_length):
    for i in range(len(contexts_responses)):
        inputs = tokenizer(contexts_responses[i]["context"], return_tensors="pt", max_length=input_max_length, truncation=True).to(device)
        response_ids = model.generate(inputs['input_ids'], max_length=100, min_length=10, num_beams=4, early_stopping=True)
        response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        contexts_responses[i]["predicted_response"] = response


def score_responses(contexts_responses):

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    predictions = [item["predicted_response"] for item in contexts_responses]
    references = [item["response"] for item in contexts_responses]
    
    rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure 
                   for ref, pred in zip(references, predictions)]
    
    _, _, F1 = score(predictions, references, lang='en', verbose=False)
    
    return {
        'rouge_l': rouge_scores,
        'bert_score': F1.tolist()
    }


def convert_examples_to_features(example_batch, tokenizer, input_max_length: int):
    input_encodings = tokenizer(example_batch["context"], max_length=input_max_length, padding=True, truncation=True, return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch["response"], max_length=100, padding=True, truncation=True, return_tensors="pt")
    return {
        "input_ids": input_encodings["input_ids"].to(device),
        "attention_mask": input_encodings["attention_mask"].to(device),
        "labels": target_encodings["input_ids"].to(device)
    }


def prepare_dialogue_data(dialogue):
    utterances = dialogue.split('\n')
    if len(utterances) < 2:
        return None, None
    context = '\n'.join(utterances[:-1])
    response = utterances[-1]
    return context, response


def finetune(csv_path: str, input_max_length: int, dialogue_base: str, model_id: str):
    base_model = csv_path.split('/')[-1].split('_')[0]
    root_folder = model_id.replace('/', '_').replace('-', '_') + f'_{base_model}'
    log_filename = f"{root_folder}/logs_{dialogue_base}_response.txt"
    if not os.path.exists(root_folder): 
        os.mkdir(root_folder)

    dialogue_ds = load_dataset("knkarthick/dialogsum" if dialogue_base=="dialoguesum" else "Samsung/samsum", trust_remote_code=True)
    
    test_df = pd.read_csv(f"data/{dialogue_base}_test.csv")
    test_contexts, test_responses = [], []
    for dialogue in test_df['dialogue']:
        context, response = prepare_dialogue_data(dialogue)
        if context is not None and response is not None:
            test_contexts.append(context)
            test_responses.append(response)
    dialogue_ds_test = Dataset.from_pandas(pd.DataFrame({'context': test_contexts, 'response': test_responses}))

    model_map = {
        "allenai/led-base-16384": LEDForConditionalGeneration,
        "sshleifer/distilbart-cnn-12-6": BartForConditionalGeneration,
        "facebook/bart-base": BartForConditionalGeneration,
        "google-t5/t5-base": T5ForConditionalGeneration
    }

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = model_map[model_id].from_pretrained(model_id).to(device)

    syn_df = pd.read_csv(csv_path)
    syn_contexts, syn_responses = [], []
    for dialogue in syn_df['dialogue']:
        context, response = prepare_dialogue_data(dialogue)
        if context is not None and response is not None:
            syn_contexts.append(context)
            syn_responses.append(response)
    ds = Dataset.from_pandas(pd.DataFrame({'context': syn_contexts, 'response': syn_responses}))

    sd_dict = DatasetDict({"train": ds, "test": dialogue_ds_test})
    
    train_contexts, train_responses = [], []
    for dialogue in dialogue_ds['train']['dialogue']:
        context, response = prepare_dialogue_data(dialogue)
        if context is not None and response is not None:
            train_contexts.append(context)
            train_responses.append(response)
    train_df = pd.DataFrame({'context': train_contexts, 'response': train_responses})
    train_sample = Dataset.from_pandas(train_df.sample(n=len(ds), random_state=42))
    dialogue_ds_dict = DatasetDict({"train": train_sample, "test": dialogue_ds_test})

    test_data = dialogue_ds_test.to_pandas().to_dict(orient="records")

    fn = partial(convert_examples_to_features, tokenizer=tokenizer, input_max_length=input_max_length)
    required_columns = ["input_ids", "attention_mask", "labels"]
    print(sd_dict)
    sd_dict_pt = sd_dict.map(fn, batched=True, remove_columns=[col for col in sd_dict['train'].column_names if col not in required_columns])
    print(sd_dict_pt)
    
   
    
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = TrainingArguments(
        output_dir=f"{root_folder}/sdg_{dialogue_base}_response",
        num_train_epochs=2,
        warmup_steps=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=50,
        save_steps=500,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        push_to_hub=False,
        remove_unused_columns=False
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
    print(f'Training done on synthetic data, generating responses now...')
    generate_responses(test_data, model, tokenizer, input_max_length)
    post_synthetic_scores = score_responses(test_data)
    
    with open(log_filename, 'a') as f:
        f.write(f'Scores after synthetic data training:\n')
        print(f'Scores after synthetic data training:')
        for metric, score_list in post_synthetic_scores.items():
            f.write(f"\t{metric}: {sum(score_list)/len(score_list)}\n")
            print(f"\t{metric}: {sum(score_list)/len(score_list)}")

    dialogue_ds_pt = dialogue_ds_dict.map(fn, batched=True)
    dialogue_ds_pt = dialogue_ds_pt.remove_columns([col for col in dialogue_ds_pt.column_names['train'] if col not in required_columns])

    del model
    model = model_map[model_id].from_pretrained(model_id).to(device)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dialogue_ds_pt['train'],
        eval_dataset=dialogue_ds_pt['test'],
        data_collator=seq2seq_data_collator
    )

    trainer.train()
    print(f'Training done on in-domain data, generating responses now...')
    generate_responses(test_data, model, tokenizer, input_max_length)
    post_indomain_scores = score_responses(test_data)
    
    with open(log_filename, 'a') as f:
        f.write(f'Scores after in-domain training:\n')
        print(f'Scores after in-domain training:')
        for metric, score_list in post_indomain_scores.items():
            f.write(f"\t{metric}: {sum(score_list)/len(score_list)}\n")
            print(f"\t{metric}: {sum(score_list)/len(score_list)}")
    
    del model

    model = model_map[model_id].from_pretrained(model_id).to(device) 
    print('Generating responses with base model...')
    generate_responses(test_data, model, tokenizer, input_max_length)
    base_scores = score_responses(test_data)
    with open(log_filename, 'w') as f:
        f.write(f'Base model scores:\n')
        print(f'Base model scores:')
        for metric, score_list in base_scores.items():
            f.write(f"\t{metric}: {sum(score_list)/len(score_list)}\n")
            print(f"\t{metric}: {sum(score_list)/len(score_list)}")
    
    # trainer.save_model(f"{model_id.replace('/', '_').replace('-', '_')}_{base_model}_{dialogue_base}_response")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--input_max_length", type=int, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--dialogue_base", type=str, default="dialoguesum")

    args = parser.parse_args()
    finetune(csv_path=args.csv_path,
             input_max_length=args.input_max_length,
             model_id=args.model_id,
             dialogue_base=args.dialogue_base)
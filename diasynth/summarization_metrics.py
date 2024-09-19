import os
import ast
import pickle
from llama_cpp import Llama
from transformers import AutoTokenizer
from typing import List, Dict
import re
import string
from collections import Counter
from bert_score import BERTScorer
import evaluate
from .prompts import qa_prompt, qg_prompt
import torch

model_path = "models/internlm"
device = "cuda" if torch.cuda.is_available() else "cpu"
n_gpu_layers = 40 if device=="cuda" else 0
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'), trust_remote_code=True)
model = Llama(
    model_path=os.path.join(model_path, 'model.gguf'), 
    n_ctx = 4096, 
    verbose = False,
    n_gpu_layers=n_gpu_layers
)


def calculate_bertscore(dialogues_summaries) -> List[float]:
    """
    calculates the BERTScore summary and predicted_summary
    """
    scorer = BERTScorer(model_type='bert-base-uncased')
    reference_summaries = [d['summary'] for d in dialogues_summaries]
    predicted_summaries = [d['predicted_summary'] for d in dialogues_summaries]
    _, _, f1 = scorer.score(refs=reference_summaries, cands=predicted_summaries)
    return f1.cpu().numpy().tolist()


def generate_answers(passage: str, questions: List[str]):
    """generates answers for the given questions based on the passage"""
    prompt = tokenizer.apply_chat_template(
        conversation = [
            {"role": "system", "content": qa_prompt},
            {"role": "user", "content": f'Passage: {passage}. Questions: {questions}'}
        ],
        tokenize=False, add_generation_prompt = True
    )
    op = model(
        prompt=prompt,
        max_tokens=1024,
        temperature=0.0,
        stop=["<|im_end|>"],
        echo=False
    )["choices"][0]["text"]
    try:
        start_idx, end_idx = op.find('['), op.find(']')
        return ast.literal_eval(op[start_idx:end_idx+1])
    except Exception as e:
        print(f'ran into error {e}, returning the full text')
        return None


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s)))) 


def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()


def f1_score(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def aggregate_score(scores: List[str], n_qns_per_doc: int):
    agg_scores = []
    for i in range(0, len(scores), n_qns_per_doc): agg_scores.append(sum(scores[i:i+n_qns_per_doc])/n_qns_per_doc)
    return agg_scores


def generate_qags_questions(dialogues_summaries: List[str], dialogue_base, num_of_questions:int=3):
    """generates 'num_of_questions' questions given the summary"""
    questions = {}
    for _dict in dialogues_summaries:
        prompt = tokenizer.apply_chat_template(
            conversation = [
                {"role": "system", "content": qg_prompt.format(num_of_questions=num_of_questions)},
                {"role": "user", "content": f'Dialog - {_dict["dialogue"]}'}
            ],
            tokenize = False, 
            add_generation_prompt = True
        )
        op = model(
            prompt = prompt,
            max_tokens = 1024,
            temperature = 0.01,
            stop=["<|user|>"],
            echo=False
        )['choices'][0]['text']
        try:
            start_idx, end_idx = op.find('['), op.find(']')
            questions[_dict['id']] = ast.literal_eval(op[start_idx:end_idx+1])
        except Exception as e:
            print(f'ran into error {e}, skipping id {_dict["id"]}')
    
    with open(f"qags_questions_{dialogue_base}.pkl", "wb") as f:
        pickle.dump(questions, f)
    
    print(f'questions generated successfully:)')


def calculate_qags(dialogues_summaries: List[Dict[str, str]], dialogue_base: str, num_of_questions: int=3) -> List[float]:
    """
    takes in a list of dialogs and its corresponding summaries
    generates questions from summary and two sets of answers from dialog and summary
    calculates and returns the avg f1 score for each pair
    """
    if not os.path.exists(f"qags_questions_{dialogue_base}.pkl"):
        print(f'questions don"t exist, creating them...')
        generate_qags_questions(dialogues_summaries, dialogue_base=dialogue_base)
    else:
        print(f'questions for {dialogue_base} exist, loading them...')
    
    with open(f"qags_questions_{dialogue_base}.pkl", "rb") as f:
        questions = pickle.load(f)

    answers_from_dialogue, answers_from_summary = [], []
    qas = []
    for _dict in dialogues_summaries:
        if _dict['id'] in questions:
            ads = generate_answers(_dict['dialogue'], questions[_dict['id']])
            ass = generate_answers(_dict['predicted_summary'], questions[_dict['id']])
            if ads is not None and ass is not None and len(ads)==len(ass):
                qas.append(dict(dialog=_dict['dialogue'], summary=_dict['predicted_summary'], questions=questions[_dict['id']], answers_from_dialogue=ads, answers_from_summary=ass))
                answers_from_dialogue.extend(ads)
                answers_from_summary.extend(ass) 

    print(f'num of answer: {len(answers_from_dialogue)} num of ans from summary: {len(answers_from_summary)}')
    
    scores = calculate_bertscore([{'summary':_ad, 'predicted_summary': _as} for _ad, _as in zip(answers_from_dialogue, answers_from_summary)])
    agg_scores = aggregate_score(scores, num_of_questions)
    return agg_scores, qas


def score_summary(dialogues_summaries: List[Dict[str, str]], dialogue_base: str):
    rouge = evaluate.load('rouge')
    return {
        "qags": calculate_qags(dialogues_summaries=dialogues_summaries, num_of_questions=3, dialogue_base=dialogue_base)[0],
        "bert_score": calculate_bertscore(dialogues_summaries),
        'rougeL': rouge.compute(predictions=[d['predicted_summary'] for d in dialogues_summaries],
                                references=[d['summary'] for d in dialogues_summaries],
                                use_aggregator=False)['rougeL'] 
    }


if __name__=="__main__":
    from argparse import ArgumentParser
    import pandas as pd
    parser = ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    args = parser.parse_args() 
    df = pd.read_csv(args.csv_path)
    dic_ = []
    for i in range(df.shape[0]):
        dic_.append({k:df[k][i] for k in ['dialogue', 'predicted_summary', 'summary']})
    
    rouge = evaluate.load('rouge')
    bert_score = calculate_bertscore(dic_)
    rouge_score = rouge.compute(predictions=[d['predicted_summary'] for d in dic_],
            references=[d['summary'] for d in dic_], use_aggregator=False)['rougeL']
    print(f'bert_score: {sum(bert_score)/len(bert_score)}')
    print(f'rouge_score: {sum(rouge_score)/len(rouge_score)}')
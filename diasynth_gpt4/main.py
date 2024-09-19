from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from llama_cpp import Llama
from typing import List, Dict
from collections import defaultdict
from itertools import combinations
import pandas as pd
import os
from openai import OpenAI
from functools import partial
import random
from argparse import ArgumentParser
import json
from .utils import gen_dialogue_util, generate_sub_topics, generate_personas, llm_judge
from .quality_metrics import calculate_metrics, g_eval, gpt_score, fed
from .prompts import dialog_system_prompt_cot, summ_prompt


client = OpenAI()

def generate_dialogues(topic_personas:Dict[str,List[str]], system_prompt:str, sample_dialogues: List[Dict[str, str]]) -> pd.DataFrame:
    """
    generates dialogs given the topic and personas and puts them into a nice dataframe
    """
    df = pd.DataFrame(columns=['topic', 'persona_1', 'persona_2', 'dialogue', 'cot'])
    for topic in topic_personas:
        for persona_1, persona_2 in combinations(topic_personas[topic], 2):
            cot, dialogue = gen_dialogue_util(system_prompt=system_prompt, 
                                            args=dict(persona_1=persona_1, persona_2=persona_2, topic=topic),
                                            sample_dialogues=sample_dialogues)
            new_row = pd.DataFrame(data={"topic": [topic], "persona_1": [persona_1], "persona_2": [persona_2], "dialogue": [dialogue], 'cot': [cot]})
            df = pd.concat([df, new_row], axis=0, ignore_index=True)
    return df


def generate_summary(df: pd.DataFrame, sample_dialogues: List[Dict[str, str]]) -> pd.DataFrame:
    """generates summaries for the dialogs given"""
    summaries = []
    for i in range(df.shape[0]):
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "cotent": summ_prompt.format(
                    dialogue_1=sample_dialogues[0]['dialogue'], summary_1=sample_dialogues[0]['summary'],
                    dialogue_2=sample_dialogues[1]['dialogue'], summary_2=sample_dialogues[1]['summary'],
                )},
                {"role": "user", "content": f"Dialog:\n{df.iat[i, 3]}"}
            ],
            temperature=0.3, 
            max_tokens=256
        )
        op = completion.choices[0].message
        summaries.append(op)
    df['summary'] = summaries
    return df


def sdg(csv_path:str, topics_file:str=None, n_sub_topics:int=None, n_personas:int=None, stage:str='scratch', 
        disable_dialog_quality_metrics: bool=False, disable_judge: bool=False, 
        dialogue_base:str='dialoguesum') -> pd.DataFrame:
    
    if stage not in ['scratch', 'metrics', 'llm_judge', 'summary']:
        raise ValueError(f'invalid stage value: {stage}')
    
    dialogs = []
    with open(f'data/{dialogue_base}_combined.jsonl', 'r') as f:
        print(f'loading {dialogue_base} conversations')
        for line in f: dialogs.append(json.loads(line))
        sample_dialogues = random.sample(dialogs, k=4)

    if stage=='scratch':
        if not os.path.exists(topics_file):
            raise OSError(f"{topics_file} does not exist")
        # read the topics file
        with open(topics_file, "r") as f: topics = [topic.strip() for topic in f.read().splitlines()]
        # generate dialogs for each topic
        df = pd.DataFrame(columns=['topic', 'persona_1', 'persona_2', 'dialogue', 'cot'])
        
        for base_topic in topics:
            print(f'processing topic: {base_topic}')
            try:
                print('\tgenerating sub topics')
                sub_topics = generate_sub_topics(topic=base_topic, n_sub_topics=n_sub_topics)
                # for each topic generate a set of personas who are likely to talk about that topic
                print('\tgenerating personas')
                sub_topic_personas = defaultdict(list)
                for sub_topic in sub_topics:
                    personas = generate_personas(topic=sub_topic, n_personas=n_personas)
                    sub_topic_personas[sub_topic].extend(personas)
                print(f'\tgenerating dialogs') 
                new_df = generate_dialogues(topic_personas=sub_topic_personas, system_prompt=dialog_system_prompt_cot, sample_dialogues=sample_dialogues)
                df = pd.concat([df, new_df], axis=0, ignore_index=True)
                print(f'\tdialogs generated for {base_topic}') 
        
            except Exception as e:
                print(f'\t ran into error: {e} for the topic: {base_topic}, moving on... ')
        df.to_csv(csv_path, index=False)
        print(f'all the generated dialogs have been saved to {csv_path}')
    
    else:
        df = pd.read_csv(csv_path)
    
    ## summarize
    if stage in ["scratch", "summary"]:
        print('generating summaries')
        try:
            df = generate_summary(df=df, sample_dialogues=sample_dialogues)
            print('summaries generated')
            df.to_csv(csv_path, index=False)
            print(f'all the generated summaries, have been saved to {csv_path}')
        except Exception as e:
            print(f'ran into error {e} while generating summaries, moving on...')

    ## llm-as-a-judge scores
    if stage in ["scratch", "summary", "llm_judge"]:
        if not disable_judge: 
            try:
                print(f'calculating judge scores')
                df = llm_judge(df=df)
                df.to_csv(csv_path, index=False)
                print(f'all the generated dialogs with judge scores calculated, have been saved to {csv_path}')
            except Exception as e:
                print(f'ran into error {e} when calculating the judge scores, moving on...')

    # generate metrics
    if stage in ["scratch", "summary", "metrics", "llm_judge"] and not disable_dialog_quality_metrics:
        print('calculating metrics for the generated dialog...')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
        metric_funcs = {
            'fed': partial(fed, model=model, tokenizer=tokenizer),
            'gpt_score': partial(gpt_score, model=model, tokenizer=tokenizer),
            'g_eval': partial(g_eval, model=model, tokenizer=tokenizer)
        }
        metric_scores = calculate_metrics(sentences=df['dialogue'].tolist(), metric_funcs=metric_funcs)
        for metric_name, scores in metric_scores.items(): 
            df[metric_name] = scores
            print(f'\tmetric {metric_name} calculated...')
        df.to_csv(csv_path, index=False)
        print(f'all the generated dialogs with metrics calculated, have been saved to {csv_path}')
    
            

if __name__ == "__main__":
    # the script can be run using the following command
    # python3 -m diasynth.main --topics_file_path "topics.txt" --dialogue_base "dialoguesum" --n_sub_topics 6 --n_personas 6 --stage "scratch" --save_path "generated_data/data.csv" --disable_judge --disable_dialog_quality_metrics --cooloff_period 180 --cooloff_frequency 6
    parser = ArgumentParser()
    parser.add_argument('--topics_file_path', default=None, type=str)
    parser.add_argument('--n_sub_topics', default=None, type=int)
    parser.add_argument('--n_personas', default=None, type=int)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--stage', type=str, default='scratch')
    parser.add_argument('--disable_judge', default=False, action="store_true")
    parser.add_argument('--disable_dialog_quality_metrics', default=False, action="store_true")
    parser.add_argument("--dialogue_base", type=str, default="dialoguesum")
    args = parser.parse_args()
    sdg(topics_file=args.topics_file_path, n_sub_topics=args.n_sub_topics, stage=args.stage, csv_path=args.csv_path, 
        n_personas=args.n_personas, disable_judge=args.disable_judge, disable_dialog_quality_metrics=args.disable_dialog_quality_metrics, 
        dialogue_base=args.dialogue_base)
    print(f'data generated and saved successfully to {args.csv_path}')
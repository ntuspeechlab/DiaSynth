import evaluate
from typing import List, Dict
from transformers import AutoTokenizer
import ast
import pandas as pd
import torch
import re
from bs4 import BeautifulSoup
from .prompts import generate_personas_prompt, generate_topics_prompt, judge_prompt, summ_prompt

## set up the model config and tokenizer
model_path = "models/llama3"
stop_tokens = ["<|eot_id|>"]
device = "cuda" if torch.cuda.is_available() else "cpu"
n_gpu_layers = 32 if device=="cuda" else 0
print(f'running on device: {device}, layers offloaded: {n_gpu_layers}')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="microsoft/Phi-3.5-mini-instruct",
                                            trust_remote_code=True)


def calculate_rouge_l(topic1: str, topic2: str) -> float:
    """calculates ROUGE-L between two strings"""
    rouge = evaluate.load('rouge')
    score = rouge.compute(references=[topic2], predictions=[topic1])
    return score['rougeL']


def filter_repeated_topics(topics: List[str], threshold: float=0.5) -> List[str]:
    """
    Filters repeated topics based on the similarity.
    Similarity is calculated using ROUGE-L
    """
    unique_topics = []
    for topic in topics:
        if all(calculate_rouge_l(topic, unique_topic) < threshold for unique_topic in unique_topics):
            unique_topics.append(topic)
    return unique_topics


def generate_sub_topics(model, topic: str, n_sub_topics: int) -> List[str]:
    """
    util function for generating niche sub topics for a given base topic
    helps in covering topics that are at tail of the curve
    """
    assert n_sub_topics<=20, "let's keep it down to 20 for now"
    prompt = tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": generate_topics_prompt.format(n_sub_topics=n_sub_topics)},
            {"role": "user", 'content': f"topic - {topic}"}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    output = model(
        prompt=prompt,
        temperature=0.1,
        stop=stop_tokens,
        max_tokens=2048,
        echo=False
    )
    topics = output['choices'][0]['text']
    start_idx, end_idx = topics.find('['), topics.find(']')
    topics = ast.literal_eval(topics[start_idx:end_idx+1])[:n_sub_topics]
    topics = filter_repeated_topics(topics)
    return topics
        

def generate_personas(model, topic: str, n_personas: int) -> List[str]:
    """
    util function for generating personas which are more likely to talk about a particular topic
    this way more realistic conversations are generated
    """
    assert n_personas<=20, "let's keep it down to 20 for now"
    prompt = tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": generate_personas_prompt.format(n_personas=n_personas)},
            {"role": "user", 'content': f"topic - {topic}"}
        ],
        tokenize=False,
        add_generation_prompt=True
    )
    output = model(
        prompt=prompt,
        temperature=0.1,
        stop=stop_tokens,
        max_tokens=2048,
        echo=False
    )
    personas = output['choices'][0]['text']
    start_idx, end_idx = personas.find('['), personas.find(']')
    personas = ast.literal_eval(personas[start_idx:end_idx+1])[:n_personas]
    personas = filter_repeated_topics(personas)
    return personas


def gen_dialogue_util(model, system_prompt: str, args: Dict[str, str], sample_dialogues: List[Dict[str, str]]) -> List[str]:
        """generate dialogue for a given domain"""
        # generate prompt
        system_prompt = system_prompt.format(dialogue_1=sample_dialogues[0]['dialogue'], 
                                             dialogue_2=sample_dialogues[1]['dialogue'],
                                             dialogue_3=sample_dialogues[2]['dialogue'],
                                             dialogue_4=sample_dialogues[3]['dialogue'],
                                             persona_1=args['persona_1'], persona_2=args['persona_2'])
        
        prompt = tokenizer.apply_chat_template(
                conversation=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f'"domain" - {args["topic"]}'}
                ],
                tokenize=False,
                add_generation_prompt=True
        )
        # gen response
        output = model(
                prompt=prompt,
                max_tokens=4096,
                temperature=0.2,
                stop=stop_tokens,
                echo=False
        )
        cot, dialogue = '', ''
        soup = BeautifulSoup(output['choices'][0]['text'], 'html.parser')
        if soup.find('dialogue'): dialogue = soup.find('dialogue').text
        if soup.find('cot'): cot = soup.find('cot').text
        if len(dialogue)>0 and len(cot)>0: return cot, dialogue
        else:
            txt = output['choices'][0]['text']
            if '<dialogue>' in txt and '</dialogue>' in txt:
                    start_idx = txt.find('<dialogue>') + len('<dialogue>')
                    end_idx = txt.find('</dialogue>')
                    dialogue = txt[start_idx:end_idx]
            else: dialogue = txt
            if '<cot>' in txt and '</cot>' in txt:
                    start_idx = txt.find('<cot>') + len('<cot>')
                    end_idx = txt.find('</cot>')
                    cot = txt[start_idx:end_idx]
            else: cot = txt
            return cot, dialogue
        

def llm_judge(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe of dialogues the rating of the dialogue is calculated
    'model' is prompted to give a score b/w 1 to 10
    """
    llm_judge_scores = []
    for i in range(df.shape[0]):    
        prompt = tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": f"Topic: {df['dialogue'][i]}\nPersona 1: {df['persona_1'][i]}\nPersona 2: {df['persona_2'][i]}\nDialog: {df['dialogue'][i]}"}
            ],
            tokenize=False, add_generation_prompt=True
        )
        op = model(
            prompt=prompt,
            temperature=0.4,
            max_tokens=1024,
            echo=False,
            stop=["<|user|>"]
        )
        pattern = re.compile(r'Score:\s*\d+')
        score = pattern.findall(op['choices'][0]['text'])[0]
        
        try: llm_judge_scores.append(int(score[-2:].strip()))
        except:llm_judge_scores.append(op['choices'][0]['text'])
        
    df['llm_judge_score'] = llm_judge_scores
    return df
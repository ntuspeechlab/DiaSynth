import evaluate
from typing import List, Dict
import ast
import pandas as pd
import re
from bs4 import BeautifulSoup
from openai import OpenAI
from .prompts import generate_personas_prompt, generate_topics_prompt, judge_prompt, summ_prompt

client = OpenAI()

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


def generate_sub_topics(topic: str, n_sub_topics: int) -> List[str]:
    """
    util function for generating niche sub topics for a given base topic
    helps in covering topics that are at tail of the curve
    """
    assert n_sub_topics<=20, "let's keep it down to 20 for now"
    completion = client.chat.completions.create(
         model="gpt-4o",
         messages=[
              {"role": "system", "content": generate_topics_prompt.format(n_sub_topics=n_sub_topics)},
              {"role": "user", "content": f"topic - {topic}"}
         ],
         temperature=0.1, max_tokens=2048
    )
    topics = completion.choices[0].message
    start_idx, end_idx = topics.find('['), topics.find(']')
    topics = ast.literal_eval(topics[start_idx:end_idx+1])[:n_sub_topics]
    topics = filter_repeated_topics(topics)
    return topics
        

def generate_personas(topic: str, n_personas: int) -> List[str]:
    """
    util function for generating personas which are more likely to talk about a particular topic
    this way more realistic conversations are generated
    """
    assert n_personas<=20, "let's keep it down to 20 for now"
    completion = client.chat.completions.create(
         model="gpt-4o",
         messages=[
              {"role": "system", "content": generate_personas_prompt.format(n_personas=n_personas)},
              {"role": "user", "content": f"topic - {topic}"}
         ],
         max_tokens=2048,
         temperature=0.1
    )
    personas = completion.choices[0].message
    start_idx, end_idx = personas.find('['), personas.find(']')
    personas = ast.literal_eval(personas[start_idx:end_idx+1])[:n_personas]
    personas = filter_repeated_topics(personas)
    return personas


def gen_dialogue_util(system_prompt: str, args: Dict[str, str], sample_dialogues: List[Dict[str, str]]) -> List[str]:
        """generate dialogue for a given domain"""
        # generate prompt
        system_prompt = system_prompt.format(dialogue_1=sample_dialogues[0]['dialogue'], 
                                             dialogue_2=sample_dialogues[1]['dialogue'],
                                             dialogue_3=sample_dialogues[2]['dialogue'],
                                             dialogue_4=sample_dialogues[3]['dialogue'],
                                             persona_1=args['persona_1'], persona_2=args['persona_2'])
        
        completion = client.chat.completions.create(
             model="gpt-4o",
             messages=[
                  {"role": "system", "content": system_prompt},
                  {"role": "user", "content": f'"domain" - {args["topic"]}'}
             ],
             max_tokens=4096,
             temperature=0.2
        )
        
        cot, dialogue = '', ''
        soup = BeautifulSoup(completion.choices[0].message, 'html.parser')
        if soup.find('dialogue'): dialogue = soup.find('dialogue').text
        if soup.find('cot'): cot = soup.find('cot').text
        if len(dialogue)>0 and len(cot)>0: return cot, dialogue
        else:
            txt = completion.choices[0].message
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
        

def llm_judge(df: pd.DataFrame) -> pd.DataFrame:
    """
    given a dataframe of dialogues the rating of the dialogue is calculated
    'model' is prompted to give a score b/w 1 to 10
    """
    llm_judge_scores = []
    for i in range(df.shape[0]):   
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": f"Topic: {df['dialogue'][i]}\nPersona 1: {df['persona_1'][i]}\nPersona 2: {df['persona_2'][i]}\nDialog: {df['dialogue'][i]}"}
            ],
            temperature=0.4,
            max_tokens=1024
        ) 
        pattern = re.compile(r'Score:\s*\d+')
        score = pattern.findall(completion.choices[0].message)[0]
        
        try: llm_judge_scores.append(int(score[-2:].strip()))
        except:llm_judge_scores.append(completion.choices[0].message)
        
    df['llm_judge_score'] = llm_judge_scores
    return df
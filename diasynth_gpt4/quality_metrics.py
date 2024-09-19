import re
import os
import pandas as pd
import numpy as np
from transformers import Phi3ForCausalLM, AutoTokenizer, PreTrainedModel
import torch
from collections import defaultdict
from typing import Dict, Any, List, Callable
from functools import partial
from .prompts import engagingness_system_prompt, naturalness_system_prompt, coherence_system_prompt, groundedness_system_prompt

device = "cuda" if torch.cuda.is_available() else "cpu"

def add_followup_utterance(conversation: str, followup: str) -> str:
    """
    util function for add a half turn to a conversation
    used in the calculation of the fed metric
    """
    # regex for finding the speakers
    turn_pattern = re.compile(r'(#.*?#):')
    turns = turn_pattern.findall(conversation) # get all speakers
    if turns:
        unique_speakers = list(dict.fromkeys(turns))
        if len(unique_speakers) >= 2: 
            last_speaker = turns[-1]
            followup_speaker = unique_speakers[0] if last_speaker == unique_speakers[1] else unique_speakers[1]
            new_conversation = conversation + f'\n    {followup_speaker}: {followup}'
        
        else: new_conversation = conversation + "\n " + followup
    else: new_conversation = conversation + "\n " + followup
    return new_conversation


def fed(conversations: str, model: PreTrainedModel, tokenizer: AutoTokenizer) -> List[Dict[str, float]]:
    """
    calculates the dialog level fed metric 
    adapted from 'https://github.com/Shikib/fed/blob/master/fed.py'
    """
    model.eval()
    scores = defaultdict(list)
    def fed_score(conversation: str, followup: str) -> float:
        """
        calculates the likelihood of a followup for a given conversation
        """
        updated_conv = add_followup_utterance(conversation, followup)
        tokenize_input = tokenizer.tokenize(updated_conv)
        tensor_input = torch.tensor([ tokenizer.convert_tokens_to_ids(tokenize_input)]).to(device) # 1, seq_len
        with torch.inference_mode():
            outputs = model(tensor_input, labels=tensor_input)
            loss = outputs.loss

        return loss.item()

    dialog_level_utts = {
        "coherent": {
            "positive": [],
            "negative": ["You're making no sense at all.", "You're changing the topic so much!", "You are so confusing."]
        },
        "error recovery": {
            "positive": [],
            "negative": ["I am so confused right now.", "You're really confusing.", "I don't understand what you're saying."]
        },
        "consistent": {
            "positive": [],
            "negative": ["That's not what you said earlier!", "Stop contradicting yourself!"],
        },
        "diverse": {
            "positive": [],
            "negative": ["Stop saying the same thing repeatedly.", "Why are you repeating yourself?", "Stop repeating yourself!"]
        },
        "depth": {
            "positive": [],
            "negative": ["Stop changing the topic so much.", "Don't change the topic!"],
        },
        "likeable": {
            "positive": ["I like you!", "You're super polite and fun to talk to", "Great talking to you."],
            "negative": ["You're not very nice.", "You're not very fun to talk to.", "I don't like you."]
        },
        "understand": {
            "positive": [],
            "negative": ["You're not understanding me!", "What are you trying to say?", "I don't understand what you're saying."]
        },
        "flexible": {
            "positive": ["You're very easy to talk to!", "Wow you can talk about a lot of things!"],
            "negative": ["I don't want to talk about that!", "Do you know how to talk about something else?"],
        },
        "informative": {
            "positive": ["Thanks for all the information!", "Wow that's a lot of information.", "You know a lot of facts!"],
            "negative": ["You're really boring.", "You don't really know much."],
        },
        "inquisitive": {
            "positive": ["You ask a lot of questions!", "That's a lot of questions!"],
            "negative": ["You don't ask many questions.", "You don't seem interested."],
        },
    }
    scores = []
    for conversation in conversations:
        metric_dict = {}
        for metric,utts in dialog_level_utts.items():
            pos = utts["positive"]
            neg = utts["negative"]
            # positive
            high_score = 0
            for m in pos:
                hs = fed_score(conversation, m) 
                high_score += hs 
            high_score = high_score/max(len(pos), 1)
            # negative
            low_score = 0
            for m in neg:
                ls = fed_score(conversation, m) 
                low_score += ls 
            low_score = low_score/max(len(neg), 1)
            metric_dict[metric] = low_score - high_score
        scores.append(metric_dict)
    return scores


def gpt_score(conversations: List[str], model: PreTrainedModel, tokenizer: AutoTokenizer) -> List[Dict[str, float]]:
    """
    calculates the gpt score for the conversations
    code adapted from https://github.com/jinlanfu/GPTScore/blob/main/gpt_inference.py
    """
    def gpt_score_util(coversation: str, eval_str: str) -> float:
        """
        puts together the string and calculates the probability of predicting 'yes'
        """
        model.eval()
        eval_str = eval_str.format(conversation=coversation)
        prompt = tokenizer.apply_chat_template(
            conversation=[{"role": "system", "content": eval_str}, {"role": "assistant", "content": "Answer: Yes"}],
            tokenize=False
        )
        tokens = tokenizer.encode(prompt)
        yes_id = tokenizer.encode('Yes')[0]
        with torch.inference_mode(): logits = model(torch.tensor([tokens], device=device)).logits
        # get logits of the token before yes
        # so that we can get the probability of predicting yes in the successive token
        before_yes_idx = len(tokens) - 1 - tokens[::-1].index(yes_id) - 1 # there might be multiple 'Yes', so it's better to reverse and get the last idx
        logits_before_yes = logits[:, before_yes_idx, :]
        yes_prob = torch.softmax(logits_before_yes, dim=-1)[0, yes_id]
        return yes_prob.item()

    metrics = {
        "coherence": "Answer the question based on the conversation generated by an AI.\nQuestion: Is the conversation coherent and maintains a good conversation flow throughout the conversation? (a) Yes. (b) No.\nConversation: {conversation}",
        "diversity": "Answer the question based on the conversation generated by an AI.\nQuestion: Is there diversity in the conversation? (a) Yes. (b) No.\nConversation: {conversation}",
        "flexibility": "Answer the question based on the conversation generated by an AI.\nQuestion: Is the conversation flexible and adaptable? (a) Yes. (b) No. \nConversation: {conversation}",
        "understandability": "Answer the question based on the conversation generated by an AI.\nQuestion: Does the conversation seem to be understandable? (a) Yes. (b) No. \nConversation: {conversation}",
        "inquisitiveness": "Answer the question based on the conversation generated by an AI.\nQuestion: Does conversation generate inquisitiveness? (a) Yes. (b) No.\nConversation: {conversation}",
        "consistency": "Answer the question based on the conversation generated by an AI.\nQuestion: Are the responses of the speakers consistent in the information they provide throughout the conversation? (a) Yes. (b) No.\nConversation: {conversation}",
        "informativeness": "Answer the question based on the conversation generated by an AI.\nQuestion: Are the responses of the speakers informative throughout the conversation? (a) Yes. (b) No.\nConversation: {conversation}",
        "likeability": "Answer the question based on the conversation generated by an AI.\nQuestion: Do both of the speakers display a likeable personality? (a) Yes. (b) No.\nConversation: {conversation}",
        "depth": "Answer the question based on the conversation generated by an AI.\nQuestion: Do both of the speakers discuss topics in depth? (a) Yes. (b) No.\nConversation: {conversation}",
        "error recovery": "Answer the question based on the conversation generated by an AI.\nQuestion: Are the speakers able to recover from errors that it makes? (a) Yes. (b) No.\nConversation: {conversation}"
    }
    scores = []
    for conversation in conversations:
        metric_dict = {}
        for metric in metrics: metric_dict[metric] = gpt_score_util(conversation, metrics[metric])
        scores.append(metric_dict)
    return scores


def g_eval(conversations: List[str], model: PreTrainedModel, tokenizer: AutoTokenizer) -> List[Dict[str, float]]:
    """
    calculates the g_eval for 4 metrics - engagingness, naturalness, coherence, groundedness
    """

    def g_eval_util(conversation, metric_name):
        score_ids = np.array([tokenizer.encode(str(i))[-1] for i in range(1, 4)])
        prompt = tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": metrics_prompts[metric_name]}, 
                {"role": "user", "content": conversation}
            ],
            tokenize=False,
            add_generation_prompt = True
        )
        tokens = torch.tensor([tokenizer.encode(prompt)], device=device)
        with torch.inference_mode(): logits = model(input_ids=tokens).logits
        
        logits = logits.cpu().numpy()[0, -1, :]
        score_probs = torch.softmax(torch.tensor(logits[score_ids]), dim=-1).numpy()
        rating = (score_probs*np.arange(start=1, stop=4)).sum(0)
        return rating

    metrics_prompts = {
        "engagingness": engagingness_system_prompt, "naturalness": naturalness_system_prompt,
        "coherence": coherence_system_prompt, "groundedness": groundedness_system_prompt
    }
    scores = []
    for conversation in conversations:
        metric_dict = {}
        for metric in metrics_prompts: metric_dict[metric] = g_eval_util(conversation, metric)
        scores.append(metric_dict) 
    return scores


def calculate_metrics(sentences: List[str], metric_funcs: Dict[str, Callable[..., Any]]) -> Dict[Any, Any]:
    """
    calculates a set of metrics which evaluate the dialog quality.
    added only fed and holistic eval so far
    """
    metric_scores = defaultdict(dict)
    for metric_name, metric_fn in metric_funcs.items():
        try:
            metric_scores[metric_name] = metric_fn(sentences)
        except Exception as e:
            print(f'ran into error {e} while processing for metric {metric_name}, moving on...')
    return metric_scores


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    model = Phi3ForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct").to(device)
    
    # calculate metrics for quality of the data
    metric_funcs = {
        'fed': partial(fed, model=model, tokenizer=tokenizer),
        'gpt_score': partial(gpt_score, model=model, tokenizer=tokenizer),
        'g_eval': partial(g_eval, model=model, tokenizer=tokenizer)
    }
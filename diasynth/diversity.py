from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import torch

# Check if GPU is available and use it
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize the required models and move to GPU
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

def calculate_diversity(sentences):
    """
    Calculate the diversity of the given sentences using ROUGE-L and cosine similarity.
    For a given sentence, calculate the maximum score it has with another sentence.
    Store the max score and take the average.
    Lesser the score, more diverse the sentences.
    """
    # Pre-encode all sentences and move embeddings to GPU
    embeddings = embedding_model.encode(sentences, convert_to_tensor=True, device=device)
    
    rouge_scores, cos_scores = [], []
    cache = {}  # Cache to avoid redundant calculations

    for i in range(len(sentences)):
        max_score_r, max_score_c = float("-inf"), float("-inf")
        for j in range(len(sentences)):
            if i != j:
                # Check if scores are cached
                if (i, j) in cache:
                    max_score_r = max(max_score_r, cache[(i, j)]['rouge'])
                    max_score_c = max(max_score_c, cache[(i, j)]['cos'])
                elif (j, i) in cache:
                    max_score_r = max(max_score_r, cache[(j, i)]['rouge'])
                    max_score_c = max(max_score_c, cache[(j, i)]['cos'])
                else:
                    # Calculate scores if not cached
                    r_score = scorer.score(sentences[i], sentences[j])['rougeL'].fmeasure
                    # Cosine similarity on GPU
                    c_score = util.cos_sim(embeddings[i], embeddings[j]).item()
                    max_score_r = max(max_score_r, r_score)
                    max_score_c = max(max_score_c, c_score)
                    # Cache the scores
                    cache[(i, j)] = {'rouge': r_score, 'cos': c_score}

        rouge_scores.append(max_score_r)
        cos_scores.append(max_score_c)

    # Calculate average of maximum scores
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
    avg_cosine = sum(cos_scores) / len(cos_scores) if cos_scores else 0
    
    return avg_rouge, avg_cosine

if __name__ == "__main__":
    files = os.listdir("generated_data")
    for file in files:
        path = f"generated_data/{file}"
        try:
            df = pd.read_csv(path)
            sentences = df['dialogue'].dropna().tolist()[:200]  # Handle NaN values
            if sentences:
                avg_rouge, avg_cosine = calculate_diversity(sentences)
                with open("logs_diversity.txt", "a") as f:
                    f.write(f"file: {file} rouge: {avg_rouge} cosine: {avg_cosine}\n")
            else:
                print(f"No valid sentences in file: {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
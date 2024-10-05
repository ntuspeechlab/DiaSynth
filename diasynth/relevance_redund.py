import torch
from transformers import AutoTokenizer, AutoModel
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)

def encode_sentences(sentences):
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
    inputs = {k:v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Sentence-level representation
    return embeddings

def centrality_scores(embeddings):
    cosine_sim = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    centrality = cosine_sim.mean(dim=1)  
    return centrality

def select_top_central_sentences(sentences, centrality, top_n=3):
    top_indices = centrality.topk(top_n).indices
    return [sentences[i] for i in top_indices]

def centrality_weighted_relevance(document, summary):
    doc_sentences = document.split('\n')

    doc_embeddings = encode_sentences(doc_sentences)    
    centrality = centrality_scores(doc_embeddings)
    top_sentences = select_top_central_sentences(doc_sentences, centrality)
    pseudo_reference = ' '.join(top_sentences)
    
    _, _, F1 = bert_score([summary], [pseudo_reference], lang="en", rescale_with_baseline=True)
    return F1.mean().item()

def self_referenced_redundancy(summary):
    summary_sentences = summary.split('. ')
    embeddings = encode_sentences(summary_sentences)    
    embeddings_np = embeddings.cpu().numpy()
    
    similarity_matrix = cosine_similarity(embeddings_np)    
    for i in range(len(summary_sentences)):
        similarity_matrix[i, i] = 0    
    redundancy = similarity_matrix.max(axis=-1).mean(axis=0)
    return redundancy

def final_evaluation_score(documents, summaries, lambda_=0.6):
    scores = []
    for (d, s) in zip(documents, summaries):
      relevance = centrality_weighted_relevance(d, s)
      redundancy = self_referenced_redundancy(s)
      final_score = (relevance - lambda_ * redundancy) / (1 + lambda_)
      scores.append(final_score)
    return sum(scores)/len(scores)
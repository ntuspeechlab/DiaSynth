import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class LSScoreModel(nn.Module):
    def __init__(self, model_id, vocab_size=30522, hidden_size=768, alpha=0.01, beta=1):
        super(LSScoreModel, self).__init__()
        # Load pre-trained BERT model (frozen)
        self.bert = BertModel.from_pretrained(model_id)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT layers

        # Define trainable layers W0 and W1
        self.W0 = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.W1 = nn.Parameter(torch.randn(hidden_size, vocab_size))
        self.gelu = nn.GELU()

        self.alpha = alpha
        self.beta = beta

    def l_score(self, summary_embeddings):
        
        Hx = summary_embeddings
        Wx0_Hx = torch.matmul(Hx, self.W0)  # Wx0^T * Hx
        Wx1_Wx0_Hx = torch.matmul(self.gelu(Wx0_Hx), self.W1)  # Wx1^T * GELU(Wx0^T * Hx)
        Px = torch.softmax(Wx1_Wx0_Hx, dim=-1)  # Softmax over vocab size
        
        # Max log probabilities
        token_probs = Px.max(dim=-1).values
        log_probs = torch.log(token_probs)
        return log_probs.mean(dim=-1)  # Mean over the sequence

    def s_score(self, dialogue_cls, summary_cls):
        return F.cosine_similarity(dialogue_cls, summary_cls, dim=-1)  # Cosine similarity between [CLS] tokens

    def forward(self, dialogue, summary, negative_samples):
        
        # BERT embeddings
        dialogue_outputs = self.bert(dialogue).last_hidden_state
        summary_outputs = self.bert(summary).last_hidden_state

        # CLS embeddings (for semantic similarity)
        dialogue_cls = dialogue_outputs[:, 0, :]
        summary_cls = summary_outputs[:, 0, :]

        # Calculate scores
        l_score_value = self.l_score(summary_outputs)  # Linguistic score
        s_score_value = self.s_score(dialogue_cls, summary_cls)  # Semantic score

        ls_positive = self.alpha * l_score_value + self.beta * s_score_value  # Combined LS_score for positive pair

        # Calculate loss
        loss = 0
        for neg_sample in negative_samples:
            neg_outputs = self.bert(neg_sample).last_hidden_state
            neg_cls = neg_outputs[:, 0, :]
            l_score_neg = self.l_score(neg_outputs)
            s_score_neg = self.s_score(dialogue_cls, neg_cls)
            ls_negative = self.alpha * l_score_neg + self.beta * s_score_neg

            # Contrastive loss
            loss += torch.relu(1 - (ls_positive - ls_negative))

        return ls_positive, loss.mean()  # Return score and average loss
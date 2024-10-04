import random
from torch.utils.data import Dataset

class LSScoreDataset(Dataset):
    def __init__(self, dialogues, summaries, tokenizer, max_length=512):
        self.dialogues = dialogues
        self.summaries = summaries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dialogues)

    def _delete_words(self, summary):
        words = summary.split()
        if len(words) > 1:
            num_deletions = random.randint(1, len(words) // 3)
            for _ in range(num_deletions):
                del words[random.randint(0, len(words) - 1)]
        return ' '.join(words)

    def _add_redundant_sentences(self, dialogue, summary):
        dialogue_sentences = dialogue.split('.')
        redundant_sentence = random.choice(dialogue_sentences).strip()
        return summary + ' ' + redundant_sentence

    def _shuffle_words(self, summary):
        sentences = summary.split('.')
        shuffled_summary = []
        for sentence in sentences:
            words = sentence.split()
            random.shuffle(words)
            shuffled_summary.append(' '.join(words))
        return '. '.join(shuffled_summary)

    def _generate_negative_samples(self, dialogue, summary):
        
        negative_samples = [
            self._delete_words(summary),
            self._add_redundant_sentences(dialogue, summary),
            self._shuffle_words(summary)
        ]
        return negative_samples

    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        summary = self.summaries[idx]
        negative_samples = self._generate_negative_samples(dialogue, summary)

        # Tokenize dialogue and summary
        dialogue_tokens = self.tokenizer(dialogue, truncation=True, max_length=self.max_length, padding='max_length', return_tensors="pt")
        summary_tokens = self.tokenizer(summary, truncation=True, max_length=self.max_length, padding='max_length', return_tensors="pt")
        
        # Tokenize negative samples
        negative_sample_tokens = [
            self.tokenizer(neg_sample, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt") 
            for neg_sample in negative_samples
        ]
        
        # Return the tokenized outputs
        return {
            "dialogue": dialogue_tokens['input_ids'].squeeze(0),  # Assuming batch size of 1
            "summary": summary_tokens['input_ids'].squeeze(0),
            "negative_samples": [neg_sample['input_ids'].squeeze(0) for neg_sample in negative_sample_tokens]
        }
import torch
import os
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from .model import LSScoreModel
from .data import LSScoreDataset

class Trainer:
    def __init__(self, args, model_id, dialogues, summaries, is_cuda=True):
        self.dialogues = dialogues
        self.summaries = summaries
        self.lr = args.lr
        self.model_id = model_id

        # Initialize the LSScoreModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.bert_model = LSScoreModel(model_id=model_id)

        # Move to the right device
        self.device = torch.device("cuda" if is_cuda else "cpu")
        self.bert_model.to(self.device)

        # Create model save directory
        self.model_save_dir = os.path.join(args.output_dir, f"model_save_{args.dataset}")
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # Initialize DataLoader and optimizer
        self.train_dataloader, self.test_dataloader = self.get_dataLoader(args, dialogues, summaries)
        self.init_optimizer(lr=args.lr)

    def init_optimizer(self, lr):
        optim_parameters = list(self.bert_model.parameters())
        self.optimizer = torch.optim.Adam(optim_parameters, lr=lr, weight_decay=1e-3)

    def get_dataLoader(self, args, dialogues, summaries):
        train_dataset = LSScoreDataset(dialogues=dialogues['train'], summaries=summaries['train'], tokenizer=self.tokenizer)
        test_dataset = LSScoreDataset(dialogues=dialogues['test'], summaries=summaries['test'], tokenizer=self.tokenizer)

        train_sampler = RandomSampler(train_dataset)
        test_sampler = RandomSampler(test_dataset)

        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1)

        return train_dataloader, test_dataloader

    def train(self, epoch):
        self.bert_model.train()
        self.run_epoch(epoch, self.train_dataloader, train=True)

    def test(self, epoch):
        self.bert_model.eval()
        with torch.no_grad():
            self.run_epoch(epoch, self.test_dataloader, train=False)

    def run_epoch(self, epoch, dataloader, train=True):
        mode = "train" if train else "test"
        data_iter = tqdm(enumerate(dataloader), desc=f"EP_{mode}:{epoch}", total=len(dataloader))

        total_loss = 0
        for i, batch in data_iter:
            dialogue = batch["dialogue"].to(self.device)
            summary = batch["summary"].to(self.device)
            negative_samples = [neg.to(self.device) for neg in batch["negative_samples"]]

            # Forward pass
            score, loss = self.bert_model(dialogue, summary, negative_samples)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} - {mode.capitalize()} Loss: {total_loss / len(dataloader)}")

    def save_model(self, epoch):
        save_path = os.path.join(self.model_save_dir, f"LS_Score.epoch.{epoch}")
        torch.save(self.bert_model.state_dict(), save_path)
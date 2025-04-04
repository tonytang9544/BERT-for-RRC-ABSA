import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


from data_utils import read_json_examples
import os
import datetime
import random
import numpy as np


class QADataset(Dataset):
    def __init__(self, questions, contexts, answers, answer_starts, tokenizer, max_length=512):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_starts = answer_starts

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        answer = self.answers[idx]

        # Tokenize the input pair (question, context)
        encoding = self.tokenizer.encode_plus(
            context,
            question,
            add_special_tokens=True,   # Add [CLS] and [SEP] tokens
            max_length=self.max_length,
            padding='max_length',      # Pad to max_length
            truncation=True,           # Truncate if too long
            return_tensors='pt'        # Return PyTorch tensors
        )

        # Find the start and end positions of the answer in the context
        start_position = self.answer_starts[idx]
        end_position = start_position + len(answer) - 1

        # align the positions
        start_token_idx = len(self.tokenizer.tokenize(context[:start_position]))
        end_token_idx = len(self.tokenizer.tokenize(context[:end_position])) -1

        # If no answer found, set to -1
        if start_token_idx is None or end_token_idx is None:
            start_token_idx = end_token_idx = -1

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'start_positions': torch.tensor(start_token_idx, dtype=torch.long),
            'end_positions': torch.tensor(end_token_idx, dtype=torch.long)
        }


def train(data_folder, run_folder, do_validation=True, model_path="bert-base-uncased", epochs=6, batch_size=32, learn_rate=5e-5, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    contexts, questions, answer_ids, answer_texts, start_positions = read_json_examples(os.path.join(data_folder, "train.json"))

    tokenizer = BertTokenizer.from_pretrained(model_path)

    dataset = QADataset(questions, contexts, answer_texts, start_positions, tokenizer)

    # mini_batch = int(batch_size/gradient_acc_steps)
    # assert mini_batch >= 1, f"wrong config of batch size = {batch_size} and gradient_acc_steps = {gradient_acc_steps}. Division less than 1!"
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = BertForQuestionAnswering.from_pretrained(model_path)

    # Set up the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learn_rate)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Set device (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(datetime.datetime.now())
    print(f"use {device}")
    print(f"Train model for {epochs} number of epochs")
    print(f"batch size = {batch_size}")
    print(f"Data folder = {data_folder}")


    # Training loop
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0
        for batch in dataloader:
            # Move batch to GPU if available
            batch = {k: v.to(device) for k, v in batch.items()}

            # Zero the gradients
            optimizer.zero_grad()            

            # Forward pass
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # Compute the loss
            start_loss = F.cross_entropy(start_logits, batch['start_positions'])
            end_loss = F.cross_entropy(end_logits, batch['end_positions'])
            loss = (start_loss + end_loss) / 2

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        # Print average loss for the epoch
        print(f"{datetime.datetime.now()} Epoch {epoch+1} Loss: {epoch_loss / len(dataloader)}")

    # Save the trained model
    model.save_pretrained(os.path.join(run_folder, str(seed), "model.pt"))

    if do_validation:
        model.eval()
        contexts, questions, answer_ids, answer_texts, start_positions = read_json_examples(os.path.join(data_folder, "dev.json"))
        dataset = QADataset(questions, contexts, answer_texts, start_positions, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size)

        total_eval_loss = 0
        for batch in dataloader:
            # Move batch to GPU if available
            batch = {k: v.to(device) for k, v in batch.items()}

            # Zero the gradients
            optimizer.zero_grad()            

            # Forward pass
            outputs = model(**batch)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            
            # Compute the loss
            start_loss = F.cross_entropy(start_logits, batch['start_positions'])
            end_loss = F.cross_entropy(end_logits, batch['end_positions'])
            loss = (start_loss + end_loss) / 2
            
            total_eval_loss += loss.item()
        print(f"{datetime.datetime.now()} validation loss = {total_eval_loss / len(dataloader)}")

def test(data_folder, run_folder):
    pass


if __name__ == "__main__":
    for i in range(3):
        train(data_folder="../data/rrc/laptop", run_folder="../run/pt_rrc/laptop", batch_size=8, seed=i)

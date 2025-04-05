import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import os, datetime, random, json
import numpy as np


def read_json_examples(input_file):
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]
    
    questions = []
    contexts = []
    question_ids = []
    answer_texts = []
    start_positions = []

    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            for qa in paragraph["qas"]:
                contexts.append(paragraph["context"])
                questions.append(qa["question"])
                question_ids.append(qa["id"])
                answer_texts.append(qa["answers"][0]["text"])
                start_positions.append(qa["answers"][0]["answer_start"])

    return contexts, questions, question_ids, answer_texts, start_positions



class QADataset(Dataset):
    def __init__(self, questions, contexts, answers, answer_starts, tokenizer, question_ids, max_length=512):
        self.questions = questions
        self.contexts = contexts
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.answer_starts = answer_starts
        self.question_ids = question_ids

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        context = self.contexts[idx]
        answer = self.answers[idx]
        question_id = self.question_ids[idx]

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
            'end_positions': torch.tensor(end_token_idx, dtype=torch.long),
            "question_ids": question_id
        }


def train(
        data_folder, 
        run_folder, 
        do_validation=True, 
        model_path="bert-base-uncased", 
        epochs=6, 
        batch_size=32, 
        gradient_accumulation_steps = 8,
        learn_rate=3e-5, 
        seed=0, 
        half_precision=False
        ):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    mini_batch = int(batch_size / gradient_accumulation_steps)
    assert mini_batch >= 1, f"Invalid batch_size or gradient_accumulation_steps."
    
    tokenizer = BertTokenizer.from_pretrained(model_path)

    contexts, questions, question_ids, answer_texts, start_positions = read_json_examples(os.path.join(data_folder, "train.json"))
    train_dataset = QADataset(questions, contexts, answer_texts, start_positions, tokenizer, question_ids)
    train_dataloader = DataLoader(train_dataset, batch_size=mini_batch, shuffle=True)

    if do_validation:
        contexts, questions, question_ids, answer_texts, start_positions = read_json_examples(os.path.join(data_folder, "dev.json"))
        val_dataset = QADataset(questions, contexts, answer_texts, start_positions, tokenizer, question_ids)
        val_dataloader = DataLoader(val_dataset, batch_size=mini_batch)

    # Initialize the model
    model = BertForQuestionAnswering.from_pretrained(model_path)

    # Set up the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=learn_rate, fused=half_precision)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Set device (GPU or CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if half_precision:
        model.half()

    print(datetime.datetime.now())
    print(f"use {device}")
    print(f"Train model for {epochs} number of epochs")
    print(f"batch size = {batch_size}")
    print(f"Data folder = {data_folder}")

    best_val_loss = float("inf")
    mini_batch_num = 0

    # Training loop
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0
        for batch in train_dataloader:
            
            batch = {k: v.to(device) for k, v in batch.items() if k != "question_ids"}

            # Zero the gradients
            # optimizer.zero_grad()            

            # Forward pass
            outputs = model(**batch)
            # start_logits = outputs.start_logits
            # end_logits = outputs.end_logits

            # # Compute the loss
            # start_loss = F.cross_entropy(start_logits, batch['start_positions'])
            # end_loss = F.cross_entropy(end_logits, batch['end_positions'])
            # loss = (start_loss + end_loss) / 2

            loss = outputs.loss / gradient_accumulation_steps

            loss.backward()
            mini_batch_num += 1
            

            # Backward pass and optimize
            if mini_batch_num >= gradient_accumulation_steps -1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                mini_batch_num = 0

            epoch_loss += loss.item()

        # Print average loss for the epoch
        print(f"{datetime.datetime.now()} Epoch {epoch+1} Loss: {epoch_loss / len(train_dataloader)}")

        if do_validation:
            model.eval()

            total_val_loss = 0
            for batch in val_dataloader:
                # Move batch to GPU if available
                batch = {k: v.to(device) for k, v in batch.items() if k != "question_ids"}

                # Forward pass
                with torch.no_grad():
                    outputs = model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps
                #     start_logits = outputs.start_logits
                #     end_logits = outputs.end_logits
                
                # # Compute the loss
                # start_loss = F.cross_entropy(start_logits, batch['start_positions'])
                # end_loss = F.cross_entropy(end_logits, batch['end_positions'])
                # loss = (start_loss + end_loss) / 2
                
                    total_val_loss += loss.item() / gradient_accumulation_steps
            
            ave_val_loss = total_val_loss / len(val_dataloader)
            print(f"{datetime.datetime.now()} validation loss = {ave_val_loss}")
            if ave_val_loss < best_val_loss:
                # Save the trained model
                model.save_pretrained(os.path.join(run_folder, str(seed)))
                tokenizer.save_pretrained(os.path.join(run_folder, str(seed)))
                best_val_loss = ave_val_loss
            


def test(        
        data_folder, 
        run_folder, 
        batch_size=32, 
        seed=0, 
        half_precision=False,
        gradient_accumulation_steps = 8
        ):
    
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed) 

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_folder = os.path.join(run_folder, str(seed))
    tokenizer = BertTokenizer.from_pretrained(model_folder)

    # construct test dataloader
    contexts, questions, question_ids, answer_texts, start_positions = read_json_examples(os.path.join(data_folder, "test.json"))
    test_dataset = QADataset(questions, contexts, answer_texts, start_positions, tokenizer, question_ids)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # load model
    model = BertForQuestionAnswering.from_pretrained(model_folder).to(device)

    if half_precision:
        model.half()

    model.eval()

    predictions = {} 

    total_test_loss = 0
    for test_batch in test_dataloader:
        # Move batch to GPU if available
        batch = {k: v.to(device) for k, v in test_batch.items() if k != "question_ids"}

        # Forward pass
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
        
        # # Compute the loss
        # start_loss = F.cross_entropy(start_logits, batch['start_positions'])
        # end_loss = F.cross_entropy(end_logits, batch['end_positions'])
        # loss = (start_loss + end_loss) / 2
        
            total_test_loss += loss.item()

            # compute the prediction using argmax
            pred_start = torch.argmax(start_logits, dim=-1)
            pred_end = torch.argmax(end_logits, dim=-1)

            for i in range(len(pred_start)):
                answer_test = tokenizer.decode(test_batch["input_ids"][i][pred_start[i]:pred_end[i]+1], skip_special_tokens=True)
                predictions[test_batch["question_ids"][i]] = answer_test
    
    ave_test_loss = total_test_loss / len(test_dataloader)
    print(f"{datetime.datetime.now()} test loss = {ave_test_loss}")

    with open(os.path.join(run_folder, str(seed), "predictions.json"), "w") as js_file:
            json.dump(predictions, js_file)


if __name__ == "__main__":
    for i in range(2):
        train(data_folder="../../../data/rrc/laptop", run_folder="./results", gradient_accumulation_steps=8, seed=i+1, half_precision=False)
        test(data_folder="../../../data/rrc/laptop", run_folder="./results", seed=i+1, half_precision=False)

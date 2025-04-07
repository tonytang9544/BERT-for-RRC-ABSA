import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForQuestionAnswering, BertModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

import os, datetime, random, json
import numpy as np

from sklearn.metrics import f1_score


class DualEncoder(torch.nn.Module):
    def __init__(self, model_name, model_hidden_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.q_linear = torch.nn.Linear(model_hidden_dim, model_hidden_dim)
        self.k_linear = torch.nn.Linear(model_hidden_dim, model_hidden_dim)

    def forward(self, question, context):
        q_e = F.normalize(self.q_linear(self.bert(**question).pooler_output), dim=-1)
        k_e = F.normalize(self.k_linear(self.bert(**context).last_hidden_state), dim=-1)

        return F.sigmoid(torch.einsum('bij,bj->bi', k_e, q_e))



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
        q_encode = self.tokenizer(
            question,
            add_special_tokens=True,   # Add [CLS] and [SEP] tokens
            max_length=self.max_length,
            padding='max_length',      # Pad to max_length
            truncation=True,           # Truncate if too long
            return_tensors='pt'        # Return PyTorch tensors
        )
        q_encode = {k:v.squeeze() for k, v in q_encode.items()}
        # print(q_encode)

        c_encode = self.tokenizer(
            context,
            add_special_tokens=True,   # Add [CLS] and [SEP] tokens
            max_length=self.max_length,
            padding='max_length',      # Pad to max_length
            truncation=True,           # Truncate if too long
            return_tensors='pt'        # Return PyTorch tensors
        )

        c_encode = {k:v.squeeze() for k, v in c_encode.items()}


        # Find the start and end positions of the answer in the context
        start_position = self.answer_starts[idx]
        end_position = start_position + len(answer) - 1

        answer_span = torch.zeros_like(q_encode["input_ids"])
        # print(answer_span.shape)


        # align the positions
        context_len = len(context)

        start_token_idx = end_token_idx = None


        if start_position <= context_len:
            start_token_idx = len(self.tokenizer.tokenize(context[:start_position]))

        if end_position <= context_len:
            end_token_idx = len(self.tokenizer.tokenize(context[:end_position])) -1

        if start_token_idx is not None:
            if end_token_idx is not None:
                answer_span[torch.arange(start_token_idx, end_token_idx+1)] = 1
            else:
                answer_span[torch.arange(start_token_idx, len(answer_span))] = 1
        

        return {
            "question_encoding": q_encode,
            "context_encoding": c_encode,
            "answer_span": answer_span,
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
    model = DualEncoder(model_path, 768)

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

    # Training loop
    for epoch in range(epochs):

        model.train()
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            
            q_e = {k: v.to(device) for k, v in batch["question_encoding"].items()}
            c_e = {k: v.to(device) for k, v in batch["context_encoding"].items()}

            # Forward pass
            outputs = model(q_e, c_e)

            loss = F.mse_loss(outputs, batch["answer_span"].to(device=device, dtype=torch.float))

            loss.backward()            

            # Backward pass and optimize
            if step % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() / gradient_accumulation_steps

        # Print average loss for the epoch
        print(f"{datetime.datetime.now()} Epoch {epoch+1} Loss: {epoch_loss / len(train_dataloader)}")

        if do_validation:
            model.eval()

            total_val_loss = 0
            for batch in val_dataloader:
                # Move batch to GPU if available
                q_e = {k: v.to(device) for k, v in batch["question_encoding"].items()}
                c_e = {k: v.to(device) for k, v in batch["context_encoding"].items()}

                # Forward pass
                with torch.no_grad():
                    outputs = model(q_e, c_e)
                    loss = F.mse_loss(outputs, batch["answer_span"].to(device=device, dtype=torch.float))
                
                    total_val_loss += loss.item() / gradient_accumulation_steps
            
            ave_val_loss = total_val_loss / len(val_dataloader)
            print(f"{datetime.datetime.now()} validation loss = {ave_val_loss}")
            if ave_val_loss < best_val_loss:
                # Save the trained model
                tokenizer.save_pretrained(os.path.join(run_folder, str(seed)))
                torch.save(model, os.path.join(run_folder, str(seed), "model.pt"))
                best_val_loss = ave_val_loss
            


def test(        
        data_folder, 
        run_folder, 
        batch_size=32, 
        seed=0, 
        half_precision=False,
        gradient_accumulation_steps = 8
        ):
    
    model_hidden_dim = 512

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
    model = torch.load(os.path.join(run_folder, str(seed), "model.pt"), weights_only=False).to(device)

    if half_precision:
        model.half()

    model.eval()

    predictions = {}
    f1_scores = {}

    total_test_loss = 0
    for batch in test_dataloader:
        # Move batch to GPU if available
            q_e = {k: v.to(device) for k, v in batch["question_encoding"].items()}
            c_e = {k: v.to(device) for k, v in batch["context_encoding"].items()}

            # Forward pass
            with torch.no_grad():
                outputs = model(q_e, c_e)
                loss = F.mse_loss(outputs, batch["answer_span"].to(device=device, dtype=torch.float))
        
            total_test_loss += loss.item()


            for i in range(len(batch["context_encoding"])):
                answer_tokens = batch["context_encoding"][i]
                predict_answer_tokens = [answer_tokens[j] for j in range(len(answer_tokens)) if outputs[i][j]>0.5]
                answer_text = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
                predictions[batch["question_ids"][i]] = answer_text
                true_span = torch.zeros_like(batch["input_ids"][i])
                if batch["start_positions"][i] <= model_hidden_dim:
                    if batch["end_positions"][i]+1 <= model_hidden_dim:

                        true_span[torch.arange(batch["start_positions"][i], batch["end_positions"][i]+1)] = 1
                    else:
                        true_span[torch.arange(batch["start_positions"][i], model_hidden_dim)] = 1
                f1_scores[batch["question_ids"][i]] = f1_score(true_span, torch.where(outputs[i]>0.5))
    
    ave_test_loss = total_test_loss / len(test_dataloader)
    print(f"{datetime.datetime.now()} test loss = {ave_test_loss}")

    with open(os.path.join(run_folder, str(seed), "predictions.json"), "w") as js_file:
        json.dump(predictions, js_file)
    with open(os.path.join(run_folder, str(seed), "f1s.json"), "w") as js_file:
        json.dump(f1_scores, js_file)


if __name__ == "__main__":
    for i in range(2):
        train(data_folder="../../../data/rrc/laptop", run_folder="./results", gradient_accumulation_steps=8, seed=i+1)
        test(data_folder="../../../data/rrc/laptop", run_folder="./results", seed=i+1)

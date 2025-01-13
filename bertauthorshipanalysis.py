import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from collections import Counter

# 1. Data Loading and Preprocessing
class AuthorshipDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        paragraph1, paragraph2 = self.texts[idx]  # unpack pair of consecutive paragraphs

        # encoding paragraph pair
        encoding = self.tokenizer.encode_plus(
            paragraph1,
            paragraph2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# Updated Load Data Function
def load_data(data_path):
    texts, labels = [], [] 
    txt_path = os.path.join(data_path, 'txts')
    json_path = os.path.join(data_path, 'jsons')

    for filename in os.listdir(txt_path):
        if filename.endswith('.txt'):
            with open(os.path.join(txt_path, filename), 'r', encoding='utf-8') as f:
                paragraphs = [p.strip() for p in f.read().split('\n') if p.strip()]

            json_filename = filename.replace('.txt', '.json')
            json_file_path = os.path.join(json_path,'truth-'+json_filename)

            if os.path.exists(json_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    ground_truth = json.load(f)["changes"]

                for i in range(len(paragraphs) - 1):
                    texts.append((paragraphs[i], paragraphs[i + 1]))
                    labels.append(1 if i in ground_truth else 0)
            else:
                print(f"Warning: Ground truth file missing for {filename}")

    
    print(len(texts), len(labels))
    return texts, labels



# 2. Model Setup
def train_model(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    print("Training...")
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 3. Evaluation
def evaluate_model(val_loader, model, device):
    model.eval()
    predictions, true_labels = [], []
    print("Evaluating...")
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return predictions, true_labels

def clear_gpu_memory():
    torch.cuda.empty_cache()   # Clears the PyTorch cache
    gc.collect()               # Python garbage collector
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()  # Cleans up inter-process communication memory
    print("GPU memory cleared.")

clear_gpu_memory()

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

difficulty_levels = ['easy', 'medium', 'hard']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
for level in difficulty_levels:
    print(f"\nRunning for {level} dataset...\n")
    
    train_data_path = f'../dataset/{level}/train'
    test_data_path = f'../dataset/{level}/validation'
    
    # Load data
    training_texts, training_labels = load_data(train_data_path)
    testing_texts, testing_labels = load_data(test_data_path)

    print("Training Labels Distribution:", Counter(training_labels))
    print("Testing Labels Distribution:", Counter(testing_labels))
    
    print("Unique labels in training data:", set(training_labels))
    print("Unique labels in testing data:", set(testing_labels))

    # Prepare dataset and DataLoader
    training_dataset = AuthorshipDataset(training_texts, training_labels, tokenizer)
    testing_dataset = AuthorshipDataset(testing_texts, testing_labels, tokenizer)

    train_loader = DataLoader(training_dataset, batch_size=4, shuffle=True)#cant do bigger batch size due to memory constraints
    val_loader = DataLoader(testing_dataset, batch_size=4) #cant do bigger batch size due to memory constraints

    # Load model and optimizer
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training
    for epoch in range(5):
        train_loss = train_model(train_loader, model, optimizer, device)
        print(f"{level.capitalize()} - Epoch {epoch + 1}, Loss: {train_loss}")

    # Evaluation
    y_pred, y_true = evaluate_model(val_loader, model, device)
    print(f"{level.capitalize()} Dataset - Classification Report:\n", classification_report(y_true, y_pred, zero_division=1))

    # Clear GPU memory
    clear_gpu_memory()

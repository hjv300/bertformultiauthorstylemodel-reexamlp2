import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from collections import Counter

class AuthorshipDataset:
    def __init__(self, data_path):
        self.txt_dir = os.path.join(data_path, 'txts')
        self.json_dir = os.path.join(data_path, 'jsons')
        self.data = self._load_data()
    
    def _load_data(self):
        data = []
        for txt_file in os.listdir(self.txt_dir):
            if txt_file.endswith(".txt"):
                txt_path = os.path.join(self.txt_dir, txt_file)
                json_filename = 'truth-' + txt_file.replace(".txt", ".json")
                json_path = os.path.join(self.json_dir, json_filename)
                
                with open(txt_path, 'r', encoding='utf-8') as f_txt:
                    paragraphs = [p.strip() for p in f_txt.read().split('\n') if p.strip()]
                
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f_json:
                        ground_truth = json.load(f_json)["changes"]
                    
                    for i in range(len(paragraphs) - 1):
                        pair = (paragraphs[i], paragraphs[i + 1])
                        label = 1 if i in ground_truth else 0
                        data.append((pair, label))
                else:
                    print(f"Warning: Ground truth file missing for {txt_file}")
        return data

    def get_features_and_labels(self):
        pairs, labels = zip(*self.data)
        combined_texts = [p1 + " " + p2 for p1, p2 in pairs]
        return combined_texts, list(labels)

data_path = "../dataset/easy/train"

dataset = AuthorshipDataset(data_path)
texts, labels = dataset.get_features_and_labels()

label_counts = Counter(labels)
print(f"Label distribution: {label_counts}")

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in splitter.split(texts, labels):
    X_train = [texts[i] for i in train_idx]
    X_val = [texts[i] for i in val_idx]
    y_train = [labels[i] for i in train_idx]
    y_val = [labels[i] for i in val_idx]

# Define Pipeline with TF-IDF and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])
#Default threshold, comment to use custom threshold
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_val)

print(classification_report(y_val, y_pred))

# Custom threshold, uncomment to use
"""y_proba = pipeline.predict_proba(X_val)[:, 1]
custom_threshold = 0.45
y_pred_custom = (y_proba >= custom_threshold).astype(int)
print(classification_report(y_val, y_pred_custom))"""
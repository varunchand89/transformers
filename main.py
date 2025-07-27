import pandas as pd
import numpy as np
from transformers import BertTokenizer
import torch
from transformers import BertForSequenceClassification
from sklearn.metrics import classification_report
from transformers import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import AdamW





# ===== 1. Load your Data =====
# Replace this with your actual data
# Example columns: 'text', 'reason', 'label' (label = 1 for positive)
df1 = pd.read_excel("C:/Users/Hp/Downloads/train.xlsx")

# ===== 2. Augment Data with Negative Class =====
def augment_negative(df1):
    df1_negative = df1.copy()
    df1_negative['reason'] = np.random.permutation(df1_negative['reason'].values)
    df1_negative['label'] = 0  # assign negative label
    # Optional: drop any accidentally valid match
    df1_negative = df1_negative[df1_negative['reason'] != df1['reason']].reset_index(drop=True)
    return df1_negative

df1_negative = augment_negative(df1)
df1_augmented = pd.concat([df1, df1_negative], ignore_index=True).sample(frac=1).reset_index(drop=True)

# ===== 3. Train-Test Split =====
train_df, test_df = train_test_split(df1_augmented, test_size=0.2, random_state=42)

# ===== 4. Tokenization =====
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize(texts, reasons):
    return tokenizer(
        texts,
        reasons,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

train_encodings = tokenize(train_df['text'].tolist(), train_df['reason'].tolist())
test_encodings = tokenize(test_df['text'].tolist(), test_df['reason'].tolist())

train_labels = train_df['label'].tolist()
test_labels = test_df['label'].tolist()

# ===== 5. PyTorch Dataset =====
class PairDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PairDataset(train_encodings, train_labels)
test_dataset = PairDataset(test_encodings, test_labels)

# ===== 6. Model Definition =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

# ===== 7. Training Loop =====
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optimizer = AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):  # or more epochs
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

print(f"Training complete. Final loss: {total_loss:.4f}")

# ===== 8. Evaluation =====
model.eval()
all_preds = []
all_labels = []

test_loader = DataLoader(test_dataset, batch_size=32)

with torch.no_grad():
    for batch in test_loader:
        labels = batch['labels'].to(device)
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, digits=4))

model.save_pretrained("C:/Users/Hp/OneDrive/Desktop/transformers_model/fine_tuned_model.pkl")
tokenizer.save_pretrained("C:/Users/Hp/OneDrive/Desktop/transformers_model/fine_tuned_model.pkl")
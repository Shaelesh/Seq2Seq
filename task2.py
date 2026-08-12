import json

with open("train.json", "r") as f:
    train_data = json.load(f)

with open("val.json", "r") as f:
    val_data = json.load(f)

with open("test.json", "r") as f:
    test_data = json.load(f)
    
print("="*60)

from collections import Counter

word_counter = Counter()

for sample in train_data:
    word_counter.update(sample["tokens"])

word2idx = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word in word_counter:
    word2idx[word] = len(word2idx)

idx2word = {
    v: k for k, v in word2idx.items()
}

tag_set = set()

for sample in train_data:
    tag_set.update(sample["tags"])

tag2idx = {
    "<PAD>": 0
}

for tag in sorted(tag_set):
    tag2idx[tag] = len(tag2idx)

idx2tag = {
    v: k for k, v in tag2idx.items()
}

def encode_tokens(tokens):

    return [
        word2idx.get(token, word2idx["<UNK>"])
        for token in tokens
    ]


def encode_tags(tags):

    return [
        tag2idx[tag]
        for tag in tags
    ]
    
from torch.utils.data import Dataset

class SlotDataset(Dataset):

    def __init__(self, data):

        self.data = data

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        tokens = encode_tokens(
            sample["tokens"]
        )

        tags = encode_tags(
            sample["tags"]
        )

        return tokens, tags
    
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):

    tokens = [
        torch.tensor(x[0])
        for x in batch
    ]

    tags = [
        torch.tensor(x[1])
        for x in batch
    ]

    tokens = pad_sequence(
        tokens,
        batch_first=True,
        padding_value=0
    )

    tags = pad_sequence(
        tags,
        batch_first=True,
        padding_value=0
    )

    return tokens, tags

from torch.utils.data import DataLoader

train_dataset = SlotDataset(train_data)
val_dataset = SlotDataset(val_data)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    collate_fn=collate_fn
)

import torch.nn as nn

class BiLSTMTagger(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        num_tags
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.bilstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            hidden_dim * 2,
            num_tags
        )

    def forward(self, x):

        x = self.embedding(x)

        output, _ = self.bilstm(x)

        logits = self.fc(output)

        return logits
    
model = BiLSTMTagger(
    vocab_size=len(word2idx),
    embedding_dim=100,
    hidden_dim=128,
    num_tags=len(tag2idx)
)

import torch
import torch.optim as optim

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

criterion = nn.CrossEntropyLoss(
    ignore_index=tag2idx["<PAD>"]
)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

def train_one_epoch(model, loader):

    model.train()

    total_loss = 0

    for tokens, tags in loader:

        tokens = tokens.to(device)
        tags = tags.to(device)

        optimizer.zero_grad()

        logits = model(tokens)

        logits = logits.view(
            -1,
            logits.size(-1)
        )

        tags = tags.view(-1)

        loss = criterion(
            logits,
            tags
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_loss(model, loader):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for tokens, tags in loader:

            tokens = tokens.to(device)
            tags = tags.to(device)

            logits = model(tokens)

            logits = logits.view(
                -1,
                logits.size(-1)
            )

            tags = tags.view(-1)

            loss = criterion(
                logits,
                tags
            )

            total_loss += loss.item()

    return total_loss / len(loader)

num_epochs = 10

for epoch in range(num_epochs):

    train_loss = train_one_epoch(
        model,
        train_loader
    )

    val_loss = evaluate_loss(
        model,
        val_loader
    )

    print(
        f"Epoch {epoch+1}/{num_epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )
    
    from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

test_dataset = SlotDataset(test_data)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

def get_test_predictions(model, loader):

    model.eval()

    all_preds = []
    all_true = []

    with torch.no_grad():

        for tokens, tags in loader:

            tokens = tokens.to(device)
            tags = tags.to(device)

            logits = model(tokens)

            preds = torch.argmax(
                logits,
                dim=-1
            )

            # Ignore PAD positions
            mask = tags != tag2idx["<PAD>"]

            preds = preds[mask]
            tags = tags[mask]

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_true.extend(
                tags.cpu().numpy()
            )

    return all_true, all_preds

y_true, y_pred = get_test_predictions(
    model,
    test_loader
)

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

print("=" * 60)
print("TASK 2 — BiLSTM SLOT FILLING TEST RESULTS")
print("=" * 60)

print(f"Test Samples : {len(test_data)}")
print(f"Token Accuracy : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")

labels = [
    idx for tag, idx in tag2idx.items()
    if tag != "<PAD>"
]

target_names = [
    idx2tag[idx]
    for idx in labels
]

print("\n" + "=" * 60)
print("PER-TAG PERFORMANCE")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    )
)


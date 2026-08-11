import json
import torch 
from sklearn.metrics import classification_report, confusion_matrix

print("="*60)

with open(r"C:\Users\SAJJAN\Desktop\Seq-seq\Seq2Seq\train.json", "r") as f:
    train_data = json.load(f)

from collections import Counter

intent_counts = Counter(
    sample["intent"]
    for sample in train_data
)

intent2id = {
    "CREATE_EVENT" : 0,
    "SET_REMINDER" : 1,
    "QUERY_FREE_TIME" : 2,
    "CANCEL" :3
}

id2intent = {
    0: "CREATE_EVENT",
    1: "SET_REMINDER",
    2: "QUERY_FREE_TIME",
    3: "CANCEL"
}

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

word2id = {
    PAD_TOKEN : 0 ,
    UNK_TOKEN : 1
}

for sample in train_data:
    for token in sample["tokens"]:
        token = token.lower()
        if token not in word2id:
            word2id[token] = len(word2id)
            
def encode_tokens(tokens, word2id = word2id):
    ids = []

    for token in tokens:
        token = token.lower()

        if token in word2id:
            ids.append(word2id[token])
        else:
            ids.append(word2id[UNK_TOKEN])

    return ids

from torch.utils.data import Dataset

class IntentDataset(Dataset):

    def __init__(self, data, word2id, intent2id):
        self.data = data
        self.word2id = word2id
        self.intent2id = intent2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        tokens = sample["tokens"]

        input_ids = encode_tokens(
            tokens,
            self.word2id
        )

        label = self.intent2id[
            sample["intent"]
        ]

        return input_ids, label
    

dataset = IntentDataset(
    train_data,
    word2id,
    intent2id
)

from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):

    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    lengths = torch.tensor(
        [len(x) for x in input_ids],
        dtype=torch.long
    )

    input_ids = [
        torch.tensor(x, dtype=torch.long)
        for x in input_ids
    ]

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long
    )

    return input_ids, lengths, labels

from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

batch = next(iter(train_loader))

input_ids, lengths, labels = batch

import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class IntentClassifier(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_classes
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, input_ids, lengths):

        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, (hidden, cell) = self.lstm(packed)

        hidden = hidden[-1]

        logits = self.fc(hidden)

        return logits

model = IntentClassifier(
    vocab_size=len(word2id),
    embedding_dim=128,
    hidden_size=128,
    num_classes=len(intent2id)
)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


with open(r"C:\Users\SAJJAN\Desktop\Seq-seq\Seq2Seq\val.json", "r", encoding="utf-8") as f:
    val_data = json.load(f)
    
val_dataset = IntentDataset(
    val_data,
    word2id,
    intent2id
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

with open(
    r"C:\Users\SAJJAN\Desktop\Seq-seq\Seq2Seq\test.json",
    "r",
    encoding="utf-8"
) as f:
    test_data = json.load(f)

test_dataset = IntentDataset(
    test_data,
    word2id,
    intent2id
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=collate_fn
)

def evaluate(model, data_loader, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for input_ids, lengths, labels in data_loader:

            logits = model(input_ids, lengths)

            loss = criterion(logits, labels)

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / len(data_loader)

    accuracy = correct / total

    return avg_loss, accuracy

def get_predictions(model, data_loader):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for input_ids, lengths, labels in data_loader:

            logits = model(
                input_ids,
                lengths
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    return all_labels, all_predictions

num_epochs = 10

for epoch in range(num_epochs):

    model.train()

    total_loss = 0

    for input_ids, lengths, labels in train_loader:

        optimizer.zero_grad()

        logits = model(input_ids, lengths)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion
    )

    print(
        f"Epoch {epoch + 1}/{num_epochs} "
        f"| Train Loss: {train_loss:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.4f}"
    )
    
    
test_loss, test_accuracy = evaluate(
    model,
    test_loader,
    criterion
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

test_labels, test_predictions = get_predictions(
    model,
    test_loader
)

print("\nTest Classification Report:")

print(
    classification_report(
        test_labels,
        test_predictions,
        labels=[0, 1, 2, 3],
        target_names=[
            "CREATE_EVENT",
            "SET_REMINDER",
            "QUERY_FREE_TIME",
            "CANCEL"
        ],
        digits=4,
        zero_division=0
    )
)

cm = confusion_matrix(
    test_labels,
    test_predictions,
    labels=[0, 1, 2, 3]
)

print("\nConfusion Matrix:")
print(cm)


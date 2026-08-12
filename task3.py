import json
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import random






with open("train.json", "r") as f:
    train_data = json.load(f)

with open("val.json", "r") as f:
    val_data = json.load(f)

with open("test.json", "r") as f:
    test_data = json.load(f)

def make_target(example):
    date = example["date_iso"]
    time = example["time_hm"]

    if date is not None and time is not None:
        return f"{date} {time}"

    elif date is not None:
        return date

    elif time is not None:
        return time

    else:
        return "NA"

from collections import Counter

word_counter = Counter()

for example in train_data:
    word_counter.update(example["tokens"])
    
input_vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word in word_counter:
    input_vocab[word] = len(input_vocab)
    
idx_to_word = {
    idx: word
    for word, idx in input_vocab.items()
}

char_set = set()

for example in train_data:
    target = make_target(example)

    for ch in target:
        char_set.add(ch)

output_vocab = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2
}

for ch in sorted(char_set):
    output_vocab[ch] = len(output_vocab)

idx_to_char = {
    idx: ch
    for ch, idx in output_vocab.items()
}

def encode_input(tokens):
    return [
        input_vocab.get(word, input_vocab["<UNK>"])
        for word in tokens
    ]

def encode_target(target):
    encoded = [output_vocab["<SOS>"]]

    for ch in target:
        encoded.append(output_vocab[ch])

    encoded.append(output_vocab["<EOS>"])

    return encoded

def decode_target(indices):

    chars = []

    for idx in indices:

        if idx == output_vocab["<EOS>"]:
            break

        if idx in [
            output_vocab["<PAD>"],
            output_vocab["<SOS>"]
        ]:
            continue

        chars.append(idx_to_char[idx])

    return "".join(chars)

class CalendarDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        example = self.data[idx]

        input_ids = encode_input(example["tokens"])

        target = make_target(example)
        target_ids = encode_target(target)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long)
        )
        
train_dataset = CalendarDataset(train_data)
val_dataset = CalendarDataset(val_data)
test_dataset = CalendarDataset(test_data)


def collate_fn(batch):

    input_ids, target_ids = zip(*batch)

    input_lengths = torch.tensor(
        [len(x) for x in input_ids],
        dtype=torch.long
    )

    target_lengths = torch.tensor(
        [len(y) for y in target_ids],
        dtype=torch.long
    )

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=input_vocab["<PAD>"]
    )

    target_ids = pad_sequence(
        target_ids,
        batch_first=True,
        padding_value=output_vocab["<PAD>"]
    )

    return (
        input_ids,
        input_lengths,
        target_ids,
        target_lengths
    )

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)


class Encoder(nn.Module):

    def __init__(
        self,
        input_vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        dropout=0.0
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            input_vocab_size,
            embedding_dim,
            padding_idx=input_vocab["<PAD>"]
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, input_ids, input_lengths):

        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed)

        return hidden, cell
    
INPUT_VOCAB_SIZE = len(input_vocab)

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 1

encoder = Encoder(
    input_vocab_size=INPUT_VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
)

class Decoder(nn.Module):

    def __init__(
        self,
        output_vocab_size,
        embedding_dim,
        hidden_dim,
        num_layers=1
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            output_vocab_size,
            embedding_dim,
            padding_idx=output_vocab["<PAD>"]
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc_out = nn.Linear(
            hidden_dim,
            output_vocab_size
        )

    def forward(self, input_token, hidden, cell):

        input_token = input_token.unsqueeze(1)

        embedded = self.embedding(input_token)

        output, (hidden, cell) = self.lstm(
            embedded,
            (hidden, cell)
        )

        prediction = self.fc_out(
            output.squeeze(1)
        )

        return prediction, hidden, cell

OUTPUT_VOCAB_SIZE = len(output_vocab)

DECODER_EMBEDDING_DIM = 64

decoder = Decoder(
    output_vocab_size=OUTPUT_VOCAB_SIZE,
    embedding_dim=DECODER_EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
)

class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        device
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        input_ids,
        input_lengths,
        target_ids,
        teacher_forcing_ratio=0.5
    ):

        batch_size = input_ids.size(0)
        target_length = target_ids.size(1)

        output_vocab_size = self.decoder.fc_out.out_features

        # Store predictions for every timestep
        outputs = torch.zeros(
            batch_size,
            target_length,
            output_vocab_size,
            device=self.device
        )

        # Encode input
        hidden, cell = self.encoder(
            input_ids,
            input_lengths
        )

        # First decoder input = <SOS>
        decoder_input = target_ids[:, 0]

        # Generate target sequence
        for t in range(1, target_length):

            output, hidden, cell = self.decoder(
                decoder_input,
                hidden,
                cell
            )

            outputs[:, t] = output

            # Choose whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # Model prediction
            top1 = output.argmax(1)

            # Next decoder input
            decoder_input = (
                target_ids[:, t]
                if teacher_force
                else top1
            )

        return outputs
    
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using:", DEVICE)

model = Seq2Seq(
    encoder,
    decoder,
    DEVICE
).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


criterion = nn.CrossEntropyLoss(
    ignore_index=output_vocab["<PAD>"]
)

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    teacher_forcing_ratio=0.5
):

    model.train()

    epoch_loss = 0

    for input_ids, input_lengths, target_ids, target_lengths in loader:

        input_ids = input_ids.to(device)
        input_lengths = input_lengths.to(device)
        target_ids = target_ids.to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids,
            input_lengths,
            target_ids,
            teacher_forcing_ratio
        )

        output_dim = outputs.size(-1)

        outputs = outputs[:, 1:].contiguous()
        target_ids = target_ids[:, 1:].contiguous()

        outputs = outputs.view(-1, output_dim)
        target_ids = target_ids.view(-1)

        loss = criterion(
            outputs,
            target_ids
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)

def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for input_ids, input_lengths, target_ids, target_lengths in loader:

            input_ids = input_ids.to(device)
            input_lengths = input_lengths.to(device)
            target_ids = target_ids.to(device)

            # No teacher forcing during validation
            outputs = model(
                input_ids,
                input_lengths,
                target_ids,
                teacher_forcing_ratio=0.0
            )

            output_dim = outputs.size(-1)

            outputs = outputs[:, 1:].contiguous()
            target_ids = target_ids[:, 1:].contiguous()

            outputs = outputs.view(-1, output_dim)
            target_ids = target_ids.view(-1)

            loss = criterion(
                outputs,
                target_ids
            )

            epoch_loss += loss.item()

    return epoch_loss / len(loader)

def predict(model, tokens, input_vocab, output_vocab, idx_to_char, device, max_length=30):

    model.eval()

    input_ids = torch.tensor(
        encode_input(tokens),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    input_lengths = torch.tensor(
        [len(tokens)],
        dtype=torch.long,
        device=device
    )

    with torch.no_grad():

        hidden, cell = model.encoder(
            input_ids,
            input_lengths
        )

        decoder_input = torch.tensor(
            [output_vocab["<SOS>"]],
            dtype=torch.long,
            device=device
        )

        generated = []

        for _ in range(max_length):

            output, hidden, cell = model.decoder(
                decoder_input,
                hidden,
                cell
            )

            # Pick highest probability character
            prediction = output.argmax(1)

            predicted_id = prediction.item()

            if predicted_id == output_vocab["<EOS>"]:
                break

            if predicted_id not in [
                output_vocab["<PAD>"],
                output_vocab["<SOS>"]
            ]:
                generated.append(
                    idx_to_char[predicted_id]
                )

            decoder_input = prediction

    return "".join(generated)

N_EPOCHS = 10

for epoch in range(N_EPOCHS):

    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        DEVICE,
        teacher_forcing_ratio=0.5
    )

    val_loss = evaluate(
        model,
        val_loader,
        criterion,
        DEVICE
    )

    print(
        f"Epoch {epoch + 1:02d} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )
    
print("\n===== FULL TEST EVALUATION =====")

total = 0
correct = 0

type_total = {
    "DATE ONLY": 0,
    "DATE + TIME": 0,
    "TIME ONLY": 0,
    "NA": 0
}

type_correct = {
    "DATE ONLY": 0,
    "DATE + TIME": 0,
    "TIME ONLY": 0,
    "NA": 0
}


def get_target_type(example):

    date = example["date_iso"]
    time = example["time_hm"]

    if date is not None and time is not None:
        return "DATE + TIME"

    elif date is not None:
        return "DATE ONLY"

    elif time is not None:
        return "TIME ONLY"

    else:
        return "NA"


for example in test_data:

    prediction = predict(
        model,
        example["tokens"],
        input_vocab,
        output_vocab,
        idx_to_char,
        DEVICE
    )

    target = make_target(example)

    target_type = get_target_type(example)

    total += 1
    type_total[target_type] += 1

    if prediction == target:
        correct += 1
        type_correct[target_type] += 1


print("\nOverall:")
print(f"Correct: {correct}/{total}")
print(f"Exact Match Accuracy: {correct / total:.4f}")


print("\nBy Target Type:")

for target_type in type_total:

    total_type = type_total[target_type]
    correct_type = type_correct[target_type]

    accuracy = (
        correct_type / total_type
        if total_type > 0
        else 0
    )

    print(
        f"{target_type:12s}: "
        f"{correct_type}/{total_type} "
        f"({accuracy:.4f})"
    )
    

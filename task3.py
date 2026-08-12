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


# ---------------------------------------------------------------------
# Extract only the date/time-relevant span from the sentence
# ---------------------------------------------------------------------
DATE_TIME_TAGS = {"B-DATE", "I-DATE", "B-TIME", "I-TIME"}

def extract_relevant_span(example):
    tokens = example["tokens"]
    tags = example["tags"]

    relevant_idx = [
        i for i, t in enumerate(tags)
        if t in DATE_TIME_TAGS
    ]

    if not relevant_idx:
        # No date/time tags at all -> fall back to full sentence
        return tokens

    start = min(relevant_idx)
    end = max(relevant_idx)

    # Keep everything between first and last relevant tag
    # (this naturally keeps untagged in-between words, e.g. "at")
    return tokens[start:end + 1]


from collections import Counter

word_counter = Counter()

for example in train_data:
    word_counter.update(extract_relevant_span(example))

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


# ---------------------------------------------------------------------
# SLOT-BASED TARGET TOKENIZATION
# Every target is exactly 5 tokens: [YEAR, MONTH, DAY, HOUR, MINUTE]
# Missing date -> YEAR/MONTH/DAY = "NA","NA","NA"
# Missing time -> HOUR/MINUTE = "NA","NA"
# Missing both -> all five slots = "NA"
# ---------------------------------------------------------------------
def tokenize_target(example):

    date = example["date_iso"]   # e.g. "2026-06-15" or None
    time = example["time_hm"]    # e.g. "20:00" or None

    if date is not None:
        year, month, day = date.split("-")
    else:
        year, month, day = "NA", "NA", "NA"

    if time is not None:
        hour, minute = time.split(":")
    else:
        hour, minute = "NA", "NA"

    return [year, month, day, hour, minute]


def reconstruct_from_slots(slots):
    """
    Inverse of tokenize_target: turns [year,month,day,hour,minute]
    back into the same string format make_target() produces, so we
    can directly compare predictions against targets.
    """
    year, month, day, hour, minute = slots

    date_part = None
    if year != "NA":
        date_part = f"{year}-{month}-{day}"

    time_part = None
    if hour != "NA":
        time_part = f"{hour}:{minute}"

    if date_part is not None and time_part is not None:
        return f"{date_part} {time_part}"
    elif date_part is not None:
        return date_part
    elif time_part is not None:
        return time_part
    else:
        return "NA"


slot_value_set = set()

for example in train_data:
    for tok in tokenize_target(example):
        slot_value_set.add(tok)

output_vocab = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2
}

for tok in sorted(slot_value_set):
    output_vocab[tok] = len(output_vocab)

idx_to_char = {
    idx: tok
    for tok, idx in output_vocab.items()
}

def encode_input(tokens):
    return [
        input_vocab.get(word, input_vocab["<UNK>"])
        for word in tokens
    ]

def encode_target(example):
    """
    Now takes the full example (not a string), since slots come
    straight from date_iso / time_hm.
    """
    encoded = [output_vocab["<SOS>"]]

    for tok in tokenize_target(example):
        # Unseen slot values (e.g. a year not in train) fall back to NA
        # rather than crashing -- shouldn't normally happen but is safe.
        encoded.append(output_vocab.get(tok, output_vocab["NA"]))

    encoded.append(output_vocab["<EOS>"])

    return encoded

def decode_target(indices):
    """
    Strips SOS/PAD/EOS, then reconstructs the date/time string from
    the 5 slot tokens.
    """
    toks = []

    for idx in indices:

        if idx == output_vocab["<EOS>"]:
            break

        if idx in [
            output_vocab["<PAD>"],
            output_vocab["<SOS>"]
        ]:
            continue

        toks.append(idx_to_char[idx])

    # Pad defensively in case generation stopped early / malformed
    while len(toks) < 5:
        toks.append("NA")

    return reconstruct_from_slots(toks[:5])

class CalendarDataset(Dataset):

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        example = self.data[idx]

        span_tokens = extract_relevant_span(example)
        input_ids = encode_input(span_tokens)

        target_ids = encode_target(example)

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


# ---------------------------------------------------------------------
# Bidirectional encoder
# ---------------------------------------------------------------------
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

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

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
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, input_ids, input_lengths):

        embedded = self.embedding(input_ids)

        packed = pack_padded_sequence(
            embedded,
            input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed)

        batch_size = hidden.size(1)

        hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_dim)
        cell = cell.view(self.num_layers, 2, batch_size, self.hidden_dim)

        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=-1)
        cell = torch.cat([cell[:, 0], cell[:, 1]], dim=-1)

        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))

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

        outputs = torch.zeros(
            batch_size,
            target_length,
            output_vocab_size,
            device=self.device
        )

        hidden, cell = self.encoder(
            input_ids,
            input_lengths
        )

        decoder_input = target_ids[:, 0]

        for t in range(1, target_length):

            output, hidden, cell = self.decoder(
                decoder_input,
                hidden,
                cell
            )

            outputs[:, t] = output

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = output.argmax(1)

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

def predict(model, example, input_vocab, output_vocab, idx_to_char, device, max_length=5):
    """
    Takes the full example dict (needs tags for the span, same as
    training). Generates exactly up to 5 slot tokens and reconstructs
    the date/time string.
    """

    model.eval()

    span_tokens = extract_relevant_span(example)

    input_ids = torch.tensor(
        encode_input(span_tokens),
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    input_lengths = torch.tensor(
        [len(span_tokens)],
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

    while len(generated) < 5:
        generated.append("NA")

    return reconstruct_from_slots(generated[:5])

N_EPOCHS = 15

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
        example,
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

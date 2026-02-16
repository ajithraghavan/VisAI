import os
import random
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Config:

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"

    MAX_LENGTH = 30

    # Encoder Architecture
    ENCODER_EMBEDDING_DIM = 64
    CONV_FILTERS = 128
    CONV_KERNEL_SIZE = 3
    ENCODER_HIDDEN_SIZE = 256
    ENCODER_NUM_LAYERS = 1

    # Decoder Architecture
    DECODER_EMBEDDING_DIM = 64
    DECODER_HIDDEN_SIZE = 256
    DECODER_NUM_LAYERS = 1

    DROPOUT = 0.3

    # Training Settings
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 50
    TEACHER_FORCING_RATIO = 0.5
    CLIP_GRAD = 1.0

    # Early Stopping
    EARLY_STOPPING_PATIENCE = 15

    # Scheduler
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5

    # Paths
    TRAIN_PATH = "ta.translit.sampled.train.tsv"
    DEV_PATH = "ta.translit.sampled.dev.tsv"
    TEST_PATH = "ta.translit.sampled.test.tsv"
    CHECKPOINT_PATH = "checkpoints/best_model_cnn_lstm.pt"

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random seed
    SEED = 42

def set_seed(seed: int):

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class Vocabulary:

    def __init__(self, name: str):

        self.name = name
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        self.char_count: Counter = Counter()
        self.n_chars = 0

        self._add_special_tokens()

    def _add_special_tokens(self):

        for token in [Config.PAD_TOKEN, Config.SOS_TOKEN, Config.EOS_TOKEN, Config.UNK_TOKEN]:
            self._add_char(token)

    def _add_char(self, char: str):

        if char not in self.char2idx:
            self.char2idx[char] = self.n_chars
            self.idx2char[self.n_chars] = char
            self.n_chars += 1

    def add_word(self, word: str):

        for char in word:
            self.char_count[char] += 1
            self._add_char(char)

    def encode(self, word: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:

        indices = []
        if add_sos:
            indices.append(self.char2idx[Config.SOS_TOKEN])

        for char in word:
            idx = self.char2idx.get(char, self.char2idx[Config.UNK_TOKEN])
            indices.append(idx)

        if add_eos:
            indices.append(self.char2idx[Config.EOS_TOKEN])

        return indices

    def decode(self, indices: List[int], remove_special: bool = True) -> str:

        chars = []
        special_tokens = {Config.PAD_TOKEN, Config.SOS_TOKEN, Config.EOS_TOKEN, Config.UNK_TOKEN}

        for idx in indices:
            char = self.idx2char.get(idx, Config.UNK_TOKEN)
            if remove_special and char in special_tokens:
                if char == Config.EOS_TOKEN:
                    break
                continue
            chars.append(char)

        return "".join(chars)

    @property
    def pad_idx(self) -> int:
        return self.char2idx[Config.PAD_TOKEN]

    @property
    def sos_idx(self) -> int:
        return self.char2idx[Config.SOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.char2idx[Config.EOS_TOKEN]

    def __len__(self) -> int:
        return self.n_chars

class TransliterationDataset(Dataset):

    def __init__(self, filepath: str, src_vocab: Optional[Vocabulary] = None, trg_vocab: Optional[Vocabulary] = None, build_vocab: bool = False):
        self.pairs: List[Tuple[str, str]] = []
        self.src_vocab = src_vocab or Vocabulary("thanglish")
        self.trg_vocab = trg_vocab or Vocabulary("tamil")

        self._load_data(filepath)

        if build_vocab:
            self._build_vocab()

        print(f"Loaded Dataset! \n Totally : {len(self.pairs)} pairs from {filepath}")


    def _load_data(self, filepath: str):

        with open(filepath, 'r', encoding = "utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                if len(parts) >= 2:
                    tamil_word = parts[0]
                    thanglish_word = parts[1].lower()

                # If you wonder, we are truncating the pair if anyone `thanglish` or `tamil` words exceeds the MAX_LENGTH
                if len(thanglish_word) <= Config.MAX_LENGTH and len(tamil_word) <= Config.MAX_LENGTH:
                    self.pairs.append((thanglish_word, tamil_word))

    def _build_vocab(self):

        for src, trg in self.pairs:
            self.src_vocab.add_word(src)
            self.trg_vocab.add_word(trg)


        print(f"Source Vocabulary Size : {len(self.src_vocab)}")
        print(f"Target Vocabulary Size : {len(self.trg_vocab)}")

    def __len__(self) -> int:
        return len(self.pairs)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:

        src_word, trg_word = self.pairs[idx]

        src_indices = self.src_vocab.encode(src_word, add_sos = False, add_eos = False)
        trg_indices = self.trg_vocab.encode(trg_word, add_sos = True, add_eos = True)

        src_tensor = torch.tensor(src_indices, dtype = torch.long)
        trg_tensor = torch.tensor(trg_indices, dtype = torch.long)

        return src_tensor, trg_tensor, len(src_indices), len(trg_indices)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    src_tensors, trg_tensors, src_lengths, trg_lengths = zip(*batch)

    sorted_indices = sorted(range(len(src_lengths)), key = lambda i: src_lengths[i], reverse = True)

    src_tensors = [src_tensors[i] for i in sorted_indices]
    trg_tensors = [trg_tensors[i] for i in sorted_indices]
    src_lengths = [src_lengths[i] for i in sorted_indices]
    trg_lengths = [trg_lengths[i] for i in sorted_indices]

    src_padded = pad_sequence(src_tensors, batch_first = True, padding_value = 0)
    trg_padded = pad_sequence(trg_tensors, batch_first = True, padding_value = 0)

    src_lengths = torch.tensor(src_lengths, dtype = torch.long)
    trg_lengths = torch.tensor(trg_lengths, dtype = torch.long)

    return src_padded, trg_padded, src_lengths, trg_lengths

def create_dataloaders(train_path: str, dev_path: str, test_path: str, batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:

    train_dataset = TransliterationDataset(train_path, build_vocab = True)
    src_vocab = train_dataset.src_vocab
    trg_vocab = train_dataset.trg_vocab

    dev_dataset = TransliterationDataset(dev_path, src_vocab = src_vocab, trg_vocab = trg_vocab)
    test_dateset = TransliterationDataset(test_path, src_vocab = src_vocab, trg_vocab = trg_vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = collate_fn,
        num_workers = 0,
        pin_memory = True if torch.cuda.is_available() else False
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size = batch_size,
        shuffle = False,
        collate_fn=collate_fn,
        num_workers = 0
    )

    test_loader = DataLoader(
        test_dateset,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = collate_fn,
        num_workers = 0
    )

    return train_loader, dev_loader, test_loader, src_vocab, trg_vocab


class CNNLSTMEncoder(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, conv_filters: int, conv_kernel_size: int, hidden_size: int, num_layers: int, dropout: float, pad_idx: int):

        super(CNNLSTMEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = pad_idx
        )

        self.conv = nn.Conv1d(
            in_channels = embedding_dim,
            out_channels = conv_filters,
            kernel_size = conv_kernel_size,
            padding = (conv_kernel_size - 1) // 2 # Which means "Same" Padding
        )

        self.lstm = nn.LSTM(
            input_size = conv_filters,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        embedded = self.dropout(self.embedding(src))

        conv_input = embedded.transpose(1, 2)
        conv_output = F.relu(self.conv(conv_input))

        lstm_input = conv_output.transpose(1, 2)
        lstm_input = self.dropout(lstm_input)

        packed = pack_padded_sequence(
            lstm_input,
            src_lengths.cpu(),
            batch_first = True,
            enforce_sorted = True
        )

        _, (hidden, cell) = self.lstm(packed)

        return hidden, cell

class LSTMDecoder(nn.Module):

    def __init__(self, vocab_size:int, embedding_dim: int, hidden_size: int, num_layers: int, dropout: float, pad_idx: int):

        super(LSTMDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx = pad_idx
        )

        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)


    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        embedded = self.dropout(self.embedding(input_token))

        embedded = embedded.unsqueeze(1)

        lstm_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))

        lstm_output = lstm_output.squeeze(1)

        output = self.fc_out(lstm_output)

        return output, hidden, cell

class CNNLSTMSeq2Seq(nn.Module):

    def __init__(self, encoder: CNNLSTMEncoder, decoder: LSTMDecoder, src_vocab: Vocabulary, trg_vocab: Vocabulary, device: torch.device):

        super(CNNLSTMSeq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.device = device

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:

        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len - 1, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src, src_lengths)

        input_token = trg[:, 0]

        for t in range(1, trg_len):

            output, hidden, cell = self.decoder(input_token, hidden, cell)

            outputs[:, t-1] = output

            use_teacher_forcing = random.random() < teacher_forcing_ratio

            if use_teacher_forcing:
                input_token = trg[:, t]
            else:
                input_token = output.argmax(1)

        return outputs

    def translate(self, src_word: str, max_length: int = Config.MAX_LENGTH) -> str:

        self.eval()

        with torch.no_grad():

            src_indices = self.src_vocab.encode(src_word.lower(), add_sos = False, add_eos = False)
            src_tensor = torch.tensor(src_indices, dtype = torch.long).unsqueeze(0).to(self.device)
            src_lengths = torch.tensor([len(src_indices)], dtype = torch.long)

            hidden, cell = self.encoder(src_tensor, src_lengths)

            input_token = torch.tensor([self.trg_vocab.sos_idx], dtype = torch.long).to(self.device)

            output_indices = []

            for _ in range(max_length):
                output, hidden, cell = self.decoder(input_token, hidden, cell)
                predicted_idx = output.argmax(1).item()

                if predicted_idx == self.trg_vocab.eos_idx:
                    break

                output_indices.append(predicted_idx)
                input_token = torch.tensor([predicted_idx], dtype = torch.long).to(self.device)

            tamil_word = self.trg_vocab.decode(output_indices, remove_special = True)

            return tamil_word

def train_epoch(model: CNNLSTMSeq2Seq, dataloader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, clip: float, device: torch.device, teacher_forcing_ratio: float) -> float:

    model.train()
    epoch_loss = 0

    for batch_idx, (src, trg, src_lengths, trg_lengths) in enumerate(dataloader):

        src = src.to(device)
        trg = trg.to(device)
        src_lengths = src_lengths.to(device)

        optimizer.zero_grad()

        outputs = model(src, src_lengths, trg, teacher_forcing_ratio)

        output_dim = outputs.shape[-1]

        outputs = outputs.reshape(-1, output_dim)

        trg = trg[:, 1:].reshape(-1)

        loss = criterion(outputs, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def evaluate(model: CNNLSTMSeq2Seq, dataloader: DataLoader, criterion: nn.Module, device: torch.device) -> float:

    model.eval()
    epoch_loss = 0

    with torch.no_grad():

        for src, trg, src_lengths, trg_lengths in dataloader:

            src = src.to(device)
            trg = trg.to(device)
            src_lengths = src_lengths.to(device)

            outputs = model(src, src_lengths, trg, teacher_forcing_ratio = 0.0)

            output_dim = outputs.shape[-1]
            outputs = outputs.reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(outputs, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def calculate_metrics(model: CNNLSTMSeq2Seq, dataloader: DataLoader, device: torch.device) -> Tuple[float, float]:

    model.eval()

    total_samples = 0
    exact_matches = 0
    total_cer = 0.0

    with torch.no_grad():

        for src, trg, src_lengths, trg_lengths in dataloader:

            src = src.to(device)
            batch_size = src.size(0)

            for i in range(batch_size):

                src_indices = src[i, :src_lengths[i]].cpu().tolist()
                src_word = model.src_vocab.decode(src_indices, remove_special = True)

                trg_indices = trg[i, 1:trg_lengths[i]].cpu().tolist()
                trg_word = model.trg_vocab.decode(trg_indices, remove_special = True)

                pred_word = model.translate(src_word)

                if pred_word == trg_word:
                    exact_matches += 1

                cer = levenshtein_distance(pred_word, trg_word) / max(len(trg_word), 1)
                total_cer += cer

                total_samples += 1

        exact_match_accuracy = (exact_matches / total_samples) * 100 if total_samples > 0 else 0
        average_cer = (total_cer / total_samples) * 100 if total_samples > 0 else 0

        return exact_match_accuracy, average_cer

def levenshtein_distance(s1: str, s2: str) -> int:

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def train(model: CNNLSTMSeq2Seq, train_loader: DataLoader, dev_loader: DataLoader, optimizer: optim.Optimizer, scheduler, criterion: nn.Module, epochs: int, clip: float, device: torch.device, checkpoint_path: str, teacher_forcing_ratio: float) -> Dict:

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_cer": [],
        "learning_rates": []
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):

        train_loss = train_epoch(model, train_loader, optimizer, criterion, clip, device, teacher_forcing_ratio)
        val_loss = evaluate(model, dev_loader, criterion, device)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_accuracy, val_cer = calculate_metrics(model, dev_loader, device)
        else:
            val_accuracy = history["val_accuracy"][-1] if history["val_accuracy"] else 0
            val_cer = history["val_cer"][-1] if history["val_cer"] else 0

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if current_lr < old_lr:
            print(f"Learning Rate Reduced From {old_lr:.6f} To {current_lr:.6f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_cer"].append(val_cer)
        history["learning_rates"].append(current_lr)

        print(f"Epoch {epoch + 1:3d}/{epochs} | "
              f"Train Loss : {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc : {val_accuracy:.2f}% | "
              f"Val CER : {val_cer:.2f}% | "
              f"LR : {current_lr:.6f}")

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "src_vocab": model.src_vocab,
                "trg_vocab": model.trg_vocab,
                "config": {
                    "encoder_embedding_dim": Config.ENCODER_EMBEDDING_DIM,
                    "conv_filters": Config.CONV_FILTERS,
                    "conv_kernel_size": Config.CONV_KERNEL_SIZE,
                    "encoder_hidden_size": Config.ENCODER_HIDDEN_SIZE,
                    "encoder_num_layers": Config.ENCODER_NUM_LAYERS,
                    "decoder_embedding_dim": Config.DECODER_EMBEDDING_DIM,
                    "decoder_hidden_size": Config.DECODER_HIDDEN_SIZE,
                    "decoder_num_layers": Config.DECODER_NUM_LAYERS,
                    "dropout": Config.DROPOUT
                }
            }, checkpoint_path)

            print(f"Saved best model (val_loss: {val_loss:.4f})")

        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    return history

def load_model(checkpoint_path: str, device: torch.device) -> CNNLSTMSeq2Seq:

    checkpoint = torch.load(checkpoint_path, map_location = device, weights_only = False)

    src_vocab = checkpoint["src_vocab"]
    trg_vocab = checkpoint["trg_vocab"]
    config = checkpoint["config"]

    encoder = CNNLSTMEncoder(
        vocab_size=len(src_vocab),
        embedding_dim=config["encoder_embedding_dim"],
        conv_filters=config["conv_filters"],
        conv_kernel_size=config["conv_kernel_size"],
        hidden_size=config["encoder_hidden_size"],
        num_layers=config["encoder_num_layers"],
        dropout=config["dropout"],
        pad_idx=src_vocab.pad_idx
    )

    decoder = LSTMDecoder(
        vocab_size=len(trg_vocab),
        embedding_dim=config["decoder_embedding_dim"],
        hidden_size=config["decoder_hidden_size"],
        num_layers=config["decoder_num_layers"],
        dropout=config["dropout"],
        pad_idx=trg_vocab.pad_idx
    )

    model = CNNLSTMSeq2Seq(encoder, decoder, src_vocab, trg_vocab, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded from {checkpoint_path}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Val Loss: {checkpoint['val_loss']:.4f}")

    return model


def translate_word(model: CNNLSTMSeq2Seq, thanglish_word: str) -> str:
    return model.translate(thanglish_word)


def translate_batch(model: CNNLSTMSeq2Seq, thanglish_words: List[str]) -> List[str]:
    return [translate_word(model, word) for word in thanglish_words]

def build_model(src_vocab: Vocabulary, trg_vocab: Vocabulary, device: torch.device) -> CNNLSTMSeq2Seq:

    encoder = CNNLSTMEncoder(
        vocab_size=len(src_vocab),
        embedding_dim=Config.ENCODER_EMBEDDING_DIM,
        conv_filters=Config.CONV_FILTERS,
        conv_kernel_size=Config.CONV_KERNEL_SIZE,
        hidden_size=Config.ENCODER_HIDDEN_SIZE,
        num_layers=Config.ENCODER_NUM_LAYERS,
        dropout=Config.DROPOUT,
        pad_idx=src_vocab.pad_idx
    )

    decoder = LSTMDecoder(
        vocab_size=len(trg_vocab),
        embedding_dim=Config.DECODER_EMBEDDING_DIM,
        hidden_size=Config.DECODER_HIDDEN_SIZE,
        num_layers=Config.DECODER_NUM_LAYERS,
        dropout=Config.DROPOUT,
        pad_idx=trg_vocab.pad_idx
    )

    model = CNNLSTMSeq2Seq(encoder, decoder, src_vocab, trg_vocab, device)
    model.to(device)

    return model

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():

    set_seed(Config.SEED)

    print("=" * 60)
    print("Thanglish to Tamil Transliteration Model (CNN-LSTM Hybrid)")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Architecture: Conv1D -> LSTM Encoder -> LSTM Decoder")
    print(f"  Encoder:")
    print(f"    - Embedding: {Config.ENCODER_EMBEDDING_DIM}")
    print(f"    - Conv1D: {Config.CONV_FILTERS} filters, kernel={Config.CONV_KERNEL_SIZE}")
    print(f"    - LSTM: {Config.ENCODER_HIDDEN_SIZE} hidden, {Config.ENCODER_NUM_LAYERS} layers")
    print(f"  Decoder:")
    print(f"    - Embedding: {Config.DECODER_EMBEDDING_DIM}")
    print(f"    - LSTM: {Config.DECODER_HIDDEN_SIZE} hidden, {Config.DECODER_NUM_LAYERS} layers")
    print(f"  Training:")
    print(f"    - Teacher Forcing: {Config.TEACHER_FORCING_RATIO}")
    print(f"    - Learning Rate: {Config.LEARNING_RATE}")
    print("\n")

    print("Loading Data")
    train_loader, dev_loader, test_loader, src_vocab, trg_vocab = create_dataloaders(
        Config.TRAIN_PATH,
        Config.DEV_PATH,
        Config.TEST_PATH,
        Config.BATCH_SIZE
    )
    print("\n")

    print("Building Model")
    model = build_model(src_vocab, trg_vocab, Config.DEVICE)
    print(f"Model Parameters : {count_parameters(model):,}")
    print("\n")

    criterion = nn.CrossEntropyLoss(ignore_index = trg_vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr = Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode = "min",
        patience = Config.SCHEDULER_PATIENCE,
        factor = Config.SCHEDULER_FACTOR
    )

    print("Starting Training")
    print("-" * 60)
    history = train(
        model = model,
        train_loader = train_loader,
        dev_loader = dev_loader,
        optimizer = optimizer,
        scheduler = scheduler,
        criterion = criterion,
        epochs = Config.EPOCHS,
        clip = Config.CLIP_GRAD,
        device = Config.DEVICE,
        checkpoint_path = Config.CHECKPOINT_PATH,
        teacher_forcing_ratio = Config.TEACHER_FORCING_RATIO
    )
    print("-" * 60)
    print("\n")

    print("Evaluating on Test Set")
    model = load_model(Config.CHECKPOINT_PATH, Config.DEVICE)
    test_accuracy, test_cer = calculate_metrics(model, test_loader, Config.DEVICE)
    print(f"Test Exact Match Accuracy : {test_accuracy:.2f}%")
    print(f"Test Character Error Rate : {test_cer:.2f}%")
    print("\n")

    print("Demo Translations :")
    print("-" * 40)
    demo_words = ["vanakkam", "puthagam", "tamizh", "nandri", "agaraathikal"]
    for word in demo_words:
        tamil = translate_word(model, word)
        print(f"  {word:20s} → {tamil}")

    return model, history



model, history = main()
import os
import random
import math
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class Config:

    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "UNK"

    MAX_LENGTH = 30

    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 3
    D_FF = 512
    DROPOUT = 0.1

    BATCH_SIZE = 64
    LR = 0.001
    EPOCHS = 50
    CLIP_GRAD = 1.0

    EARLY_STOPPING_PATIENCE = 8
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5

    TRAIN_PATH = "dataset/ta.translit.sampled.train.tsv"
    DEV_PATH = "dataset/ta.translit.sampled.dev.tsv"
    TEST_PATH = "dataset/ta.translit.sampled.test.tsv"
    CHECKPINT_PATH = "checkpoints/best_model_transformer.pt"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    

class TransLiterationDataset(Dataset):

    def __init__(self, filepath: str, src_vocab: Optional[Vocabulary] = None, trg_vocab: Optional[Vocabulary] = None, build_vocab: bool = False):

        self.pairs: List[Tuple[str, str]] = []
        self.src_vocab = src_vocab or Vocabulary("thanglish")
        self.trg_vocab = trg_vocab or Vocabulary("tamil")
        self._load_data(filepath)

        if build_vocab:
            self._build_vocab()

        print(f"Loaded Dataset! \nTotally : {len(self.pairs)} from {filepath}")

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
    
    def __getitem__(self, idx: int):

        src_word, trg_word = self.pairs[idx]
        src_indices = self.src_vocab.encode(src_word, add_sos = False, add_eos = False)
        trg_indices = self.trg_vocab.encode(trg_word, add_sos = True, add_eos = True)
        return (torch.tensor(src_indices, dtype = torch.long), torch.tensor(trg_indices, dtype = torch.long), len(src_indices), len(trg_indices))
    
def collate_fn(batch):

    src_tensors, trg_tensors, src_lengths, trg_lenths = zip(*batch)

    src_padded = pad_sequence(src_tensors, batch_first = True, padding_value = 0)
    trg_padded = pad_sequence(trg_tensors, batch_first = True, padding_value = 0)

    return src_padded, trg_padded, torch.tensor(src_lengths), torch.tensor(trg_lenths)

def create_dataloaders(train_path, dev_path, test_path, batch_size):

    train_dataset = TransLiterationDataset(train_path, build_vocab = True)
    src_vocab, trg_vocab = train_dataset.src_vocab, train_dataset.trg_vocab
    
    dev_dataset = TransLiterationDataset(dev_path, src_vocab = src_vocab, trg_vocab = trg_vocab)
    test_dataset = TransLiterationDataset(test_path, src_vocab = src_vocab, trg_vocab = trg_vocab)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn, pin_memory = torch.cuda.is_available())
    dev_loader = DataLoader(dev_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_fn)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, collate_fn = collate_fn)

    return train_loader, dev_loader, test_loader, src_vocab, trg_vocab


class PositionalEncodeing(nn.Module):

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
    
class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        d_k = query.size(-1)

        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(d_k)

        # Only for Masked Attention
        if mask is not None:
            scores = scores.masked_fill(mask == True, float('-inf'))

        attention_weights = F.softmax(scores, dim = -1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)

        return output

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:

        batch_size = query.size(0)

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attended = self.attention(Q, K, V, mask)

        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.W_o(attended)

        return output
    
class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):

        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:

        attended = self.self_attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attended))

        feed_forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(feed_forward))

        return x
    
class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):

        super().__init__()

        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, trg_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:

        self_attended = self.self_attention(x, x, x, trg_mask)

        x = self.norm1(x + self.dropout(self_attended))

        cross_attended = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attended))
        
        feed_forward = self.feed_forward(x)
        x = self.norm3(x + self.dropout(feed_forward))

        return x
    
class Encoder(nn.Module):

    def __init__(self, vocab_size: int, d_model:int, n_layers: int, n_heads: int, d_ff: int, dropout: float, pad_idx: int):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx = pad_idx)
        self.positional_encoding = PositionalEncodeing(d_model, dropout = dropout)
        self.scale = math.sqrt(d_model)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:

        x = self.embedding(src) * self.scale

        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
    
class Decoder(nn.Module):
    
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float, pad_idx: int):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx = pad_idx)
        self.positional_encoding = PositionalEncodeing(d_model, dropout = dropout)
        self.scale = math.sqrt(d_model)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg: torch.Tensor, encoder_output: torch.Tensor, trg_mask: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:

        x = self.embedding(trg) * self.scale

        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, encoder_output, trg_mask, src_mask)

        return x
    

class Transformer(nn.Module):

    def __init__(self, src_vocab_size: int, trg_vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int, dropout: float, src_pad_idx: int, trg_pad_idx: int, device: torch.device):

        super().__init__()

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, src_pad_idx)
        self.decoder = Decoder(trg_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, trg_pad_idx)

        self.fc_out = nn.Linear(d_model, trg_vocab_size)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:

        src_mask = (src == self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        
        return src_mask
    
    def make_trg_mask(self, trg: torch.Tensor) -> torch.Tensor:

        batch_size, trg_len = trg.shape

        trg_pad_mask = (trg == self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        trg_casual_mask = torch.triu(torch.ones(trg_len, trg_len, device = self.device), diagonal = 1).bool()
        trg_causal_mask = trg_casual_mask.unsqueeze(0).unsqueeze(1)

        trg_mask = trg_pad_mask | trg_casual_mask

        return trg_mask
    
    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:

        src_mask = self.make_src_mask(src)
        
        trg_input = trg[:, :-1]
        trg_mask = self.make_trg_mask(trg_input)

        encoder_output = self.encoder(src, src_mask)

        decoder_output = self.decoder(trg_input, encoder_output, trg_mask, src_mask)

        output = self.fc_out(decoder_output)

        return output

    def translate(self, src_word: str, src_vocab: Vocabulary, trg_vocab: Vocabulary, max_length: int = Config.MAX_LENGTH) -> str:

        self.eval()

        with torch.no_grad():

            src_indices = src_vocab.encode(src_word.lower())
            src_tensor = torch.tensor(src_indices, dtype = torch.long).unsqueeze(0).to(self.device)
            src_mask = self.make_src_mask(src_tensor)
            encoder_output = self.encoder(src_tensor, src_mask)

            trg_indices = [trg_vocab.sos_idx]

            for _ in range(max_length):

                trg_tensor = torch.tensor(trg_indices, dtype = torch.long).unsqueeze(0).to(self.device)
                trg_mask = self.make_trg_mask(trg_tensor)

                decoder_output = self.decoder(trg_tensor, encoder_output, trg_mask, src_mask)
                output = self.fc_out(decoder_output)

                pred = output[0, -1, :].argmax().item()

                if pred == trg_vocab.eos_idx:
                    break

                trg_indices.append(pred)
                
            return trg_vocab.decode(trg_indices[1:], remove_special = True)
        
def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    
    model.train()

    epoch_loss = 0

    for src, trg, _, _ in dataloader:

        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        output = model(src, trg)

        output = output.reshape(-1, output.shape[-1])

        trg = trg[:, 1:].reshape(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for src, trg, _, _ in dataloader:

            src, trg = src.to(device), trg.to(device)

            output = model(src, trg)
            output = output.reshape(-1, output.shape[-1])

            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def calculate_metrics(model, dataloader, src_vocab, trg_vocab, device):
    
    model.eval()

    total, exact_matches, total_cer = 0, 0, 0.0

    with torch.no_grad():

        for src, trg, src_lengths, trg_lengths in dataloader:

            src = src.to(device)

            for i in range(src.size(0)):

                src_word = src_vocab.decode(src[i, :src_lengths[i]].cpu().tolist())
                trg_word = trg_vocab.decode(trg[i, 1:trg_lengths[i]].cpu().tolist())
                pred_word = model.translate(src_word, src_vocab, trg_vocab)

                if pred_word == trg_word:
                    exact_matches += 1
                total_cer += levenshtein_distance(pred_word, trg_word) / max(len(trg_word), 1)

                total += 1
    return (exact_matches / total) * 100, (total_cer / total) * 100

def levenshtein_distance(s1: str, s2: str) -> int:

    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            curr_row.append(min(prev_row[j + 1] + 1, curr_row[j] + 1, prev_row[j] + (c1 != c2)))
        prev_row = curr_row

    return prev_row[-1]

def train(model, train_loader, dev_loader, optimizer, scheduler, criterion, src_vocab, trg_vocab, epochs, clip, device, checkpoint_path):

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], 
               "val_loss": [],
               "val_accuracy": [],
               "val_cer": []}
    
    for epoch in range(epochs):

        train_loss = train_epoch(model, train_loader, optimizer, criterion, clip, device)
        val_loss = evaluate(model, dev_loader, criterion, device)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_acc, val_cer = calculate_metrics(model, dev_loader, src_vocab, trg_vocab, device)
        else:
            val_acc = history["val_accuracy"][-1] if history["val_accuracy"] else 0
            val_cer = history["val_cer"][-1] if history["val_cer"] else 0

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr < old_lr:
            print(f"Learning Rage Reduced : {old_lr:.6f} -> {new_lr:.6f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_cer"].append(val_cer)

        print(f"Epoch {epoch + 1: 3d}/{epochs} | Train Loss : {train_loss:.4f} | " f"Val Loss : {val_loss:.4f} | Val Acc : {val_acc:.2f}% | Val CER : {val_cer:.2f}%")

        if val_loss < best_val_loss:
            
            best_val_loss = val_loss
            patience_counter = 0
            
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok = True)
            
            torch.save({
                "epoch": epoch, 
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "src_vocab": src_vocab,
                "trg_vocab": trg_vocab,
                "config": {
                    "d_model": Config.D_MODEL,
                    "n_heads": Config.N_HEADS,
                    "n_layers": Config.N_LAYERS,
                    "d_ff": Config.D_FF,
                    "dropout": Config.DROPOUT
                }
            }, checkpoint_path)
            print(f"Saved the Best Model!")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly Stopping After {epoch + 1}")
                break

    return history


def load_model(checkpoint_path: str, device: torch.device):

    checkpoint = torch.load(checkpoint_path, map_location = device, weights_only = False)

    src_vocab, trg_vocab = checkpoint["src_vocab"], checkpoint["trg_vocab"]

    config = checkpoint["config"]

    model = Transformer(len(src_vocab), len(trg_vocab), config["d_model"], config["n_layers"], config["n_heads"], config["d_ff"], config["dropout"], src_vocab.pad_idx, trg_vocab.pad_idx, device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded Model from {checkpoint_path}")

    return model, src_vocab, trg_vocab

def main():
    
    set_seed(Config.SEED)

    print("=" * 100)
    print("Thanglish to Tamil Tranliteration using Transformer(Based on 'Attention Is All You Need')")
    print("=" * 100)
    print(f"\nDevice : {Config.DEVICE}")
    print(f"\nArchitecture : ")
    print(f"    d_model (Embedding Dim)      : {Config.D_MODEL}")
    print(f"    n_heads (Attenstion Heads)   : {Config.N_HEADS}")
    print(f"    n_layers (Encoder / Decoder) : {Config.N_LAYERS}")
    print(f"    d_ff (Feed Formward Dim)     : {Config.D_FF}")
    print(f"    dropout                      : {Config.DROPOUT}")
    print()

    print("Loading Data..........")
    train_loader, dev_loader, test_loader, src_vocab, trg_vocab = create_dataloaders(Config.TRAIN_PATH, Config.DEV_PATH, Config.TEST_PATH, Config.BATCH_SIZE)
    
    model = Transformer(
        src_vocab_size = len(src_vocab),
        trg_vocab_size = len(trg_vocab),
        d_model = Config.D_MODEL,
        n_layers = Config.N_LAYERS,
        n_heads = Config.N_HEADS,
        d_ff = Config.D_FF,
        dropout = Config.DROPOUT,
        src_pad_idx = src_vocab.pad_idx,
        trg_pad_idx = trg_vocab.pad_idx,
        device = Config.DEVICE
    ).to(Config.DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters : {n_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index = trg_vocab.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr = Config.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", patience = Config.SCHEDULER_PATIENCE, factor = Config.SCHEDULER_FACTOR)

    print("\n" + "=" * 100)
    print("Training")
    print("=" * 100)

    history = train(model, train_loader, dev_loader, optimizer, scheduler, criterion, src_vocab, trg_vocab, Config.EPOCHS, Config.CLIP_GRAD, Config.DEVICE, Config.CHECKPINT_PATH)

    print("\n" + "=" * 100)
    print("Evaluation on Test Set")
    print("=" * 100)

    model, src_vocab, trg_vocab = load_model(Config.CHECKPINT_PATH, Config.DEVICE)

    test_acc, test_cer = calculate_metrics(model, test_loader, src_vocab, trg_vocab, Config.DEVICE)

    print(f"Test Accuracy : {test_acc:.2f}%")
    print(f"Test CER      : {test_cer:.2f}%")

    print("\n" + "=" * 100)
    print("Demo Tranlations")
    print("=" * 100)
    
    demo_words = ["vanakkam", "puthagam", "tamizh", "nandri", "agaraathikal"]

    for word in demo_words:
        tranlation = model.translate(word, src_vocab, trg_vocab)
        print(f"    {word:20s} -> {tranlation}")

    return model, history

if __name__ == "__main__":
    model, hisotry = main()
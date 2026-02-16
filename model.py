from typing import List, Dict, Optional
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    MAX_LENGTH = 30
    
    CHECKPOINT_PATH = "checkpoints/best_model_cnn_lstm.pt"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class CNNLSTMEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, conv_filters: int, 
                 conv_kernel_size: int, hidden_size: int, num_layers: int, 
                 dropout: float, pad_idx: int):
        super(CNNLSTMEncoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )
        
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=conv_filters,
            kernel_size=conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2
        )

        self.lstm = nn.LSTM(
            input_size=conv_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor):
        embedded = self.dropout(self.embedding(src))
        
        conv_input = embedded.transpose(1, 2)
        conv_output = F.relu(self.conv(conv_input))
        
        lstm_input = conv_output.transpose(1, 2)
        lstm_input = self.dropout(lstm_input)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            lstm_input,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=True
        )

        _, (hidden, cell) = self.lstm(packed)

        return hidden, cell


class LSTMDecoder(nn.Module):

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 num_layers: int, dropout: float, pad_idx: int):
        super(LSTMDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        embedded = self.dropout(self.embedding(input_token))
        embedded = embedded.unsqueeze(1)
        
        lstm_output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        lstm_output = lstm_output.squeeze(1)
        
        output = self.fc_out(lstm_output)

        return output, hidden, cell


class CNNLSTMSeq2Seq(nn.Module):
    def __init__(self, encoder: CNNLSTMEncoder, decoder: LSTMDecoder, 
                 src_vocab: Vocabulary, trg_vocab: Vocabulary, device: torch.device):
        super(CNNLSTMSeq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.device = device

    def translate(self, src_word: str, max_length: int = Config.MAX_LENGTH) -> str:

        self.eval()

        with torch.no_grad():
            # Encode source word
            src_indices = self.src_vocab.encode(src_word.lower(), add_sos=False, add_eos=False)
            src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(self.device)
            src_lengths = torch.tensor([len(src_indices)], dtype=torch.long)

            # Get encoder hidden states
            hidden, cell = self.encoder(src_tensor, src_lengths)

            # Start decoding with SOS token
            input_token = torch.tensor([self.trg_vocab.sos_idx], dtype=torch.long).to(self.device)
            
            output_indices = []

            # Generate output character by character
            for _ in range(max_length):
                output, hidden, cell = self.decoder(input_token, hidden, cell)
                predicted_idx = output.argmax(1).item()

                if predicted_idx == self.trg_vocab.eos_idx:
                    break

                output_indices.append(predicted_idx)
                input_token = torch.tensor([predicted_idx], dtype=torch.long).to(self.device)

            # Decode output indices to Tamil string
            tamil_word = self.trg_vocab.decode(output_indices, remove_special=True)
            
            return tamil_word

def load_model(checkpoint_path: str, device: Optional[torch.device] = None) -> CNNLSTMSeq2Seq:

    if device is None:
        device = Config.DEVICE
    
    print(f"Loading model from: {checkpoint_path}")
    print(f"Using device: {device}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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
    
    print(f"Model loaded successfully!")
    print(f"  - Training epoch: {checkpoint['epoch'] + 1}")
    print(f"  - Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"  - Source vocab size: {len(src_vocab)}")
    print(f"  - Target vocab size: {len(trg_vocab)}")
    
    return model


def translate_word(model: CNNLSTMSeq2Seq, thanglish_word: str) -> str:
    return model.translate(thanglish_word)


def translate_batch(model: CNNLSTMSeq2Seq, thanglish_words: List[str]) -> List[str]:
    return [translate_word(model, word) for word in thanglish_words]


def translate_file(model: CNNLSTMSeq2Seq, input_path: str, output_path: Optional[str] = None) -> List[tuple]:
    results = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    for word in words:
        tamil = translate_word(model, word)
        results.append((word, tamil))
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            for thanglish, tamil in results:
                f.write(f"{thanglish}\t{tamil}\n")
        print(f"Results saved to: {output_path}")
    
    return results

def interactive_mode(model: CNNLSTMSeq2Seq):

    print("\n" + "=" * 50)
    print("Interactive Thanglish to Tamil Translator")
    print("=" * 50)
    print("Enter Thanglish words to translate.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("Thanglish > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            words = user_input.split()
            
            for word in words:
                tamil = translate_word(model, word)
                print(f"  {word} → {tamil}")
            
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def main():

    try:
        model = load_model(Config.CHECKPOINT_PATH, Config.DEVICE)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found: {Config.CHECKPOINT_PATH}")
        print("Please ensure the model has been trained and the checkpoint exists.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    
    interactive_mode(model)


if __name__ == "__main__":
    main()
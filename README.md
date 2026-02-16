# Thanglish to Tamil Transliteration

A character-level Seq2Seq deep learning system that transliterates **Thanglish** (Tamil written in English script) to **Tamil script**.

Type `vanakkam` &rarr; get `வணக்கம்`

Inspired by [Google Input Tools](https://www.google.com/inputtools/), this project explores building a transliteration engine from scratch, experimenting with multiple neural architectures to find the best fit.

---

## Table of Contents

- [Demo](#demo)
- [Dataset](#dataset)
- [Architecture](#architecture)
  - [Why Seq2Seq?](#why-seq2seq)
  - [The Intuition](#the-intuition)
  - [Architectures Explored](#architectures-explored)
- [Evaluation & Results](#evaluation--results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)

---

## Demo

The project includes a web-based UI (**VisaAI**) where you type Thanglish and it translates to Tamil in real-time as you press Space or Enter.

> **How it works:** Type a Thanglish word &rarr; press Space/Enter &rarr; the word is replaced with its Tamil transliteration.

---

## Dataset

Uses the [Google Dakshina Dataset](https://github.com/google-research-datasets/dakshina) — a large-scale collection of transliteration pairs.

| Split | File |
|---|---|
| Train | `ta.translit.sampled.train.tsv` |
| Dev | `ta.translit.sampled.dev.tsv` |
| Test | `ta.translit.sampled.test.tsv` |

This is a **character-level** task — the model learns mappings between individual characters, not words or tokens.

---

## Architecture

### Why Seq2Seq?

Transliteration is inherently a sequence-to-sequence problem: an input sequence of English characters maps to an output sequence of Tamil characters. The model uses an **Encoder** to read the input and produce a context representation, and a **Decoder** to generate the output.

### The Intuition

Consider the word `vanakkam`:
- To generate **"வ"**, the model needs to look at both `v` and `a`
- To generate **"ண"**, the model needs to look at `na` and decide between "ந", "ன", "ண" — which requires understanding what comes **before** and **after**
  - "ன", "ண" don't appear at the **start** of words
  - "ந" rarely appears at the **end** of words

This means the model needs to capture **local character patterns** (like `tt` &rarr; `ட்ட` in `pattam`).

**Key insight:** Both Attention and Convolution compute weighted combinations of input values, but Convolution is restricted to a **small local window** — which is exactly what character-level transliteration needs. This led to the CNN-LSTM architecture.

### Architectures Explored

| # | Architecture | Parameters | Approach |
|---|---|---|---|
| 1 | **Vanilla LSTM** | 1,411,890 | Standard LSTM Encoder-Decoder |
| 2 | **BiGRU + Attention** | 12,580,914 | Bidirectional GRU with Attention Mechanism |
| 3 | **CNN-LSTM** | 767,666 | 1D CNN Encoder + LSTM Decoder |

---

## Evaluation & Results

Trained on Google Colab. Results reflect current hyperparameter tuning with limited compute — further improvements are possible.

| Architecture | Parameters | Val Loss | Val Acc | Test Acc | Test CER | Train Loss |
|---|---|---|---|---|---|---|
| Vanilla LSTM | 1.41M | 1.4453 | 53.41% | 51.57% | 16.36% | 0.1314 |
| BiGRU + Attention | 12.58M | 1.3492 | 54.74% | 50.60% | 16.44% | 0.1460 |
| **CNN-LSTM** | **767K** | **0.9868** | 53.13% | 50.55% | **15.81%** | 0.0903 |

### Winner: CNN-LSTM

All three models perform similarly on test accuracy (~50-51%), but **CNN-LSTM wins** because:

- **Best generalization** — Lowest validation loss (0.9868) by a significant margin vs ~1.35-1.45 for others
- **Best Character Error Rate** — 15.81% on test
- **Smallest model** — 767K parameters, ~16x smaller than BiGRU+Attention, ~2x smaller than Vanilla LSTM
- **Best efficiency** — Comparable results with far fewer parameters signals the architecture suits this task well

**Why it works:** 1D Convolutions are excellent at capturing **local n-gram patterns** in character sequences — exactly what transliteration requires. The CNN extracts local features, and the LSTM handles sequential decoding.

> CNN-LSTM is not new — it's proven in OCR (Optical Character Recognition), speech recognition, and other sequence tasks where local pattern extraction matters.

---

## Project Structure

```
.
├── CNN_LSTM.py              # CNN-LSTM architecture (training code)
├── GRU_With_Attention.py    # BiGRU + Attention architecture (training code)
├── model.py                 # CNN-LSTM model for inference
├── app.py                   # FastAPI server
├── index.html               # Web UI (VisaAI)
├── requirements.txt         # Python dependencies
├── dataset/
│   ├── ta.translit.sampled.train.tsv
│   ├── ta.translit.sampled.dev.tsv
│   └── ta.translit.sampled.test.tsv
└── checkpoints/
    └── best_model_cnn_lstm.pt
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Thanglish_To_Tamil.git
cd Thanglish_To_Tamil
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the API server

```bash
uvicorn app:app --reload --port 8000
```

You should see:

```
Loading CNN-LSTM Translation Model...
Model loaded successfully!
Model ready for translation!
```

API docs will be available at: http://localhost:8000/docs

### 5. Open the Web UI

Open `index.html` in your browser. The status bar at the bottom should show **"API Connected & Model Loaded"**.

Type a Thanglish word and press **Space** or **Enter** to see the Tamil transliteration.

### 6. (Alternative) Interactive CLI mode

```bash
python model.py
```

This starts a terminal-based translator where you can type Thanglish words and see Tamil output directly.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check & model status |
| `POST` | `/translate` | Translate Thanglish to Tamil |

### Example request

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "vanakkam"}'
```

### Example response

```json
{
  "success": true,
  "input": "vanakkam",
  "translations": [
    { "thanglish": "vanakkam", "tamil": "வணக்கம்" }
  ]
}
```

---

## Tech Stack

- **PyTorch** — Model training & inference
- **FastAPI** — API server
- **HTML/CSS/JS** — Web UI
- **Google Dakshina Dataset** — Training data

# 🌿 Ayurvedic Leaf Detection and Retrieving Its Medicinal Properties using Deep Learning and NLP

A two-stage system that **identifies an Ayurvedic medicinal plant from a leaf
image** and then **retrieves its medicinal properties** — all offline, with
**no LLM dependency**.

| Stage | Technique | What it does |
|-------|-----------|--------------|
| 1. Detection | **Deep Learning** — CNN transfer learning (PyTorch) | Classifies the plant species from a leaf photo |
| 2. Retrieval | **NLP** — curated knowledge base + TF-IDF semantic search (scikit-learn) | Maps the species to its medicinal properties and lets you search herbs by symptom |

> ⚠️ **Educational project.** The medicinal information is compiled from common
> Ayurvedic / ethnobotanical references for learning and identification only.
> It is **not medical advice**.

---

## Results (this build)

Trained on the **Mendeley Medicinal Leaf Dataset** — 30 species, 1,835 images
(segmented), split 70 / 15 / 15 (train / val / test).

| Setting | Value |
|---------|-------|
| Backbone | ResNet18 (ImageNet transfer learning) |
| Hardware | NVIDIA RTX 3050 6 GB (CUDA) |
| Training time | ~48 s (early-stopped at epoch 9) |
| Best validation accuracy | **99.3 %** |
| **Test accuracy** | **99.64 %** (273 / 274) |
| Macro F1 (test) | **1.00** |

Per-class report and confusion matrix are written to `outputs/`. Example
end-to-end prediction on a Tulsi leaf → `Ocimum Tenuiflorum (Tulsi)` at 95.2 %
confidence, with its medicinal properties retrieved from the knowledge base.

> Reproduce: `python -m src.train` then `python -m src.evaluate`.

---

## Project structure

```
ayur/
├── app.py                     # Streamlit web app (identify + search + browse)
├── config.yaml                # All hyper-parameters & paths
├── requirements.txt
├── data/
│   ├── raw/                   # << put your dataset here (one folder per class)
│   └── knowledge_base/
│       └── plants.json        # 30 Ayurvedic herbs + medicinal properties
├── models/                    # trained checkpoints (best_model.pt)
├── outputs/                   # training curves, confusion matrix, metrics
├── scripts/
│   └── check_data.py          # validate dataset layout & KB coverage
└── src/
    ├── config.py              # typed config loader (YAML + CLI overrides)
    ├── data.py                # transforms, stratified split, dataloaders
    ├── model.py               # backbone builder + checkpoint I/O
    ├── train.py               # training loop (transfer learning)
    ├── evaluate.py            # test metrics + confusion matrix
    ├── predict.py             # image -> plant -> properties (CLI)
    └── nlp/
        ├── knowledge_base.py  # load + fuzzy-lookup herbs
        └── retriever.py       # TF-IDF semantic search (the "NLP" engine)
```

---

## 1. Setup

> **Python 3.11 or 3.12** — PyTorch has no wheels for 3.14 yet. A ready-to-use
> virtual environment is already created at `.venv/` (Python 3.11, all deps
> installed, CUDA build of torch). Just activate it:

```bash
cd ayur
source .venv/bin/activate          # already set up in this build
# To recreate from scratch on another machine:
#   python3.11 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
```

The **NLP half works immediately** (no model/data needed):

```bash
python -m src.nlp.retriever        # demo: ranks herbs for a few sample queries
```

---

## 2. Get a dataset

You said you don't have data yet. This project is wired for the popular
**Indian Medicinal Leaf** datasets whose folder names match the knowledge base:

- **Mendeley "Medicinal Leaf Dataset"** — 30 species (recommended; folder names
  already match `plants.json`). Search: *"Medicinal Leaf Dataset" Mendeley S. Roopashree*.
- **Kaggle — "Indian Medicinal Leaves Image Datasets"** / *"Indian Medicinal Plant Image dataset"*.

Download via the [Hugging Face Hub](https://huggingface.co/datasets) or Kaggle,
then arrange it as **one folder per plant class**:

```
data/raw/
├── Ocimum Tenuiflorum (Tulsi)/   img001.jpg  img002.jpg ...
├── Azadirachta Indica (Neem)/    img001.jpg ...
└── ...
```

Folder names should match the dataset; the knowledge base is indexed by
botanical **and** common names, so small differences still resolve. Validate it:

```bash
python scripts/check_data.py
```

This reports image counts per class and flags any class with **no knowledge-base
entry** (so you can add it to `plants.json`).

---

## 3. Train

```bash
python -m src.train                                   # uses config.yaml defaults
python -m src.train --backbone efficientnet_b0 --epochs 30
python -m src.train --freeze-backbone                 # train only the head (fast, low-data)
```

Backbones: `resnet18` (default), `resnet50`, `mobilenet_v2`, `efficientnet_b0`.
The best checkpoint is saved to `models/best_model.pt`; training curves go to
`outputs/training_curves.png`.

## 4. Evaluate

```bash
python -m src.evaluate
```

Prints per-class precision/recall/F1 and writes
`outputs/confusion_matrix.png`.

## 5. Predict from the command line

```bash
python -m src.predict path/to/leaf.jpg --topk 3
```

Outputs the top predictions with confidence **and** each plant's medicinal
properties, uses, and precautions.

## 6. Run the web app

```bash
streamlit run app.py
```

- **Identify a leaf** — upload an image → species + medicinal properties.
- **Search by symptom** — type *"cough and cold"* → ranked herbs (TF-IDF NLP).
- **Browse herbs** — explore all 30 herbs in the knowledge base.

(The Search and Browse tabs work even before you train a model.)

---

## How the "NLP without an LLM" part works

`src/nlp/retriever.py` builds a text profile for each herb (common names +
Ayurvedic name + properties + uses + description) and vectorises them with a
**TF-IDF** model (`scikit-learn`). A user query is vectorised the same way and
ranked by **cosine similarity** — classic information-retrieval NLP that is
fast, fully offline, explainable, and needs no API keys or GPUs.

To extend the herb database, just add entries to
`data/knowledge_base/plants.json` (keep the same fields). The `folder_aliases`
field is what links a classifier class name to its medicinal data.

---

## Configuration

Everything is in [`config.yaml`](config.yaml) — image size, split ratios,
batch size, backbone, learning rate, epochs, early-stopping patience, and the
knowledge-base path. Any field can be overridden on the command line (see
`python -m src.train --help`).

# BrewView ☕

**NLP-powered cafe recommendation engine for Philadelphia**

UTS 36118 ANLP - AT2 Group 21

## What it does

BrewView analyses ~382K review sentences from Philadelphia cafes and recommends places based on what reviewers actually say - not just star ratings. Ask for "a quiet cafe with strong espresso and good wifi" and it finds matching spots using aspect-based sentiment analysis and dense semantic retrieval.

## How it works

1. **Aspect extraction** - A regex lexicon detects 14 aspects (coffee, tea, food, vibe, service, seating, price, parking, location, hours, internet, alcohol, study, noise) per sentence.
2. **Sentiment scoring** - VADER scores each mention; Bayesian shrinkage (K=5) builds stable cafe profiles.
3. **Dense retrieval** - Sentence embeddings (all-MiniLM-L6-v2) enable semantic search across all 382K sentences.
4. **Hybrid ranking** - Aspect fit (50%) + semantic relevance (35%) + overall sentiment (15%).
5. **RAG summaries** - Optional local LLM (Qwen2.5-0.5B) generates one-line summaries grounded in retrieved quotes.

## Repository structure

```
brewview/
├── app.py                      # Streamlit web app
├── NLP_AT2_2.ipynb             # Full analysis notebook (pipeline, evaluation, figures)
├── data/                       # App data (not in repo - see setup)
│   ├── sentences.parquet       # 382K review sentences with aspects & sentiment
│   ├── dense_embeddings.npy    # Sentence embeddings (all-MiniLM-L6-v2)
│   ├── dense_meta.json         # Embedding metadata
│   ├── shops.csv               # Cafe metadata (name, coords, review count)
│   └── shop_profiles.csv       # Bayesian-shrunk aspect profiles per cafe
├── evaluation/
│   └── aspect_label_sheet.csv  # 210-row human annotation sheet
├── requirements.txt            # Python dependencies
└── README.md
```

## Setup & run

### Prerequisites

- Python 3.10+ (tested on 3.12)
- pip

### 1. Clone the repo

```bash
git clone https://github.com/Aryaman2382/brewview.git
cd brewview
```

### 2. Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** If you have Python 3.9 or older, use `python3.12` (or whichever 3.10+ version you have) instead of `python3` when creating the venv.

### 3. Add the data

The `data/` folder is too large for GitHub (~1 GB). Download it from the [shared Google Drive](https://drive.google.com/drive/folders/1lJCm4VBT7VAQ7L2SxCvUSP2m6HX26ar9?usp=sharing) and extract it into the repo:

```bash
unzip brewview_app_data.zip -d data/
```

Verify you have these five files inside `data/`:
- `sentences.parquet` - 382K review sentences with aspects and sentiment
- `dense_embeddings.npy` - sentence embeddings
- `dense_meta.json` - embedding metadata
- `shops.csv` - cafe metadata (name, coordinates, review count)
- `shop_profiles.csv` - Bayesian-shrunk aspect profiles per cafe

### 4. Run the app

```bash
python3 -m streamlit run app.py
```

The app opens at `http://localhost:8501`. Try queries like:
- *"quiet place to study with good wifi"*
- *"best espresso in center city"*
- *"cozy vibe, good pastries, not too loud"*

## Evaluation

- **Cross-city validation**: Spearman correlation between predicted sentiment and star ratings across Philadelphia (dev), New Orleans & Tampa (test).
- **Austin benchmark**: Aspect extraction accuracy on a labelled subset.
- **Human annotation**: 210 sentences hand-labelled for aspect and polarity correctness (`evaluation/aspect_label_sheet.csv`).
- **LLM comparison**: Wilcoxon signed-rank test comparing Qwen 0.5B vs 1.5B grounding scores.

## Team & contributions

**Aryaman Malik** - Evaluation & reproducibility
- Reviewed and finalised the 210-sentence aspect evaluation set, checking whether each detected aspect and sentiment was correct
- Labelled a 50-sentence overlap set used to measure agreement between annotators
- Set up this repository, wrote the setup guide, and confirmed the app runs from a clean install

**Vedang Gosavi** - NLP pipeline & system design
- Built the aspect extraction and sentiment scoring pipeline
- Designed how the components fit together, from review text to ranked cafe recommendations

**Prachi** - LLM evaluation & references
- Blind-evaluated 60 AI-generated cafe summaries for faithfulness to the source reviews and for specificity
- Independently labelled the 50-sentence overlap set, which gives the inter-annotator agreement score
- Wrote the report's data sources section and compiled the APA references (VADER, Sentence-BERT, RAG, Qwen, Yelp Open Dataset)

**Lavil Saldanha** - Validation & error analysis
- Checked the model's "quiet" and "wifi" predictions against Yelp's own NoiseLevel and WiFi attributes as an external ground truth
- Grouped the model's mistakes into recurring error types (sarcasm, negation, keyword false triggers)
- Wrote the report's challenges, solutions and limitations sections

**Chakshu Karwal** - User study & demo
- Ran a usability study with 5-8 participants on set search tasks, scored with the System Usability Scale (SUS)
- Recorded where users got confused and proposed interface fixes
- Produced the app demo video and wrote the report's outcomes and value section

**Aditya Ahlawat** - Market analysis, poster & editing
- Compared how Google Maps and Yelp recommend cafes with BrewView's review-based approach
- Updated the poster with final results and app screenshots
- Wrote the report's objectives, scope and recommendations sections, and did the final edit for flow

## License

Academic use only - UTS 36118 AT2.

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

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/brewview.git
cd brewview
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Add the data

The `data/` folder is not tracked in git (too large). Download `brewview_app_data.zip` from the [shared Google Drive](https://drive.google.com/drive/folders/1lJCm4VBT7VAQ7L2SxCvUSP2m6HX26ar9?usp=sharing) and extract it:

```bash
unzip brewview_app_data.zip -d data/
```

You should see five files inside `data/`: `sentences.parquet`, `dense_embeddings.npy`, `dense_meta.json`, `shops.csv`, `shop_profiles.csv`.

### 3. Run the app

```bash
streamlit run app.py
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

## Team

| Member            | Contributions |
|-------------------|--------------|
| Aryaman Malik     | Aspect annotation (210-row hand-labelling), inter-annotator agreement (50-row overlap), GitHub repo setup, app deployment testing |
| Vedang Gosavi     | NLP pipeline development, system architecture |
| Prachi            | LLM summary evaluation (60 blind-labelled summaries), inter-annotator agreement (50-row overlap), report writing (data sources, references) |
| Lavil Saldanha    | Ground-truth validation (Yelp WiFi/NoiseLevel cross-check), error analysis & categorisation, report writing (challenges, limitations) |
| Chakshu Karwal    | User study (SUS usability survey, 5-8 participants), UX feedback & recommendations, demo video, report writing (outcomes & value) |
| Aditya Ahlawat    | Market comparison (Google Maps, Yelp vs BrewView), poster update with final results, report writing (objectives, recommendations, final edit) |

## License

Academic use only - UTS 36118 AT2.

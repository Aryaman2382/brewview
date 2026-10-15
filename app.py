"""
BrewView: find a Philadelphia cafe by what reviewers actually say.

Run:  streamlit run app.py
Data: the five files from brewview_app_data.zip in a folder called "data" next
      to this file (sentences.parquet, dense_embeddings.npy, dense_meta.json,
      shops.csv, shop_profiles.csv).

How a search works
  1. The request is parsed into aspects (quiet, wifi, studying...) with the same
     lexicon used to build the cafe profiles.
  2. Dense retrieval: the request is embedded (all-MiniLM-L6-v2) and compared
     with 382,400 review sentences, so matches are by meaning, not keywords.
  3. Cafes are ranked on aspect fit (50%), semantic relevance (35%) and overall
     review sentiment (15%), each converted to a percentile.
  4. Every result shows the review quotes behind it and a downside if one exists.
  5. Optional: a small local LLM (Qwen2.5-0.5B) writes a one-line summary from
     that cafe's quotes only.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent / "data"
LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

st.set_page_config(page_title="BrewView", page_icon="☕", layout="wide")

# ----------------------------------------------------------------- styling
# Cupping-sheet look: white paper, espresso ink, coffee-cherry accent, and a
# light-to-dark roast scale for score bars.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=Work+Sans:wght@400;500;600&family=Caveat:wght@500;700&display=swap');
/* Espresso bar after dark: roast-brown room, crema-coloured type, cherry accent. */
:root {
  --room: #1b1411; --counter: #251b17; --ink: #efe6dc; --ink-2: #b3a597;
  --line: #3a2c26; --cherry: #d4475e; --crema: #e3b27c;
  --roast-1: #7a5236; --roast-2: #b07a4a; --roast-3: #e3b27c; --track: #33261f;
}
/* Font set on the app root and inherited; never on [class*=st-], which would
   also override Streamlit's icon font and show icon names as text. */
.stApp { background: var(--room); color: var(--ink); font-family: 'Work Sans', sans-serif; }
.stApp input, .stApp button, .stApp textarea { font-family: 'Work Sans', sans-serif; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 3.2rem; }
.hero { display: flex; align-items: center; justify-content: space-between; gap: 2rem; margin-bottom: 1.2rem; }
.wordmark { font-family: 'Newsreader', serif; font-weight: 600; font-size: 3.4rem;
            letter-spacing: -0.02em; color: var(--ink); line-height: 1; margin: 0; }
.wordmark span { color: var(--crema); font-style: italic; }
.tagline { color: var(--ink-2); font-size: 1.1rem; margin: 0.6rem 0 0 0; max-width: 34rem; line-height: 1.5; }
.hero-cup { flex: 0 0 auto; width: 190px; }
.steam path { animation: rise 6s ease-in-out infinite; transform-box: fill-box; }
.steam path:nth-child(2) { animation-delay: 2s; } .steam path:nth-child(3) { animation-delay: 4s; }
@keyframes rise { 0% { opacity: 0; transform: translateY(6px); } 30% { opacity: .55; }
                  100% { opacity: 0; transform: translateY(-14px); } }
@media (prefers-reduced-motion: reduce) { .steam path { animation: none; opacity: .35; } }
@media (max-width: 640px) { .hero-cup { width: 110px; } .wordmark { font-size: 2.4rem; } }
.understood { color: var(--ink-2); font-size: 0.95rem; margin: 0.4rem 0 1rem 0; }
.chip { display: inline-block; border: 1px solid var(--line); border-radius: 999px;
        padding: 0.1rem 0.65rem; margin: 0 0.3rem 0.3rem 0; color: var(--ink); font-size: 0.88rem; }
.sheet-head { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.7rem; }
.cup { flex: 0 0 64px; width: 64px; height: 64px; }
.rank { font-family: 'Newsreader', serif; color: var(--cherry); font-size: 1.05rem; font-weight: 600; }
.name { font-family: 'Newsreader', serif; font-size: 1.5rem; font-weight: 600; line-height: 1.15; color: var(--ink); }
.meta { color: var(--ink-2); font-size: 0.88rem; margin: 0.2rem 0 0 0; line-height: 1.45; }
.meta b { color: var(--ink); font-weight: 600; }
.row { display: grid; grid-template-columns: 9.5rem 1fr; gap: 0.75rem; align-items: center;
       margin: 0 0 0.45rem 0; font-size: 0.92rem; }
.row .label { color: var(--ink); font-weight: 500; }
.bar { height: 0.5rem; border-radius: 4px; background: var(--track); position: relative; }
.bar i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 4px; }
.verdict { color: var(--ink-2); font-size: 0.85rem; margin-top: 0.2rem; }
.thin { color: var(--ink-2); font-style: italic; }
.notes { margin: 0.9rem 0 0.2rem 0; padding-left: 0.9rem; border-left: 2px solid var(--roast-2); }
.quote { font-family: 'Newsreader', serif; font-size: 1.1rem; line-height: 1.5; color: var(--ink); margin: 0 0 0.5rem 0; }
.watch { margin: 0.7rem 0 0 0; font-size: 0.9rem; color: var(--cherry); }
.watch .quote { font-size: 1rem; color: var(--ink-2); margin-top: 0.15rem; }
.ai { margin: 0.6rem 0 0 0; font-size: 0.95rem; color: var(--ink); }
.ai small { color: var(--ink-2); display: block; margin-top: 0.15rem; }

/* A faint coffee-ring stain on the "table" (the page background). */
.stApp {
  background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='420' height='420'><circle cx='210' cy='210' r='150' fill='none' stroke='%23e3b27c' stroke-opacity='0.05' stroke-width='9'/><circle cx='214' cy='206' r='143' fill='none' stroke='%23e3b27c' stroke-opacity='0.035' stroke-width='3'/></svg>");
  background-repeat: no-repeat; background-position: right -140px top 520px; background-attachment: fixed;
}
/* Pendant-light glow behind the hero cup. */
.hero-glow { position: relative; flex: 0 0 auto; }
.hero-glow::before { content: ""; position: absolute; inset: -30%; border-radius: 50%;
  background: radial-gradient(circle, rgba(227,178,124,.22) 0%, rgba(227,178,124,0) 65%); }
.hero-glow svg { position: relative; }
/* Chalkboard: the one bold element. Slate-green board in a wooden frame. */
.st-key-board { background: #24302a; border: 10px solid #5b3f2b; border-radius: 6px;
  padding: 0.9rem 1.1rem 1.1rem 1.1rem; box-shadow: inset 0 0 40px rgba(0,0,0,.45); margin-bottom: 0.6rem; }
.board-title { font-family: 'Caveat', 'Segoe Print', 'Bradley Hand', cursive; font-size: 2rem; font-weight: 700; color: #f3efe6;
  line-height: 1; margin: 0 0 0.2rem 0; }
.board-sub { font-family: 'Caveat', 'Segoe Print', 'Bradley Hand', cursive; font-size: 1.25rem; color: #c9d1c7; margin: 0 0 0.4rem 0; }
.st-key-board button { background: transparent; border: 1px dashed rgba(243,239,230,.35);
  color: #f3efe6; }
.st-key-board button p { font-family: 'Caveat', 'Segoe Print', 'Bradley Hand', cursive; font-size: 1.35rem; line-height: 1.15; }
.st-key-board button:hover { border-color: #e3b27c; color: #e3b27c; background: rgba(227,178,124,.06); }
.empty { border-left: 3px solid var(--crema); padding: 0.6rem 0 0.6rem 1rem; color: var(--ink-2);
  font-size: 1.02rem; max-width: 40rem; margin-top: 0.4rem; }
/* Bean ratings. */
.beans { display: inline-flex; gap: 2px; vertical-align: -2px; margin-right: 0.35rem; }
/* Pour-in: result cups swirl into place once, one after another. */
@keyframes pour { from { opacity: 0; transform: rotate(-160deg) scale(.55); } to { opacity: 1; transform: none; } }
.sheet-head .cup { animation: pour .8s cubic-bezier(.2,.8,.2,1) both; }
@media (prefers-reduced-motion: reduce) { .sheet-head .cup { animation: none; } }
/* Copyright line at the foot of every page. */
.footer { margin-top: 3rem; padding: 1.1rem 0 0.4rem 0; border-top: 1px solid var(--line);
  color: var(--ink-2); font-size: 0.85rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- lexicon
# Same lexicon as the analysis notebook (Cell 1), so a request is understood
# the same way the cafe profiles were built.
ASPECTS = {
    "coffee": [r"coffee", r"espresso", r"latte", r"lattes", r"cappuccino", r"cortado",
               r"americano", r"macchiato", r"mocha", r"cold brew", r"drip", r"pour[- ]?over",
               r"flat white", r"brew", r"beans?", r"roast(?:ed|s|er|ery)?", r"barista"],
    "tea": [r"tea", r"teas", r"chai", r"matcha", r"kombucha", r"herbal"],
    "food": [r"food", r"pastr(?:y|ies)", r"croissants?", r"muffins?", r"bagels?",
             r"sandwich(?:es)?", r"breakfast", r"brunch", r"lunch", r"tacos?", r"cakes?",
             r"cookies?", r"donuts?", r"doughnuts?", r"scones?", r"kolaches?", r"toast",
             r"waffles?", r"snacks?", r"menu"],
    "vibe": [r"vibe", r"vibes", r"atmosphere", r"ambien(?:ce|t)", r"ambiance", r"decor",
             r"aesthetic", r"cozy", r"cosy", r"charming", r"hip", r"interior", r"music",
             r"feel(?:s)? like", r"homey"],
    "service": [r"service", r"staff", r"baristas", r"employees?", r"friendly", r"rude",
                r"unfriendly", r"welcoming", r"customer service", r"cashier", r"owner",
                r"server", r"helpful"],
    "seating": [r"seat(?:s|ing)?", r"chairs?", r"tables?", r"couch(?:es)?", r"sofas?",
                r"booths?", r"patio", r"outdoor", r"spacious", r"cramped", r"comfy", r"comfortable"],
    "price": [r"price[sd]?", r"pricey", r"pricy", r"expensive", r"cheap", r"overpriced",
              r"afford(?:able)?", r"value", r"cost(?:s|ly)?", r"worth", r"\$\d+"],
    "parking": [r"parking", r"parked", r"parking lot", r"garage"],
    "location": [r"location", r"located", r"neighbou?rhood", r"downtown", r"close to",
                 r"nearby", r"convenient(?:ly)?", r"walking distance"],
    "hours": [r"hours", r"open(?:s)? (?:late|early)", r"close[sd]? (?:early|late)",
              r"24 hours", r"late night", r"opening", r"closing time"],
    "internet": [r"wi-?fi", r"internet", r"wireless", r"connection", r"password"],
    "alcohol": [r"beers?", r"wines?", r"cocktails?", r"alcohol", r"booze", r"mimosas?", r"happy hour"],
    "study": [r"stud(?:y|ying|ied)", r"homework", r"laptops?", r"work(?:ing)? (?:from|on|remotely)",
              r"get (?:some )?work done", r"outlets?", r"plugs?", r"remote work",
              r"productive", r"read(?:ing)? a book", r"exam"],
    "noise": [r"quiet", r"calm", r"peaceful", r"loud", r"noisy", r"noise", r"crowded",
              r"packed", r"busy", r"chill", r"relaxed", r"relaxing", r"blaring", r"hectic",
              r"tranquil", r"serene"],
}
QUIET_CUES = [r"quiet", r"calm", r"peaceful", r"chill", r"relaxed", r"relaxing",
              r"tranquil", r"serene", r"low[- ]key"]
LOUD_CUES = [r"loud", r"noisy", r"noise", r"crowded", r"packed", r"busy", r"blaring",
             r"hectic", r"chaotic"]
ASPECT_RE = {a: re.compile(r"\b(?:" + "|".join(p) + r")\b", re.I) for a, p in ASPECTS.items()}
QUIET_RE = re.compile(r"\b(?:" + "|".join(QUIET_CUES) + r")\b", re.I)
LOUD_RE = re.compile(r"\b(?:" + "|".join(LOUD_CUES) + r")\b", re.I)
NEGATION_RE = re.compile(
    r"\b(?:not|never|no|isn't|wasn't|aren't|weren't|isnt|wasnt|doesn't|doesnt|don't|dont|"
    r"didn't|didnt|won't|wont|can't|cant|cannot|hardly|rarely|without)\b"
    r"(?:\s+(?!and\b|or\b|but\b)\w+){0,3}\s*$", re.I)
QUERY_HINTS = {
    "study": ["study", "studying", "work", "working", "laptop", "productive", "homework", "focus", "read"],
    "internet": ["wifi", "wi-fi", "internet"],
    "seating": ["seat", "seats", "seating", "comfy", "comfortable", "couch", "sofa", "space", "spacious", "outdoor", "patio"],
    "vibe": ["vibe", "atmosphere", "cozy", "cosy", "aesthetic", "ambience", "ambiance", "charming"],
    "price": ["cheap", "affordable", "budget", "inexpensive", "value"],
    "food": ["food", "pastry", "pastries", "breakfast", "brunch", "lunch", "sandwich", "snack", "bakery"],
    "coffee": ["coffee", "espresso", "latte", "cappuccino", "cold brew", "pour over", "flat white"],
    "tea": ["tea", "matcha", "chai", "boba", "bubble tea"],
    "service": ["friendly", "service", "staff", "barista"],
    "hours": ["late", "early", "open late", "24 hours"],
    "parking": ["parking", "park"],
    "alcohol": ["beer", "wine", "cocktail", "alcohol"],
}
LIVELY_WORDS = ["lively", "buzzy", "bustling", "busy", "energetic", "social", "loud"]

# What users see instead of internal aspect names.
LABELS = {"coffee": "Coffee", "tea": "Tea", "food": "Food", "vibe": "Atmosphere",
          "service": "Service", "seating": "Seating", "price": "Value for money", "parking": "Parking",
          "location": "Location", "hours": "Opening hours", "internet": "Wifi",
          "alcohol": "Drinks (alcohol)", "study": "Good for studying", "noise": "Noise level"}
# The chalkboard: a mood on the board, and the request it sends.
BOARD = [("Exam week cram session", "Quiet cafe for studying with good wifi"),
         ("Brunch with the crew", "Lively place for brunch with friends"),
         ("First date nerves", "Great spot for a first date"),
         ("Board games and a latte", "Cafe with board games"),
         ("Bubble tea on a budget", "Cheap bubble tea"),
         ("Laptop in the sunshine", "Outdoor seating to work on my laptop")]
# Approximate centre points for the "Neighbourhood" filter (latitude, longitude).
# South Philly is a large district, so its label names the point we measure from.
PLACES = {"Center City": (39.9526, -75.1635),     # City Hall
          "University City": (39.9530, -75.1920), # Penn and Drexel
          "Old City": (39.9510, -75.1440),        # around 2nd and Market
          "Fishtown": (39.9690, -75.1340),        # Frankford and Girard
          "South Philly (East Passyunk)": (39.9290, -75.1640)}


def parse_query(text):
    low = f" {text.lower()} "
    found = {a for a, rx in ASPECT_RE.items() if a != "noise" and rx.search(text)}
    found |= {a for a, words in QUERY_HINTS.items() if any(f" {w} " in low for w in words)}
    quiet = sum(1 for m in QUIET_RE.finditer(text) if not NEGATION_RE.search(text[:m.start()]))
    quiet += sum(1 for m in LOUD_RE.finditer(text) if NEGATION_RE.search(text[:m.start()]))
    lively = sum(1 for w in LIVELY_WORDS
                 if f" {w} " in low and not NEGATION_RE.search(low[:low.find(f" {w} ")]))
    direction = 0
    if quiet or lively:
        direction = 1 if quiet >= lively else -1
        found.add("noise")
    return sorted(found), direction


# ----------------------------------------------------------------- engine
W_ASPECT, W_SEMANTIC, W_QUALITY = 0.5, 0.35, 0.15
TOP_SENTS, CANDIDATE_SENTS = 3, 3000
EVIDENCE_PER_CAFE, EVIDENCE_WORDS = 3, (5, 40)


def haversine_km(lat, lon, lat0, lon0):
    """Straight-line distance in km between cafes and one point."""
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    a = np.sin((lat - lat0) / 2) ** 2 + np.cos(lat) * np.cos(lat0) * np.sin((lon - lon0) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


class DenseRetriever:
    def __init__(self, emb_path, model_name):
        from sentence_transformers import SentenceTransformer
        self.E = np.load(emb_path).astype(np.float32)
        self.model = SentenceTransformer(model_name, device="cpu")

    def similarities(self, text):
        q = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return self.E @ q


class Recommender:
    def __init__(self, sents, shops_df, profiles_df, retriever):
        self.sents, self.retriever = sents, retriever
        self.shops = shops_df.set_index("shop_id")
        self.score = profiles_df.pivot(index="shop_id", columns="aspect", values="score")
        self.mentions = profiles_df.pivot(index="shop_id", columns="aspect", values="n_mentions")
        self.claimable = profiles_df.pivot(index="shop_id", columns="aspect", values="claimable")
        self.sent_shop = sents["shop_id"].to_numpy()
        self.coverage = profiles_df.groupby("aspect")["coverage"].mean().to_dict()
        n_words = sents["sentence"].str.split().str.len()
        self.quotable = (n_words.between(*EVIDENCE_WORDS)
                         & ~sents["sentence"].str.rstrip().str.endswith("?")
                         & ~sents["sentence"].str.contains(r"\.\.\.\s*$")).to_numpy()

    def recommend(self, text, k=5, hide_possibly_closed=True, near=None, radius_km=None):
        aspects, direction = parse_query(text)
        sims = self.retriever.similarities(text)
        top = np.argpartition(-sims, CANDIDATE_SENTS)[:CANDIDATE_SENTS]
        cand = pd.DataFrame({"shop_id": self.sent_shop[top], "sim": sims[top]})
        cand = cand[cand["sim"] > 0]
        relevance = (cand.sort_values("sim", ascending=False).groupby("shop_id")["sim"]
                         .apply(lambda s: s.head(TOP_SENTS).sum() / TOP_SENTS))
        df = self.shops.copy()
        if hide_possibly_closed:
            df = df[~df["possibly_closed"]]
        df["relevance"] = relevance.reindex(df.index).fillna(0.0)
        fit = pd.DataFrame({a: self.score[a].reindex(df.index) * (-1 if a == "noise" and direction == -1 else 1)
                            for a in aspects})
        pct = lambda s: s.rank(pct=True)
        w_aspect = W_ASPECT if aspects else 0.0
        df["final"] = ((w_aspect * pct(fit.mean(axis=1)) if aspects else 0)
                       + (W_SEMANTIC + W_ASPECT - w_aspect) * pct(df["relevance"])
                       + W_QUALITY * pct(df["overall_sent"]))
        # Filters are applied after scoring, so a cafe's score does not depend on which
        # other cafes happen to be filtered out.
        keep = pd.Series(True, index=df.index)
        df["distance_km"] = np.nan
        if near is not None and radius_km:
            df["distance_km"] = haversine_km(df["latitude"], df["longitude"], *near)
            keep &= df["distance_km"] <= radius_km
        df = df[keep].sort_values(["final", "n_reviews", "shop"], ascending=[False, False, True])
        return {"query": text, "aspects": aspects, "direction": direction,
                "results": [self._explain(sid, row, aspects, direction, sims)
                            for sid, row in df.head(k).iterrows()]}

    def _aspect_score(self, row, a, direction):
        tags = row["aspects"].split(",") if row["aspects"] else []
        if a not in tags:
            return None
        sc = float(row["aspect_scores"].split(",")[tags.index(a)])
        return -sc if (a == "noise" and direction == -1) else sc

    def _explain(self, shop_id, row, aspects, direction, sims):
        idx = np.flatnonzero((self.sent_shop == shop_id) & self.quotable)
        pool = self.sents.iloc[idx].assign(sim=sims[idx])
        chosen = []
        for a in sorted(aspects, key=lambda x: self.coverage.get(x, 1)):
            if len(chosen) == EVIDENCE_PER_CAFE or pool.empty:
                break
            sc = pool.apply(lambda r: self._aspect_score(r, a, direction), axis=1)
            good = pool[(sc.fillna(-9) > 0.3) & ~pool.index.isin(chosen)]
            if len(good):
                chosen.append(good.assign(asc=sc[good.index]).sort_values(["sim", "asc"], ascending=False).index[0])
        for i in pool[pool["sim"] > 0].sort_values("sim", ascending=False).index:
            if len(chosen) == EVIDENCE_PER_CAFE:
                break
            if i not in chosen:
                chosen.append(i)
        caveat, best = None, None
        for a in aspects:
            for i, r in pool.iterrows():
                sc = self._aspect_score(r, a, direction)
                if sc is not None and sc < -0.3:
                    key = (r["stars"] > 3, self.coverage.get(a, 1), r["stars"], sc)
                    if best is None or key < best:
                        best, caveat = key, r["sentence"]
        return {
            "shop": row["shop"], "address": row["address"], "stars": float(row["mean_stars"]),
            "n_reviews": int(row["n_reviews"]), "last_review": str(row["last_review"])[:10],
            "latitude": float(row["latitude"]), "longitude": float(row["longitude"]),
            "distance_km": None if pd.isna(row["distance_km"]) else float(row["distance_km"]),
            "aspects": {a: {"score": float(self.score.at[shop_id, a]),
                            "mentions": int(self.mentions.at[shop_id, a]),
                            "enough_evidence": bool(self.claimable.at[shop_id, a])} for a in aspects},
            "evidence": [self.sents.at[i, "sentence"] for i in chosen],
            "caveat": caveat,
        }


@st.cache_resource(show_spinner="Loading 382,400 review sentences and the search model...")
def load_engine():
    sents = pd.read_parquet(DATA / "sentences.parquet")
    shops = pd.read_csv(DATA / "shops.csv")
    shops["possibly_closed"] = shops["possibly_closed"].astype(str).str.lower().eq("true")
    profiles = pd.read_csv(DATA / "shop_profiles.csv")
    profiles["claimable"] = profiles["claimable"].astype(str).str.lower().eq("true")
    meta = json.loads((DATA / "dense_meta.json").read_text())
    retriever = DenseRetriever(DATA / "dense_embeddings.npy", meta["model"])
    return Recommender(sents, shops, profiles, retriever), len(shops), len(sents)


@st.cache_resource(show_spinner="Loading the summary model (first time only, about 1 GB)...")
def load_llm():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LLM_NAME)
    llm = AutoModelForCausalLM.from_pretrained(LLM_NAME)
    llm.eval()
    return tok, llm


def summarise(query, r):
    """One sentence from this cafe's quotes only (constrained generation)."""
    import torch
    tok, llm = load_llm()
    reviews = "\n".join(f'- "{q}"' for q in r["evidence"])
    messages = [
        {"role": "system", "content": "You are BrewView, a cafe recommendation assistant. Use ONLY "
                                      "the review evidence provided. Never invent cafes, facts or quotes."},
        {"role": "user", "content": f'Request: "{query}"\nReviews of {r["shop"]}:\n{reviews}\n\n'
                                    f"In one sentence of at most 25 words, say why {r['shop']} fits the "
                                    "request. Use only facts from these reviews. Do not use quotation marks."}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = llm.generate(**inputs, max_new_tokens=60, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return re.split(r"(?<=[.!?])\s", text)[0]



# ----------------------------------------------------------------- illustrations
# Every cafe gets its own top-down cup with latte art (heart, tulip or rosetta),
# chosen and rotated from a hash of its name, so the same cafe always gets the
# same cup. Pure SVG: no image files, works offline.
import zlib

MILK, CREMA_IN, CREMA_OUT, CERAMIC, RIM = "#f6ecdf", "#c98d55", "#7e4b2a", "#e7ddd1", "#f4ece2"


def latte_art(kind):
    if kind == "heart":
        return (f'<path d="M50 67 C31 54 33 35 45.5 38 C48.5 38.8 50 41.5 50 44 C50 41.5 51.5 38.8 54.5 38 '
                f'C67 35 69 54 50 67 Z" fill="{MILK}"/>'
                f'<path d="M50 30 L50 70" stroke="{MILK}" stroke-width="1.6" stroke-linecap="round"/>')
    if kind == "tulip":
        blobs = "".join(f'<ellipse cx="50" cy="{cy}" rx="{rx}" ry="{ry}" fill="{MILK}"/>'
                        for cy, rx, ry in [(60, 15, 8), (48.5, 11.5, 6.4), (39, 8, 4.8)])
        return blobs + f'<path d="M50 31 L50 68" stroke="{CREMA_IN}" stroke-width="1.4"/>'
    leaves = "".join(f'<path d="M{50 - w:.1f} {y:.1f} Q50 {y + 6.5:.1f} {50 + w:.1f} {y:.1f}" fill="none" '
                     f'stroke="{MILK}" stroke-width="2.6" stroke-linecap="round"/>'
                     for i in range(7) for y, w in [(34 + i * 4.6, 5 + i * 2.1)])
    return leaves + (f'<circle cx="50" cy="31" r="2.6" fill="{MILK}"/>'
                     f'<path d="M50 31 L50 69" stroke="{MILK}" stroke-width="1.5" stroke-linecap="round"/>')


def cup_svg(seed_text, size=64, css_class="cup", label=None, steam=False):
    h = zlib.crc32(seed_text.encode())
    kind = ["heart", "tulip", "rosetta"][h % 3]
    tilt = (h >> 3) % 50 - 25
    handle = (h >> 9) % 360
    gid = f"g{h:x}"
    steam_svg = ""
    if steam:
        steam_svg = ('<g class="steam" fill="none" stroke="#efe6dc" stroke-width="1.6" stroke-linecap="round">'
                     '<path d="M40 30 C34 22 46 16 40 6"/><path d="M50 28 C44 20 56 14 50 3"/>'
                     '<path d="M60 30 C54 22 66 16 60 6"/></g>')
    title = f"<title>{label}</title>" if label else ""
    return (f'<svg class="{css_class}" viewBox="0 0 100 100" width="{size}" height="{size}" role="img" '
            f'aria-label="{label or "cup of coffee"}">{title}'
            f'<defs><radialGradient id="{gid}" cx="50%" cy="45%" r="55%">'
            f'<stop offset="0" stop-color="{CREMA_IN}"/><stop offset="1" stop-color="{CREMA_OUT}"/></radialGradient></defs>'
            f'<circle cx="50" cy="50" r="48" fill="{CERAMIC}" opacity=".16"/>'
            f'<g transform="rotate({handle} 50 50)"><rect x="80" y="44" width="15" height="12" rx="6" fill="{RIM}"/></g>'
            f'<circle cx="50" cy="50" r="36" fill="{RIM}"/>'
            f'<circle cx="50" cy="50" r="31" fill="url(#{gid})"/>'
            f'<g transform="rotate({tilt} 50 50)">{latte_art(kind)}</g>{steam_svg}</svg>')

def beans_svg(stars):
    """Five coffee beans, filled to the cafe's average star rating."""
    out = []
    for i in range(5):
        fill = max(0.0, min(1.0, stars - i))
        cid = f"b{i}{int(stars * 100)}"
        out.append(f'<svg width="14" height="16" viewBox="0 0 14 16" aria-hidden="true">'
                   f'<defs><clipPath id="{cid}"><rect width="{14 * fill:.1f}" height="16"/></clipPath></defs>'
                   f'<ellipse cx="7" cy="8" rx="5.5" ry="7" fill="#33261f"/>'
                   f'<ellipse cx="7" cy="8" rx="5.5" ry="7" fill="#e3b27c" clip-path="url(#{cid})"/>'
                   f'<path d="M7 1.6 C4.6 5.5 9.4 10.5 7 14.4" stroke="#1b1411" stroke-width="1.2" fill="none"/></svg>')
    return f'<span class="beans" role="img" aria-label="{stars:.1f} out of 5">{"".join(out)}</span>'


LOADING = ["Grinding through 382,400 review sentences...", "Pulling the shot...",
           "Steaming the milk...", "Asking the regulars...", "Reading the reviews so you don't have to..."]

# ----------------------------------------------------------------- display helpers
def verdict(aspect, score, direction):
    """Turn a score in [-1, 1] into plain words."""
    if aspect == "noise":
        if score > 0.25:
            return "Reviewers call it quiet"
        if score < -0.25:
            return "Reviewers call it loud and busy"
        return "Reviewers are split on noise"
    if score > 0.55:
        return "Reviewers love it"
    if score > 0.3:
        return "Mostly positive"
    if score > 0.05:
        return "Mixed"
    return "Mostly negative"


def bar(aspect, score, direction):
    """Fill shows how well the cafe matches the request (for 'lively', loud is good)."""
    fit = -score if (aspect == "noise" and direction == -1) else score
    width = max(4, min(100, (fit + 1) / 2 * 100))
    color = "var(--roast-3)" if fit > 0.45 else "var(--roast-2)" if fit > 0.15 else "var(--roast-1)"  # brighter = better
    return f'<div class="bar"><i style="width:{width:.0f}%;background:{color}"></i></div>'


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def render_sheet(rank, r, direction, near_label=None):
    stars = f"{r['stars']:.1f}"
    dist = (f'<br>{r["distance_km"]:.1f} km from {esc(near_label)}'
            if near_label and r["distance_km"] is not None else "")
    cup = cup_svg(r["shop"], label=f"Latte art for {esc(r['shop'])}").replace(
        "<svg ", f'<svg style="animation-delay:{(rank - 1) * 0.12:.2f}s" ', 1)
    head = (f'<div class="sheet-head">{cup}'
            f'<div><div class="rank">No. {rank}</div><div class="name">{esc(r["shop"])}</div>'
            f'<p class="meta">{beans_svg(r["stars"])}<b>{stars}</b> from {r["n_reviews"]} reviews, '
            f'last reviewed {r["last_review"]}<br>{esc(r["address"])}{dist}</p></div></div>')
    rows = ""
    for a, v in r["aspects"].items():
        if v["enough_evidence"]:
            detail = (f'{bar(a, v["score"], direction)}'
                      f'<div class="verdict">{verdict(a, v["score"], direction)}, '
                      f'from {v["mentions"]} reviews</div>')
        else:
            detail = (f'<div class="thin">Not enough reviews mention this '
                      f'({v["mentions"]} so far), so we make no claim</div>')
        rows += f'<div class="row"><div class="label">{LABELS[a]}</div><div>{detail}</div></div>'
    notes = "".join(f'<p class="quote">"{esc(q)}"</p>' for q in r["evidence"])
    notes = f'<div class="notes">{notes}</div>' if notes else ""
    watch = (f'<div class="watch">Worth knowing<p class="quote">"{esc(r["caveat"])}"</p></div>'
             if r["caveat"] else "")
    st.markdown(head + rows + notes + watch, unsafe_allow_html=True)


def render_map(results, near_label=None):
    import pydeck as pdk
    pts = pd.DataFrame([{"rank": str(i), "name": r["shop"], "lat": r["latitude"], "lon": r["longitude"]}
                        for i, r in enumerate(results, 1)])
    view = pdk.ViewState(latitude=pts["lat"].mean(), longitude=pts["lon"].mean(), zoom=12)
    layers = []
    if near_label:
        lat0, lon0 = PLACES[near_label]
        hub = pd.DataFrame([{"rank": "", "name": near_label, "lat": lat0, "lon": lon0}])
        layers += [
            pdk.Layer("ScatterplotLayer", hub, get_position="[lon, lat]", get_radius=70,
                      get_fill_color=[239, 230, 220, 230], pickable=True),
            pdk.Layer("TextLayer", hub, get_position="[lon, lat]", get_text="name", get_size=13,
                      get_color=[239, 230, 220], get_pixel_offset=[0, -18]),
        ]
    layers += [
        pdk.Layer("ScatterplotLayer", pts, get_position="[lon, lat]", get_radius=110,
                  get_fill_color=[212, 71, 94, 235], pickable=True),
        pdk.Layer("TextLayer", pts, get_position="[lon, lat]", get_text="rank", get_size=14,
                  get_color=[255, 255, 255], get_alignment_baseline="'center'"),
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style=pdk.map_styles.CARTO_DARK,
                             tooltip={"text": "{rank}. {name}"}), height=420)


# ----------------------------------------------------------------- page
st.markdown('<div class="hero"><div><div class="wordmark">Brew<span>View</span></div>'
            '<p class="tagline">Find a Philadelphia cafe by what reviewers actually say, '
            'not just its star rating.</p></div>'
            + '<div class="hero-glow">' + cup_svg("BrewView hero", size=190, css_class="hero-cup", steam=True)
            + '</div></div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Filters")
    near_choice = st.selectbox("Neighbourhood", ["Anywhere"] + list(PLACES),
                               help="Distance is measured in a straight line from the centre of "
                                    "the neighbourhood you pick.")
    radius = st.slider("Within (km)", 1.0, 5.0, 2.0, 0.5, disabled=near_choice == "Anywhere")
    k = st.slider("Number of cafes", 3, 10, 5)
    hide_closed = st.toggle("Hide cafes with no reviews since early 2020", value=True,
                            help="Our reviews end in early 2022. Cafes with no reviews since early "
                                 "2020 may have closed, so we hide them. Switch this off to see every cafe.")
    st.markdown("### How BrewView ranks cafes")
    st.markdown(
        "1. **What you asked for.** We pick out features like quiet, wifi or studying, and score each "
        "cafe on them from its reviews.\n"
        "2. **What you meant.** Your whole request is compared by meaning with 382,400 review "
        "sentences.\n"
        "3. **Overall impression.** The average tone of the cafe's reviews.\n\n"
        "We only claim a feature when at least 3 reviews mention it. Quotes are always real "
        "reviews, and the downside shown is the most critical relevant comment we found.")
    st.caption("Data: Yelp Open Dataset (academic use), 594 Philadelphia cafes, reviews 2005 to 2022.")

if "query" not in st.session_state:
    st.session_state.query = ""
if "query_box" not in st.session_state:
    st.session_state.query_box = ""
if "summaries" not in st.session_state:
    st.session_state.summaries = {}


# The search box is bound to session state by its key and is never given a `value`.
# Passing value=query made Streamlit treat it as a new widget whenever the query
# changed, so every other search returned the previous request. Callbacks run before
# the page is redrawn, so the results always match the request just made.
def run_search():
    st.session_state.query = st.session_state.query_box.strip()


def use_example(text):
    st.session_state.query = st.session_state.query_box = text


with st.form("search", border=False):
    c1, c2 = st.columns([5, 1])
    c1.text_input("What are you looking for?", key="query_box",
                  placeholder="e.g. a quiet cafe with good wifi where I can study",
                  label_visibility="collapsed")
    c2.form_submit_button("Find cafes", type="primary", use_container_width=True, on_click=run_search)

with st.container(key="board"):
    st.markdown('<div class="board-title">On the board today</div>'
                '<div class="board-sub">Not sure what to type? Pick a mood.</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, (label, ex) in enumerate(BOARD):
        cols[i % 3].button(label, key=f"ex{i}", use_container_width=True, help=ex,
                           on_click=use_example, args=(ex,))

def footer():
    """Copyright line. Called before every st.stop() and at the end, so it shows on every page."""
    st.markdown('<div class="footer">© 2026 BrewView by Group 21. All rights reserved.</div>',
                unsafe_allow_html=True)


query = st.session_state.query
if not query:
    st.markdown('<div class="empty">Describe the cafe you want in your own words: the mood, what you plan '
                'to do there, what matters to you. BrewView reads the reviews so you don\'t have to.</div>',
                unsafe_allow_html=True)
    footer()
    st.stop()

engine, n_cafes, n_sents = load_engine()
with st.spinner(LOADING[sum(map(ord, query)) % len(LOADING)]):
    near_label = None if near_choice == "Anywhere" else near_choice
    out = engine.recommend(query, k=k, hide_possibly_closed=hide_closed,
                           near=PLACES.get(near_label), radius_km=radius if near_label else None)

if out["aspects"]:
    noise_note = {1: " (you want it quiet)", -1: " (you want it lively)", 0: ""}[out["direction"]]
    chips = "".join(f'<span class="chip">{LABELS[a]}{noise_note if a == "noise" else ""}</span>'
                    for a in out["aspects"])
    st.markdown(f'<p class="understood">We scored cafes on: {chips}</p>', unsafe_allow_html=True)
else:
    st.markdown('<p class="understood">No specific features recognised, so we matched the meaning of '
                f'your request against {n_sents:,} review sentences.</p>', unsafe_allow_html=True)

if near_label:
    st.markdown(f'<p class="understood">Filtered to cafes within {radius:g} km of {near_label}.</p>',
                unsafe_allow_html=True)

if not out["results"]:
    st.warning("No cafes matched. Try a larger distance or another neighbourhood in the sidebar, "
               "or switch off 'Hide cafes with no reviews since early 2020'.")
    footer()
    st.stop()

left, right = st.columns([3, 2], gap="large")
with right:
    render_map(out["results"], near_label)
    st.caption("Numbers match the ranking on the left."
               + (f" The light dot marks {near_label}." if near_label else ""))
with left:
    for rank, r in enumerate(out["results"], 1):
        with st.container(border=True):
            render_sheet(rank, r, out["direction"], near_label)
            key = (query, r["shop"])
            if key in st.session_state.summaries:
                st.markdown(f'<div class="ai">{esc(st.session_state.summaries[key])}'
                            f'<small>Written by a small local AI model from the quotes above only.'
                            f'</small></div>', unsafe_allow_html=True)
            elif r["evidence"] and st.button("Summarise with AI (about 20 seconds)", key=f"ai{rank}"):
                with st.spinner("Writing a summary from these quotes..."):
                    st.session_state.summaries[key] = summarise(query, r)
                st.rerun()

footer()

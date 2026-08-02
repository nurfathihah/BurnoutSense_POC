"""
BurnoutSense - multilingual work-burnout expression classifier
UiTM MSc Data Science (CDCS779)

Run:   streamlit run app.py
Needs: artifacts/{vectorizer,classifier,label_encoder}.joblib + config.json
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS = Path("artifacts")

# Results weaker than this are shown as uncertain rather than as an answer.
# With three categories random guessing sits at 0.33, so anything under ~0.40 is
# barely a decision. 0.50 means the top category holds at least half the confidence.
LOW_CONFIDENCE = 0.50

DIMENSION_META = {
    "emotional_exhaustion": {
        "label": "Emotional Exhaustion", "short": "Exhaustion", "colour": "#D97A1F",
        "gloss": "Wording about depletion, fatigue, and being drained by work."},
    "depersonalisation": {
        "label": "Depersonalisation", "short": "Detachment", "colour": "#1F7A8C",
        "gloss": "Wording about detachment, cynicism, or distance from work and people."},
    "depersonalization": {
        "label": "Depersonalisation", "short": "Detachment", "colour": "#1F7A8C",
        "gloss": "Wording about detachment, cynicism, or distance from work and people."},
    "reduced_accomplishment": {
        "label": "Reduced Personal Accomplishment", "short": "Ineffectiveness",
        "colour": "#6B4E9E",
        "gloss": "Wording about ineffectiveness, self-doubt, or lack of achievement."},
    "non_burnout": {
        "label": "No burnout wording found", "short": "None", "colour": "#5B6B78",
        "gloss": "Nothing characteristic of the three MBI dimensions was found."},
}


def meta(raw):
    key = str(raw).strip().lower()
    return DIMENSION_META.get(key, {"label": str(raw), "short": str(raw),
                                    "colour": "#5B6B78", "gloss": ""})


EXAMPLES = [
    ("English", "I feel completely drained by this job, every single day."),
    ("Malay", "Penat sangat kerja, dah tak larat nak teruskan."),
    ("Manglish", "Dah tak kisah dah pasal kerja ni, whatever lah."),
    ("Neutral", "Hari ni cuaca okay, lepas kerja nak pergi makan."),
]

st.set_page_config(page_title="BurnoutSense", page_icon="◑", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
  :root { --ink:#14232E; --muted:#5B6B78; --line:#DCE3E8; --canvas:#EDF1F4; --surface:#FFFFFF; }
  .stApp { background:
      radial-gradient(1100px 380px at 8% -8%, #DCE9EE 0%, rgba(220,233,238,0) 62%),
      radial-gradient(900px 340px at 96% 0%, #EFE6F5 0%, rgba(239,230,245,0) 58%),
      var(--canvas); }
  html, body, [class*="css"] { font-family:'Inter',sans-serif; color:var(--ink); }
  h1,h2,h3 { font-family:'Fraunces',Georgia,serif !important; font-weight:600 !important;
             letter-spacing:-0.02em; color:var(--ink); }
  .masthead { padding:.4rem 0 1.1rem 0; }
  .masthead h1 { font-size:3rem; margin:0; line-height:1.02;
    background:linear-gradient(96deg,#14232E 0%,#1F7A8C 46%,#6B4E9E 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
  .masthead .sub { font-family:'JetBrains Mono',monospace; font-size:.7rem;
    letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin-top:.45rem; }
  .notice { background:var(--surface); border:1px solid var(--line);
    border-left:4px solid #1F7A8C; border-radius:10px; padding:.85rem 1.1rem;
    font-size:.87rem; color:#33454F; margin:.6rem 0 1.4rem 0;
    box-shadow:0 1px 2px rgba(20,35,46,.05); }
  .card { background:var(--surface); border:1px solid var(--line); border-radius:14px;
    padding:1.15rem 1.3rem; margin-bottom:.9rem;
    box-shadow:0 1px 2px rgba(20,35,46,.05); animation:rise .34s ease both;
    transition:box-shadow .18s ease, transform .18s ease; }
  .card:hover { box-shadow:0 6px 18px rgba(20,35,46,.09); transform:translateY(-1px); }
  @keyframes rise { from{opacity:0;transform:translateY(7px);} to{opacity:1;transform:none;} }
  .quoted { font-size:.93rem; color:#3D4E58; border-left:3px solid var(--line);
    padding-left:.85rem; margin-bottom:1rem; line-height:1.5; }
  .verdict { display:flex; align-items:baseline; gap:.7rem; flex-wrap:wrap; margin-bottom:.15rem; }
  .verdict .name { font-family:'Fraunces',serif; font-size:1.5rem; font-weight:600; }
  .verdict .pct { font-family:'JetBrains Mono',monospace; font-size:.95rem; color:var(--muted); }
  .gloss { font-size:.83rem; color:var(--muted); margin-bottom:1rem; }
  .meter { display:flex; height:13px; border-radius:7px; overflow:hidden;
    background:#EDF1F4; margin-bottom:.75rem; }
  .meter span { height:100%; animation:grow .5s cubic-bezier(.2,.8,.3,1) both; }
  @keyframes grow { from{width:0 !important;} }
  .legend { display:flex; gap:1.1rem; flex-wrap:wrap; }
  .legend div { display:flex; align-items:center; gap:.4rem; font-size:.78rem; color:#42535D; }
  .dot { width:9px; height:9px; border-radius:50%; display:inline-block; }
  .num { font-family:'JetBrains Mono',monospace; color:var(--muted); }
  .flag { font-family:'JetBrains Mono',monospace; font-size:.63rem; letter-spacing:.1em;
    text-transform:uppercase; padding:.2rem .55rem; border-radius:20px;
    background:#FDF3E3; color:#9A6612; border:1px solid #F0DCB8; }
  .stButton>button { border-radius:9px; border:1px solid var(--line);
    background:var(--surface); color:var(--ink); font-size:.82rem; font-weight:500;
    padding:.34rem .8rem; transition:all .15s ease; }
  .stButton>button:hover { border-color:#1F7A8C; color:#1F7A8C;
    transform:translateY(-1px); box-shadow:0 3px 9px rgba(31,122,140,.14); }
  .stButton>button[kind="primary"] { background:linear-gradient(96deg,#1F7A8C,#2E6E9E);
    color:#fff; border:none; }
  .stButton>button[kind="primary"]:hover { box-shadow:0 5px 15px rgba(31,122,140,.32); color:#fff; }
  .stTabs [data-baseweb="tab-list"] { gap:.35rem; border-bottom:1px solid var(--line); }
  .stTabs [data-baseweb="tab"] { font-size:.9rem; font-weight:500; color:var(--muted);
    padding:.55rem 1rem; }
  .stTabs [aria-selected="true"] { color:#1F7A8C !important; }
  .stTextArea textarea { border-radius:11px !important; border:1px solid var(--line) !important;
    font-size:.93rem !important; background:var(--surface) !important; }
  .stTextArea textarea:focus { border-color:#1F7A8C !important;
    box-shadow:0 0 0 3px rgba(31,122,140,.13) !important; }
  [data-testid="stMetric"] { background:var(--surface); border:1px solid var(--line);
    border-radius:12px; padding:.75rem .9rem; }
  [data-testid="stMetricValue"] { font-family:'JetBrains Mono',monospace; font-size:1.5rem; }
  [data-testid="stMetricLabel"] { font-size:.76rem !important; color:var(--muted) !important; }
  [data-testid="stSidebar"] { background:#F6F8F9; border-right:1px solid var(--line); }
  #MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="masthead">
  <h1>BurnoutSense</h1>
  <div class="sub">Multilingual work-burnout expression classifier &middot; English &middot; Bahasa Malaysia &middot; Manglish</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice">
  <strong>Research prototype, not a diagnostic tool.</strong> This sorts wording into three
  categories from the Maslach Burnout Inventory. It reflects the language you type, not your
  wellbeing, and it is not an assessment of any person. If work is affecting your health,
  please speak to a doctor or a qualified mental health professional.
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import joblib

    cfg_p = ARTIFACTS / "config.json"
    clf_p = ARTIFACTS / "classifier.joblib"
    enc_p = ARTIFACTS / "label_encoder.joblib"
    missing = [str(p) for p in (cfg_p, clf_p, enc_p) if not p.exists()]
    if missing:
        return None, None, None, None, missing

    cfg = json.loads(cfg_p.read_text())
    clf, enc = joblib.load(clf_p), joblib.load(enc_p)
    rep = cfg.get("representation", "sbert")

    if rep == "tfidf":
        vec_p = ARTIFACTS / "vectorizer.joblib"
        if not vec_p.exists():
            return None, None, None, None, [str(vec_p)]
        feat = joblib.load(vec_p)
    else:
        from sentence_transformers import SentenceTransformer
        feat = SentenceTransformer(cfg.get("sbert_model",
                                           "paraphrase-multilingual-mpnet-base-v2"))
    return clf, enc, (rep, feat), cfg, []


def featurise(texts, bundle):
    rep, feat = bundle
    if rep == "tfidf":
        return feat.transform(list(texts))
    return feat.encode(list(texts), batch_size=32, convert_to_numpy=True,
                       show_progress_bar=False)


def predict(texts):
    clf, enc, bundle, _, _ = load_model()
    X = featurise(texts, bundle)
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
    else:
        s = np.atleast_2d(clf.decision_function(X))
        e = np.exp(s - s.max(axis=1, keepdims=True))
        proba = e / e.sum(axis=1, keepdims=True)

    classes = list(enc.classes_)
    out = pd.DataFrame({
        "text": list(texts),
        "predicted_dimension": enc.inverse_transform(proba.argmax(axis=1)),
        "confidence": proba.max(axis=1).round(4),
    })
    for j, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, j].round(4)
    return out, classes


def readout(row, classes):
    m = meta(row["predicted_dimension"])
    conf = float(row["confidence"])
    low = conf < LOW_CONFIDENCE

    bar, legend = "", ""
    for c in classes:
        p = float(row[f"p_{c}"])
        cm = meta(c)
        bar += f'<span style="width:{p*100:.2f}%;background:{cm["colour"]}"></span>'
        legend += (f'<div><i class="dot" style="background:{cm["colour"]}"></i>'
                   f'{cm["short"]} <b class="num">{p:.2f}</b></div>')

    flag = '<span class="flag">uncertain</span>' if low else ""
    note = ('<div class="gloss" style="color:#9A6612">The three scores are close together, '
            'so this result is not a confident one.</div>') if low else \
           f'<div class="gloss">{m["gloss"]}</div>'

    st.markdown(
        f'<div class="card" style="border-left:4px solid {m["colour"]}">'
        f'<div class="quoted">{row["text"][:420]}</div>'
        f'<div class="verdict"><span class="name" style="color:{m["colour"]}">{m["label"]}</span>'
        f'<span class="pct">{conf*100:.1f}%</span>{flag}</div>'
        f'{note}<div class="meter">{bar}</div><div class="legend">{legend}</div></div>',
        unsafe_allow_html=True)


_, enc_obj, _, cfg, missing = load_model()

with st.sidebar:
    st.markdown("### BurnoutSense")
    if missing:
        st.error("Model files not found.")
        for p in missing:
            st.code(p, language=None)
        st.caption("Run the export cell in the notebook, then copy `artifacts/` next to app.py.")
    else:
        st.success("Ready")
        st.caption("Paste any sentence about work in English, Malay, or Manglish.")
        with st.expander("About this model"):
            rep_name = {"tfidf": "TF-IDF", "sbert": "Sentence-BERT"}.get(
                cfg.get("representation"), cfg.get("representation"))
            st.markdown(
                f"**Representation** {rep_name}  \n"
                f"**Classifier** {cfg.get('classifier', '—')}  \n"
                f"**Trained on** {cfg.get('trained_on', '—')}  \n\n"
                f"**Test weighted F1** {cfg.get('test_f1_weighted', float('nan')):.4f}  \n"
                f"**Test macro F1** {cfg.get('test_f1_macro', float('nan')):.4f}  \n\n"
                f"Results below {LOW_CONFIDENCE:.2f} confidence are marked *uncertain*. "
                f"With three categories, random guessing scores about 0.33.")

if missing:
    st.warning("Add the model files to enable classification.")
    st.stop()

CLASSES = list(enc_obj.classes_)

tab1, tab2, tab3 = st.tabs(["Single sentence", "Several sentences", "Upload a file"])

with tab1:
    if "single_text" not in st.session_state:
        st.session_state.single_text = ""

    st.caption("Try an example")
    for col, (tag, sentence) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        col.button(tag, key=f"ex_{tag}", use_container_width=True,
                   on_click=lambda s=sentence: st.session_state.update(single_text=s))

    st.text_area("Your sentence", height=120, key="single_text",
                 placeholder="Penat sangat kerja, dah tak larat nak teruskan…",
                 label_visibility="collapsed")

    if st.button("Classify", type="primary", key="b1"):
        if not st.session_state.single_text.strip():
            st.info("Type or choose a sentence first.")
        else:
            res, classes = predict([st.session_state.single_text.strip()])
            readout(res.iloc[0], classes)

with tab2:
    blob = st.text_area("One sentence per line", height=200,
                        placeholder="I feel completely drained by this job\n"
                                    "Dah tak kisah dah pasal kerja ni\n"
                                    "Rasa macam tak pernah capai apa-apa")
    if st.button("Classify all", type="primary", key="b2"):
        lines = [l.strip() for l in blob.splitlines() if l.strip()]
        if not lines:
            st.info("Paste at least one line.")
        else:
            res, classes = predict(lines)
            counts = res["predicted_dimension"].value_counts()
            for col, c in zip(st.columns(len(CLASSES)), CLASSES):
                col.metric(meta(c)["short"], int(counts.get(c, 0)))
            st.write("")
            for _, row in res.iterrows():
                readout(row, classes)
            st.download_button("Download results (CSV)",
                               res.to_csv(index=False).encode("utf-8-sig"),
                               "burnoutsense_results.csv", "text/csv")

with tab3:
    up = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls", "tsv"])
    if up is not None:
        try:
            name = up.name.lower()
            if name.endswith((".xlsx", ".xls")):
                sheets = pd.read_excel(up, sheet_name=None)
                sheet = st.selectbox("Sheet", list(sheets.keys()))
                df_in = sheets[sheet]
            else:
                raw = up.getvalue()
                sep = "\t" if name.endswith(".tsv") else ","
                for encd in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
                    try:
                        df_in = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=encd)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    st.error("Could not read this file. Re-save it as UTF-8 CSV.")
                    st.stop()
        except Exception as e:
            st.error(f"Could not open the file: {e}")
            st.stop()

        st.caption(f"{len(df_in):,} rows · {len(df_in.columns)} columns")
        st.dataframe(df_in.head(5), use_container_width=True)

        text_cols = [c for c in df_in.columns if df_in[c].dtype == object] or list(df_in.columns)
        col = st.selectbox("Which column holds the text?", text_cols)
        limit = st.number_input("Maximum rows to classify", 1, 20000,
                                min(1000, len(df_in)), step=100)

        if st.button("Classify file", type="primary", key="b3"):
            series = df_in[col].astype(str).str.strip()
            keep = series[series.str.len() > 0].head(int(limit))
            if keep.empty:
                st.info(f"Column '{col}' has no usable text.")
            else:
                with st.spinner(f"Classifying {len(keep):,} rows…"):
                    res, classes = predict(keep.tolist())
                merged = df_in.loc[keep.index].copy()
                merged["predicted_dimension"] = res["predicted_dimension"].values
                merged["confidence"] = res["confidence"].values
                for c in classes:
                    merged[f"p_{c}"] = res[f"p_{c}"].values

                counts = res["predicted_dimension"].value_counts()
                for cm_, c in zip(st.columns(len(CLASSES)), CLASSES):
                    cm_.metric(meta(c)["short"], int(counts.get(c, 0)))

                n_low = int((res["confidence"] < LOW_CONFIDENCE).sum())
                if n_low:
                    st.caption(f"{n_low:,} of {len(res):,} results were uncertain.")

                st.bar_chart(counts, color="#1F7A8C")
                st.dataframe(merged.head(200), use_container_width=True)

                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
                    merged.to_excel(xw, index=False, sheet_name="predictions")
                    counts.rename("count").to_frame().to_excel(xw, sheet_name="summary")
                st.download_button("Download results (Excel)", buf.getvalue(),
                                   "burnoutsense_predictions.xlsx",
                                   "application/vnd.openxmlformats-officedocument."
                                   "spreadsheetml.sheet")
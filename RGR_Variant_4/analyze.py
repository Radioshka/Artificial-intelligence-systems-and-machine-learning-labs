"""Предобработка, частотный анализ, TF-IDF и наивный байесовский классификатор.

Классы размечены правилами по предметным словарям варианта «ИИ в финансах».
Правила прозрачны и сохранены в results/summary.json. Классификатор учится
на тексте публикации. Во втором эксперименте слова словаря из текста убраны,
чтобы проверить, остаётся ли тема в окружающем контексте, а не только в метке.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "publications.json"
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "Segoe UI"
plt.rcParams["axes.unicode_minus"] = False

# Термины из задания и термины предметной области варианта 4.
KEY_TERMS = [
    ("ML", r"\bml\b|\bmachine learning\b"),
    ("DL", r"\bdl\b"),
    ("Deep Learning", r"\bdeep learning\b"),
    ("NLP", r"\bnlp\b|\bnatural language processing\b"),
    ("Computer Vision", r"\bcomputer vision\b"),
    ("LLM", r"\bllm\b|\blarge language model"),
    ("Generative AI", r"\bgenerative ai\b|\bgenai\b"),
    ("Algorithmic trading", r"\balgorithmic trading\b|\balgo trading\b"),
    ("Fraud", r"\bfraud\b"),
    ("Risk", r"\brisk\b"),
    ("Fintech", r"\bfintech\b"),
    ("Portfolio", r"\bportfolio\b"),
]

CLASSES = [
    (
        "trading",
        "торговые стратегии",
        [
            "algorithmic trading", "algo trading", "high-frequency", "high frequency",
            "market making", "trading bot", "hedge fund", "quantitative trading",
            "quant fund", "order book", "execution algorithm", "hft",
        ],
    ),
    (
        "risk",
        "риск и регулирование",
        [
            "fraud", "anti-money", "money laundering", "aml", "kyc", "compliance",
            "regulation", "regulatory", "credit risk", "cybersecurity", "audit",
        ],
    ),
    (
        "investment",
        "инвестиции",
        [
            "portfolio", "asset management", "wealth management", "robo-advisor",
            "robo advisor", "mutual fund", "investment research", "valuation",
        ],
    ),
    (
        "banking",
        "банки и финтех",
        [
            "banking", "fintech", "payment", "lending", "loan", "insurance",
            "neobank", "core banking", "customer service",
        ],
    ),
]


def ensure_nltk() -> None:
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger_eng",
    ]
    for name in resources:
        try:
            nltk.download(name, quiet=True)
        except Exception as exc:
            print(f"nltk {name}: {exc}")


def wordnet_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def preprocess(text: str, stop: set[str], stemmer: PorterStemmer, lemmatizer: WordNetLemmatizer):
    cleaned = re.sub(r"[^A-Za-z\s-]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    tokens = [tok for tok in word_tokenize(cleaned) if tok.isalpha() and len(tok) > 2]
    no_stop = [tok for tok in tokens if tok not in stop]
    stems = [stemmer.stem(tok) for tok in no_stop]
    try:
        tagged = nltk.pos_tag(no_stop)
        lemmas = [lemmatizer.lemmatize(tok, wordnet_pos(tag)) for tok, tag in tagged]
    except LookupError:
        lemmas = [lemmatizer.lemmatize(tok) for tok in no_stop]
    return {"tokens": tokens, "no_stop": no_stop, "stems": stems, "lemmas": lemmas}


def label_of(text: str) -> tuple[str, str]:
    lowered = text.lower()
    scores = []
    for code, title, phrases in CLASSES:
        score = sum(lowered.count(phrase) for phrase in phrases)
        scores.append((score, code, title))
    scores.sort(reverse=True)
    if scores[0][0] == 0:
        return "other", "прочие материалы"
    return scores[0][1], scores[0][2]


def mask_lexicon(text: str) -> str:
    masked = text
    phrases = [phrase for _, _, phrases in CLASSES for phrase in phrases]
    for phrase in sorted(phrases, key=len, reverse=True):
        masked = re.sub(re.escape(phrase), " ", masked, flags=re.I)
    return masked


def count_terms(texts: list[str]) -> list[dict]:
    rows = []
    for name, pattern in KEY_TERMS:
        regex = re.compile(pattern, re.I)
        occ = sum(len(regex.findall(text)) for text in texts)
        docs = sum(1 for text in texts if regex.search(text))
        rows.append({"term": name, "occurrences": occ, "documents": docs})
    return rows


def fit_nb(texts: list[str], labels: list[str], fit_prior: bool = True) -> dict:
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.9,
                    max_features=4000,
                ),
            ),
            ("nb", MultinomialNB(alpha=1.0, fit_prior=fit_prior)),
        ]
    )
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.25, random_state=42, stratify=labels
        )
    except ValueError:
        x_train, x_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.25, random_state=42
        )
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    names = sorted(set(labels))
    matrix = confusion_matrix(y_test, predicted, labels=names).tolist()
    report = classification_report(y_test, predicted, labels=names, digits=3, zero_division=0)
    majority = Counter(y_train).most_common(1)[0][0]
    baseline = accuracy_score(y_test, [majority] * len(y_test))
    return {
        "n_train": len(x_train),
        "n_test": len(x_test),
        "fit_prior": fit_prior,
        "alpha": 1.0,
        "labels": names,
        "accuracy": round(float(accuracy_score(y_test, predicted)), 3),
        "macro_f1": round(float(f1_score(y_test, predicted, average="macro", zero_division=0)), 3),
        "baseline_accuracy": round(float(baseline), 3),
        "majority_class": majority,
        "confusion_matrix": matrix,
        "report": report,
    }


def save_bar(rows: list[dict], path: Path, title: str, value_key: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    names = [row["term"] for row in rows]
    values = [row[value_key] for row in rows]
    sns.barplot(x=values, y=names, ax=ax, color="#1f4e79")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion(result: dict, path: Path, title: str, names: dict[str, str]) -> None:
    labels = [names.get(code, code) for code in result["labels"]]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sns.heatmap(
        result["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Размеченный класс")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    ensure_nltk()
    payload = json.loads(DATA.read_text(encoding="utf-8"))
    articles = payload["articles"]
    off_topic = re.compile(r"job market|real estate|health insurance|attention economy", re.I)
    bodies = [
        row for row in articles
        if row["text_source"] != "title"
        and row["n_chars"] >= 500
        and not off_topic.search(row["title"])
    ]
    if len(bodies) < 24:
        bodies = articles
    texts = [f"{row['title']}. {row['text']}" for row in bodies]

    stop = set(stopwords.words("english"))
    stop.update({"forbes", "said", "also", "would", "could", "one", "new", "year", "years"})
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    prepared = [preprocess(text, stop, stemmer, lemmatizer) for text in texts]
    example_source = texts[0][:280]
    example = prepared[0]
    lemma_counts = Counter(tok for row in prepared for tok in row["lemmas"])
    top_lemmas = lemma_counts.most_common(20)

    term_rows = count_terms(texts)
    save_bar(
        term_rows,
        FIG / "term_frequency.png",
        "Частота ключевых терминов в собранном корпусе",
        "occurrences",
        "Число вхождений",
    )
    save_bar(
        term_rows,
        FIG / "term_documents.png",
        "В скольких публикациях встречается термин",
        "documents",
        "Число публикаций",
    )

    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        max_features=5000,
    )
    matrix = tfidf.fit_transform(texts)
    means = matrix.mean(axis=0).A1
    pairs = sorted(zip(means, tfidf.get_feature_names_out()), reverse=True)[:20]
    tfidf_top = [{"term": term, "mean_tfidf": round(float(score), 4)} for score, term in pairs]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x=[row["mean_tfidf"] for row in tfidf_top],
        y=[row["term"] for row in tfidf_top],
        ax=ax,
        color="#0e7c66",
    )
    ax.set_title("Топ-20 терминов и словосочетаний по среднему TF-IDF")
    ax.set_xlabel("Средний TF-IDF")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG / "tfidf_top.png", dpi=160)
    plt.close(fig)

    labeled = []
    for row, text in zip(bodies, texts):
        code, title = label_of(text)
        labeled.append({**row, "class_code": code, "class_title": title, "full_text": text})
    counts = Counter(item["class_code"] for item in labeled)
    # Класс «прочие» — остаток без устойчивого словаря, в классификатор не входит.
    usable_codes = {code for code, n in counts.items() if n >= 3 and code != "other"}
    usable = [item for item in labeled if item["class_code"] in usable_codes]
    names = {code: title for code, title, _ in CLASSES}
    names["other"] = "прочие материалы"

    fig, ax = plt.subplots(figsize=(8, 4.6))
    order = [code for code, _title, _ in CLASSES if code in counts] + (
        ["other"] if "other" in counts else []
    )
    sns.barplot(
        x=[counts[code] for code in order],
        y=[names[code] for code in order],
        ax=ax,
        color="#b85c38",
    )
    ax.set_title("Распределение публикаций по тематическим классам")
    ax.set_xlabel("Число публикаций")
    fig.tight_layout()
    fig.savefig(FIG / "class_distribution.png", dpi=160)
    plt.close(fig)

    full_texts = [item["full_text"] for item in usable]
    full_labels = [item["class_code"] for item in usable]
    masked_texts = [mask_lexicon(text) for text in full_texts]
    nb_full = fit_nb(full_texts, full_labels, fit_prior=True)
    nb_balanced = fit_nb(full_texts, full_labels, fit_prior=False)
    nb_masked = fit_nb(masked_texts, full_labels, fit_prior=False)
    save_confusion(
        nb_full,
        FIG / "confusion_full.png",
        "Naive Bayes с априорными частотами классов",
        names,
    )
    save_confusion(
        nb_balanced,
        FIG / "confusion_balanced.png",
        "Naive Bayes с равными априорными вероятностями",
        names,
    )
    save_confusion(
        nb_masked,
        FIG / "confusion_masked.png",
        "Равные априорные вероятности, словарь меток удалён",
        names,
    )

    def site_name(row: dict) -> str:
        url = row.get("url") or ""
        if "arxiv.org" in url:
            return "arXiv"
        if "techcrunch.com" in url:
            return "TechCrunch"
        if "cnbc.com" in url:
            return "CNBC"
        if "theverge.com" in url:
            return "The Verge"
        if "forbes.com" in url:
            return "Forbes"
        return "другой"

    dates = sorted(row["published"] for row in bodies if row["published"])
    site_counts = Counter(site_name(row) for row in bodies)
    summary = {
        "n_collected": payload["n_kept"],
        "n_with_body": payload["n_with_body"],
        "n_analyzed": len(bodies),
        "n_classified": len(usable),
        "date_min": dates[0] if dates else "",
        "date_max": dates[-1] if dates else "",
        "site_counts": dict(site_counts),
        "collected_at": payload["collected_at"],
        "class_counts": {names[code]: counts[code] for code in order},
        "class_counts_used": {
            names[code]: sum(1 for item in usable if item["class_code"] == code)
            for code in sorted(usable_codes)
        },
        "term_rows": term_rows,
        "top_lemmas": [{"lemma": lemma, "count": count} for lemma, count in top_lemmas],
        "tfidf_top": tfidf_top,
        "preprocess_example": {
            "source": example_source,
            "tokens": example["tokens"][:25],
            "no_stop": example["no_stop"][:25],
            "stems": example["stems"][:25],
            "lemmas": example["lemmas"][:25],
        },
        "nb_full": nb_full,
        "nb_balanced": nb_balanced,
        "nb_masked": nb_masked,
        "sample_titles": [
            {
                "published": item["published"],
                "title": item["title"],
                "class_title": item["class_title"],
                "url": item["url"],
                "text_source": item["text_source"],
            }
            for item in labeled[:12]
        ],
    }
    (RES / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RES / "classification_report_full.txt").write_text(nb_full["report"], encoding="utf-8")
    (RES / "classification_report_balanced.txt").write_text(nb_balanced["report"], encoding="utf-8")
    (RES / "classification_report_masked.txt").write_text(nb_masked["report"], encoding="utf-8")
    print(
        f"analyzed {len(bodies)}, classes {dict(counts)}, "
        f"NB acc {nb_full['accuracy']} / balanced {nb_balanced['accuracy']} / masked {nb_masked['accuracy']}"
    )


if __name__ == "__main__":
    main()

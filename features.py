"""Shared code: sentence splitting, stylometric features, document aggregation.

Every model file imports from here so training and the app use identical features.
"""
import re
import numpy as np
from joblib import Parallel, delayed

FUNCTION_WORDS = {
    'a', 'an', 'the', 'and', 'but', 'or', 'nor', 'for', 'so', 'yet', 'although', 'because',
    'since', 'unless', 'while', 'whereas', 'in', 'on', 'at', 'by', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'of', 'off', 'over', 'under', 'again', 'further', 'once', 'i', 'me', 'my',
    'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they',
    'them', 'their', 'theirs', 'themselves', 'this', 'that', 'these', 'those', 'who', 'whom',
    'whose', 'which', 'what', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must', 'not', 'no', 'if', 'than', 'too', 'very', 'just',
    'also', 'only',
}
FIRST_PERSON = {'i', 'me', 'my', 'myself', 'we', 'our', 'us'}
MIN_WORDS = 4  # sentences shorter than this are not scored (too little signal)

FEATURE_NAMES = [
    'log_n_words', 'avg_word_length', 'unique_word_ratio', 'punctuation_ratio',
    'uppercase_ratio', 'stopword_ratio', 'comma_per_word', 'long_word_ratio',
    'first_person_ratio',
]

_SPLIT = re.compile(r'(?<=[.!?])["\')\]]*\s+|\n+')


def split_sentences(text):
    return [s.strip() for s in _SPLIT.split(text.strip()) if s and s.strip()]


def is_scorable(sentence):
    return len(sentence.split()) >= MIN_WORDS


def sentence_features(s):
    words = s.split()
    n = max(len(words), 1)
    clean = [w.strip('.,!?;:"\'()[]').lower() for w in words]
    chars = max(len(s), 1)
    return [
        np.log1p(len(words)),
        sum(len(w) for w in clean) / n,
        len(set(clean)) / n,
        sum(1 for c in s if not (c.isalnum() or c.isspace())) / chars,
        sum(1 for c in s if c.isupper()) / chars,
        sum(w in FUNCTION_WORDS for w in clean) / n,
        s.count(',') / n,
        sum(len(w) >= 7 for w in clean) / n,
        sum(w in FIRST_PERSON for w in clean) / n,
    ]


def _featurize_chunk(texts):
    out = []
    for t in texts:
        sents = [s for s in split_sentences(t) if is_scorable(s)]
        out.append(np.array([sentence_features(s) for s in sents],
                            dtype=np.float32).reshape(-1, len(FEATURE_NAMES)))
    return out


def featurize_docs(texts, n_jobs=-1, chunk=200):
    """List of (n_sentences_i, n_features) arrays, one per document."""
    texts = list(texts)
    chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    res = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_featurize_chunk)(c) for c in chunks)
    return [a for r in res for a in r]


def sentence_dataset(df, n_docs, max_per_doc, seed=42, n_jobs=-1):
    """Sentence-level X, y. Each sentence inherits its document's label."""
    df = df.sample(min(n_docs, len(df)), random_state=seed)
    Fs = featurize_docs(df['text'], n_jobs)
    rng = np.random.default_rng(seed)
    X, y = [], []
    for F, label in zip(Fs, df['is_ai'].to_numpy()):
        if len(F) == 0:
            continue
        if len(F) > max_per_doc:
            F = F[rng.choice(len(F), max_per_doc, replace=False)]
        X.append(F)
        y.append(np.full(len(F), label))
    return np.vstack(X), np.concatenate(y)


# ---------- document-level aggregation (input to the RandomForestRegressor) ----------
def doc_feature_names(k):
    return (['n_sentences', 'mean_p', 'std_p', 'median_p', 'max_p', 'min_p',
             'frac_p_gt_0.5', 'frac_p_gt_0.8', 'frac_p_lt_0.2',
             'mean_sent_len', 'sent_len_cv', 'mean_cluster_entropy', 'mean_cluster_maxpost']
            + [f'mean_cluster_{i}' for i in range(k)])


def aggregate_doc(p, post, lengths):
    """p: (n,) sentence AI-probabilities; post: (n,k) GMM posteriors; lengths: (n,) words."""
    entropy = -(post * np.log(post + 1e-12)).sum(axis=1)
    feats = [len(p), p.mean(), p.std(), np.median(p), p.max(), p.min(),
             (p > 0.5).mean(), (p > 0.8).mean(), (p < 0.2).mean(),
             lengths.mean(), lengths.std() / max(lengths.mean(), 1e-6),  # CV = burstiness
             entropy.mean(), post.max(axis=1).mean(), *post.mean(axis=0)]
    return np.array(feats, dtype=np.float32)
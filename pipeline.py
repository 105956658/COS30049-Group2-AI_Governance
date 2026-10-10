"""Glue used by regressor.py and by the app: chains classifier -> GMM -> RFR."""
"""Mean tree prediction plus the 5th-95th percentile spread across trees."""

import joblib
import numpy as np
from features import (featurize_docs, aggregate_doc, split_sentences, is_scorable,
                      sentence_features)


def load_models(model_dir='models'):
    return {
        'clf': joblib.load(f'{model_dir}/classifier.joblib'),
        'gmm': joblib.load(f'{model_dir}/gmm.joblib'),      # dict: scaler, gmm, k
        'rfr': joblib.load(f'{model_dir}/regressor.joblib'),  # dict: model, k
    }


def sentence_scores(F, clf, gmm_bundle):
    p = clf.predict_proba(F)[:, 1]                           # P(AI) per sentence
    Z = np.clip(gmm_bundle['scaler'].transform(F), -5, 5)
    return p, gmm_bundle['gmm'].predict_proba(Z)             # soft cluster memberships


def build_doc_matrix(texts, clf, gmm_bundle, n_jobs=-1):
    """Document-level feature matrix. Returns (X, kept_row_indices)."""
    Fs = featurize_docs(texts, n_jobs)
    keep = [i for i, F in enumerate(Fs) if len(F) > 0]
    big = np.vstack([Fs[i] for i in keep])
    p, post = sentence_scores(big, clf, gmm_bundle)
    offs = np.cumsum([0] + [len(Fs[i]) for i in keep])
    rows = [aggregate_doc(p[a:b], post[a:b], np.expm1(big[a:b, 0]))
            for a, b in zip(offs[:-1], offs[1:])]
    return np.vstack(rows), np.array(keep)


def predict_with_interval(rfr, X, lo=5, hi=95):
    """Mean tree prediction plus a percentile band across the individual trees."""
    per_tree = np.stack([t.predict(X.astype(np.float32)) for t in rfr.estimators_])
    return per_tree.mean(0), np.percentile(per_tree, lo, axis=0), np.percentile(per_tree, hi, axis=0)


def analyze(text, models):
    """What the app calls: per-sentence scores + overall score with confidence range."""
    sents = split_sentences(text)
    mask = [is_scorable(s) for s in sents]
    scored = [s for s, m in zip(sents, mask) if m]
    if not scored:
        return None
    F = np.array([sentence_features(s) for s in scored], dtype=np.float32)
    p, post = sentence_scores(F, models['clf'], models['gmm'])
    x = aggregate_doc(p, post, np.expm1(F[:, 0]))[None]
    score, low, high = predict_with_interval(models['rfr']['model'], x)
    it = iter(p)
    return {
        'sentences': [{'text': s, 'p_ai': float(next(it)) if m else None}
                      for s, m in zip(sents, mask)],   # None = too short, render neutral
        'overall': float(score[0]), 'low': float(low[0]), 'high': float(high[0]),
    }
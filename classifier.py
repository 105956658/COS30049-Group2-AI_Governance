"""Model 1 - per-sentence classifier (sentence-level P(AI)). Drives the colouring.

Usage: python 01_classifier.py --model logreg   (or --model svc for calibrated LinearSVC)
"""
import argparse, os
import joblib, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from features import sentence_dataset, FEATURE_NAMES

ap = argparse.ArgumentParser()
ap.add_argument('--model', choices=['logreg', 'svc'], default='logreg')
ap.add_argument('--train', default='data/splits/train.parquet')
ap.add_argument('--val', default='data/splits/val.parquet')
ap.add_argument('--n-docs', type=int, default=40000)
ap.add_argument('--max-sent-per-doc', type=int, default=12)
ap.add_argument('--out', default='models/classifier.joblib')
a = ap.parse_args()

X, y = sentence_dataset(pd.read_parquet(a.train), a.n_docs, a.max_sent_per_doc, seed=1)
Xv, yv = sentence_dataset(pd.read_parquet(a.val), a.n_docs // 8, a.max_sent_per_doc, seed=2)
print(f'train sentences: {len(y):,} (AI share {y.mean():.2f}) | val sentences: {len(yv):,}')

if a.model == 'logreg':
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight='balanced'))
else:  # LinearSVC has no predict_proba -> Platt scaling via CalibratedClassifierCV
    clf = make_pipeline(StandardScaler(),
                        CalibratedClassifierCV(LinearSVC(dual=False, class_weight='balanced'),
                                               method='sigmoid', cv=3))
clf.fit(X, y)

p = clf.predict_proba(Xv)[:, 1]
print(f'val sentence-level  acc={accuracy_score(yv, p > 0.5):.3f}  '
      f'AUC={roc_auc_score(yv, p):.3f}  Brier={brier_score_loss(yv, p):.3f}')
if a.model == 'logreg':
    coefs = clf[-1].coef_[0]
    for n, c in sorted(zip(FEATURE_NAMES, coefs), key=lambda t: t[1]):
        print(f'  {n:22s} {c:+.3f}')

os.makedirs(os.path.dirname(a.out), exist_ok=True)
joblib.dump(clf, a.out)
print('saved', a.out)
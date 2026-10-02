"""Model 1 - per-sentence classifier (sentence-level P(AI)). Drives the colouring in the app.

Two candidate models are compared on the SAME validation, test and external holdout sets:
  logreg : standardised features -> logistic regression (native probabilities)
  svc    : standardised features -> LinearSVC wrapped in Platt scaling, because
           LinearSVC has no predict_proba and the app needs a probability per sentence.

Usage:
  python classifier.py --model logreg --out models/clf_logreg.joblib
  python classifier.py --model svc    --out models/clf_svc.joblib
Then copy the winner to models/classifier.joblib (pipeline.py and regressor.py load that name).
"""
import argparse, os
import joblib, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                             precision_recall_fscore_support, confusion_matrix)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from features import sentence_dataset, FEATURE_NAMES

# ---------------------------------------------------------------- arguments
ap = argparse.ArgumentParser()
ap.add_argument('--model', choices=['logreg', 'svc'], default='logreg')
ap.add_argument('--train', default='data/splits/train.csv')
ap.add_argument('--val', default='data/splits/val.csv')
ap.add_argument('--test', default='data/splits/test.csv')
ap.add_argument('--holdout', default='data/splits/holdout_2026.csv')
ap.add_argument('--n-docs', type=int, default=40000)        # training documents sampled
ap.add_argument('--n-eval-docs', type=int, default=5000)    # documents per evaluation set
ap.add_argument('--max-sent-per-doc', type=int, default=12)  # cap so long essays do not dominate
ap.add_argument('--out', default='models/classifier.joblib')
ap.add_argument('--results', default='results/classifier_results.csv')
a = ap.parse_args()

# ---------------------------------------------------------------- training data
# Each sentence inherits the label of its document (sentence-level labels do not exist).
X, y = sentence_dataset(pd.read_csv(a.train), a.n_docs, a.max_sent_per_doc, seed=1)
print(f'train sentences: {len(y):,} (AI share {y.mean():.2f})')

# ---------------------------------------------------------------- model
if a.model == 'logreg':
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=1000, class_weight='balanced'))
else:
    # Platt scaling (sigmoid) with 3-fold CV turns LinearSVC margins into probabilities
    clf = make_pipeline(StandardScaler(),
                        CalibratedClassifierCV(LinearSVC(dual=False, class_weight='balanced'),
                                               method='sigmoid', cv=3))
clf.fit(X, y)


# ---------------------------------------------------------------- evaluation
def evaluate(name, path, seed):
    """Score one split. Fixed seeds mean every model sees exactly the same sentences."""
    Xe, ye = sentence_dataset(pd.read_csv(path), a.n_eval_docs, a.max_sent_per_doc, seed=seed)
    p = clf.predict_proba(Xe)[:, 1]
    pred = p > 0.5
    prec, rec, f1, _ = precision_recall_fscore_support(ye, pred, average='binary')
    row = {'model': a.model, 'split': name, 'n_sentences': len(ye),
           'accuracy': accuracy_score(ye, pred), 'precision': prec, 'recall': rec,
           'f1': f1, 'auc': roc_auc_score(ye, p), 'brier': brier_score_loss(ye, p)}
    print(f"\n[{name}] n={len(ye):,}  acc={row['accuracy']:.3f}  P={prec:.3f}  R={rec:.3f}  "
          f"F1={f1:.3f}  AUC={row['auc']:.3f}  Brier={row['brier']:.3f}")
    print('confusion matrix [[TN FP] [FN TP]]:')
    print(confusion_matrix(ye, pred))
    return row


# val: used for tuning decisions; test: untouched final score; holdout: newer 2026 data
# from a different source, the real test of generalisation
rows = [evaluate('val', a.val, seed=2),
        evaluate('test', a.test, seed=6),
        evaluate('holdout_2026', a.holdout, seed=7)]

# Feature weights (logreg only): sign shows which direction pushes towards "AI"
if a.model == 'logreg':
    print('\nStandardised coefficients (positive = more AI-like):')
    for n, c in sorted(zip(FEATURE_NAMES, clf[-1].coef_[0]), key=lambda t: t[1]):
        print(f'  {n:22s} {c:+.3f}')

# ---------------------------------------------------------------- save
os.makedirs(os.path.dirname(a.out), exist_ok=True)
joblib.dump(clf, a.out)
print('\nsaved', a.out)

# Append to a results table so both models end up side by side for the report
os.makedirs(os.path.dirname(a.results), exist_ok=True)
new = pd.DataFrame(rows).round(4)
if os.path.exists(a.results):
    old = pd.read_csv(a.results)
    old = old[old['model'] != a.model]          # re-running a model replaces its old rows
    new = pd.concat([old, new], ignore_index=True)
new.to_csv(a.results, index=False)
print('results table ->', a.results)
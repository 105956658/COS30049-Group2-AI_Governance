"""Innovation model A - LightGBM (gradient-boosted trees) sentence classifier.

Same 9 stylometric features, same training sentences and the SAME val/test/holdout
sentences as classifier.py (identical seeds and sizes), so the comparison with the
LinearSVC / logistic-regression baseline is like for like.

Why it is worth testing: the baselines are linear, so each feature pushes the score in
one fixed direction. Boosted trees can learn interactions (e.g. a long sentence with few
commas means something different from a short one with few commas). If LightGBM is not
better, the limit is the features, not the model.

Setup:  python -m pip install lightgbm
Usage:  python lgbm_classifier.py
"""
import argparse, os
import joblib, numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, roc_auc_score, brier_score_loss,
                             precision_recall_fscore_support, confusion_matrix)
from features import sentence_dataset, FEATURE_NAMES

# ---------------------------------------------------------------- arguments
# defaults MUST match classifier.py so every model sees exactly the same sentences
ap = argparse.ArgumentParser()
ap.add_argument('--train', default='data/splits/train.csv')
ap.add_argument('--val', default='data/splits/val.csv')
ap.add_argument('--test', default='data/splits/test.csv')
ap.add_argument('--holdout', default='data/splits/holdout_2026.csv')
ap.add_argument('--n-docs', type=int, default=40000)
ap.add_argument('--n-eval-docs', type=int, default=5000)
ap.add_argument('--max-sent-per-doc', type=int, default=12)
ap.add_argument('--out', default='models/clf_lgbm.joblib')
ap.add_argument('--results', default='results/classifier_results.csv')
a = ap.parse_args()

# ---------------------------------------------------------------- data
X, y = sentence_dataset(pd.read_csv(a.train), a.n_docs, a.max_sent_per_doc, seed=1)
Xv, yv = sentence_dataset(pd.read_csv(a.val), a.n_eval_docs, a.max_sent_per_doc, seed=2)
print(f'train sentences: {len(y):,} (AI share {y.mean():.2f})')

# ---------------------------------------------------------------- model
# Early stopping on the validation sentences stops boosting before it overfits.
# No scaling needed: trees are insensitive to feature scale.
clf = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, num_leaves=63,
    min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    class_weight='balanced',          # same class handling as the baselines
    random_state=0, n_jobs=-1, verbose=-1)
clf.fit(X, y, eval_set=[(Xv, yv)], eval_metric='auc',
        callbacks=[lgb.early_stopping(50, verbose=False)])
print('trees used:', clf.best_iteration_)


# ---------------------------------------------------------------- evaluation
def evaluate(name, path, seed):
    """Same function and seeds as classifier.py."""
    Xe, ye = sentence_dataset(pd.read_csv(path), a.n_eval_docs, a.max_sent_per_doc, seed=seed)
    p = clf.predict_proba(Xe)[:, 1]
    pred = p > 0.5
    prec, rec, f1, _ = precision_recall_fscore_support(ye, pred, average='binary')
    row = {'model': 'lightgbm', 'split': name, 'n_sentences': len(ye),
           'accuracy': accuracy_score(ye, pred), 'precision': prec, 'recall': rec,
           'f1': f1, 'auc': roc_auc_score(ye, p), 'brier': brier_score_loss(ye, p)}
    print(f"\n[{name}] n={len(ye):,}  acc={row['accuracy']:.3f}  P={prec:.3f}  R={rec:.3f}  "
          f"F1={f1:.3f}  AUC={row['auc']:.3f}  Brier={row['brier']:.3f}")
    print('confusion matrix [[TN FP] [FN TP]]:')
    print(confusion_matrix(ye, pred))
    return row


# note: val was used for early stopping, so test is the fair number to report
rows = [evaluate('val', a.val, seed=2),
        evaluate('test', a.test, seed=6),
        evaluate('holdout_2026', a.holdout, seed=7)]

print('\nFeature importance (split gain share):')
gain = clf.booster_.feature_importance(importance_type='gain')
for n, g in sorted(zip(FEATURE_NAMES, gain / gain.sum()), key=lambda t: -t[1]):
    print(f'  {n:22s} {g:.3f}')

# ---------------------------------------------------------------- save
os.makedirs(os.path.dirname(a.out), exist_ok=True)
joblib.dump(clf, a.out)
print('\nsaved', a.out)

os.makedirs(os.path.dirname(a.results), exist_ok=True)
new = pd.DataFrame(rows).round(4)
if os.path.exists(a.results):
    old = pd.read_csv(a.results)
    old = old[old['model'] != 'lightgbm']
    new = pd.concat([old, new], ignore_index=True)
new.to_csv(a.results, index=False)
print('results table ->', a.results)
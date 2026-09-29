"""Model 3 - RandomForestRegressor: document-level score + confidence range.

Fit on the VALIDATION split: the classifier and GMM never saw those documents, so
their sentence outputs are honest (not over-confident, as they would be on train data).
Evaluate on the untouched test split and the external 2026 holdout.
"""
import argparse, os
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error
from features import doc_feature_names
from pipeline import build_doc_matrix, predict_with_interval

ap = argparse.ArgumentParser()
ap.add_argument('--fit', default='data/splits/val.parquet')
ap.add_argument('--evals', nargs='+', default=['data/splits/test.parquet',
                                               'data/splits/holdout_2026.parquet'])
ap.add_argument('--n-fit-docs', type=int, default=20000)
ap.add_argument('--n-eval-docs', type=int, default=10000)
ap.add_argument('--model-dir', default='models')
a = ap.parse_args()

clf = joblib.load(f'{a.model_dir}/classifier.joblib')
gb = joblib.load(f'{a.model_dir}/gmm.joblib')
k = gb['k']

df = pd.read_parquet(a.fit)
df = df.sample(min(a.n_fit_docs, len(df)), random_state=4).reset_index(drop=True)
X, keep = build_doc_matrix(df['text'], clf, gb)
y = df['is_ai'].to_numpy()[keep].astype(float)
print(f'fitting on {len(y):,} documents, {X.shape[1]} features')

rfr = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                            n_jobs=-1, random_state=0).fit(X, y)
for n, imp in sorted(zip(doc_feature_names(k), rfr.feature_importances_), key=lambda t: -t[1])[:8]:
    print(f'  {n:24s} {imp:.3f}')

names = doc_feature_names(k)
mean_p_col = names.index('mean_p')
for path in a.evals:
    d = pd.read_parquet(path)
    d = d.sample(min(a.n_eval_docs, len(d)), random_state=5).reset_index(drop=True)
    Xe, kp = build_doc_matrix(d['text'], clf, gb)
    ye = d['is_ai'].to_numpy()[kp]
    score, lo, hi = predict_with_interval(rfr, Xe)
    print(f'\n[{os.path.basename(path)}] n={len(ye):,}')
    print(f'  RFR score      AUC={roc_auc_score(ye, score):.3f}  acc@0.5={((score > .5) == ye).mean():.3f}  '
          f'MAE={mean_absolute_error(ye, score):.3f}')
    print(f'  baseline mean sentence prob  AUC={roc_auc_score(ye, Xe[:, mean_p_col]):.3f}')
    print(f'  90% tree band  mean width={np.mean(hi - lo):.3f}')

os.makedirs(a.model_dir, exist_ok=True)
joblib.dump({'model': rfr, 'k': k, 'feature_names': names}, f'{a.model_dir}/regressor.joblib')
print('\nsaved', f'{a.model_dir}/regressor.joblib')
"""Model 2 - Gaussian Mixture on sentence stylometry (unsupervised, labels never used to fit).

Outputs soft cluster memberships per sentence. Labels are only used afterwards to
describe what each cluster looks like.
"""
import argparse, os
import joblib, numpy as np, pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from features import sentence_dataset

ap = argparse.ArgumentParser()
ap.add_argument('--train', default='data/splits/train.parquet')
ap.add_argument('--n-docs', type=int, default=20000)
ap.add_argument('--max-sent-per-doc', type=int, default=8)
ap.add_argument('--k', type=int, default=None, help='fix k instead of choosing by BIC')
ap.add_argument('--kmax', type=int, default=8)
ap.add_argument('--out', default='models/gmm.joblib')
a = ap.parse_args()

X, y = sentence_dataset(pd.read_parquet(a.train), a.n_docs, a.max_sent_per_doc, seed=3)
scaler = StandardScaler().fit(X)
Z = np.clip(scaler.transform(X), -5, 5)
print(f'{len(Z):,} sentences')

k = a.k
if k is None:
    sub = Z[np.random.default_rng(0).choice(len(Z), min(40000, len(Z)), replace=False)]
    bics = {}
    for kk in range(2, a.kmax + 1):
        bics[kk] = GaussianMixture(kk, covariance_type='full', random_state=0).fit(sub).bic(sub)
        print(f'  k={kk}  BIC={bics[kk]:.0f}')
    k = min(bics, key=bics.get)
print('using k =', k)

gmm = GaussianMixture(k, covariance_type='full', n_init=2, random_state=0).fit(Z)
hard = gmm.predict(Z)
print(f'NMI(cluster, label) = {normalized_mutual_info_score(y, hard):.3f}  (interpretation only)')
print(pd.DataFrame({'c': hard, 'ai': y}).groupby('c')['ai'].agg(['size', 'mean'])
      .rename(columns={'mean': 'ai_share'}).round(3))

os.makedirs(os.path.dirname(a.out), exist_ok=True)
joblib.dump({'scaler': scaler, 'gmm': gmm, 'k': k}, a.out)
print('saved', a.out)
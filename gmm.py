"""Model 2 - Gaussian Mixture Model on AI-written sentences only (unsupervised).

Labels are used ONLY to choose which class we cluster; they are never a model input.
"""
import argparse, os
import joblib, numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from features import split_sentences, is_scorable, sentence_features, FEATURE_NAMES

ap = argparse.ArgumentParser()
ap.add_argument('--train', default='data/splits/train.csv')
ap.add_argument('--n-docs', type=int, default=20000)
ap.add_argument('--max-sent-per-doc', type=int, default=8)
ap.add_argument('--k', type=int, default=None)      # run once to see BIC, then fix k
ap.add_argument('--kmax', type=int, default=12)
ap.add_argument('--out', default='models/gmm.joblib')
a = ap.parse_args()

# 1. Keep ONE class (AI) and sample sentences, remembering text + generator for profiling
df = pd.read_csv(a.train)
df = df[df['is_ai'] == 1]
df = df.sample(min(a.n_docs, len(df)), random_state=3)
rng = np.random.default_rng(3)
texts, groups = [], []
for text, g in zip(df['text'], df['generator']):
    sents = [s for s in split_sentences(text) if is_scorable(s)]
    if len(sents) > a.max_sent_per_doc:               # cap so long essays don't dominate
        sents = [sents[i] for i in rng.choice(len(sents), a.max_sent_per_doc, replace=False)]
    texts += sents
    groups += [g] * len(sents)
X = np.array([sentence_features(s) for s in texts], dtype=np.float32)
print(f'{len(X):,} AI sentences')

# 2. Standardise (clip outliers) so no feature dominates the covariance
scaler = StandardScaler().fit(X)
Z = np.clip(scaler.transform(X), -5, 5)

# 3. Choose k with BIC on a subsample (lower = better; look for the elbow)
k = a.k
if k is None:
    sub = Z[rng.choice(len(Z), min(40000, len(Z)), replace=False)]
    bics = {kk: GaussianMixture(kk, covariance_type='full', reg_covar=1e-4,
                                random_state=0).fit(sub).bic(sub)
            for kk in range(2, a.kmax + 1)}
    for kk, b in bics.items():
        print(f'  k={kk}  BIC={b:.0f}')
    k = min(bics, key=bics.get)
print('using k =', k)

gmm = GaussianMixture(k, covariance_type='full', reg_covar=1e-4, n_init=2, random_state=0).fit(Z)
hard = gmm.predict(Z)

# 4. Describe each cluster: which features differ most from the AI-class average?
prof = pd.DataFrame(X, columns=FEATURE_NAMES)
mu, sd = prof.mean(), prof.std()
prof['c'] = hard
for c, g in prof.groupby('c'):
    z = ((g[FEATURE_NAMES].mean() - mu) / sd).sort_values(key=abs, ascending=False)
    print(f'\nCluster {c}: n={len(g):,} ({len(g)/len(prof):.1%})')
    print('  top features (z-score vs AI average):', z.head(3).round(2).to_dict())
    for i in g.sample(min(3, len(g)), random_state=0).index:
        print('   e.g.', texts[i][:200])

# 5. Composition check: which generator produced each cluster's sentences?
print('\nGenerator composition per cluster:')
print(pd.crosstab(hard, np.array(groups), normalize='index').round(2))

os.makedirs(os.path.dirname(a.out), exist_ok=True)
joblib.dump({'scaler': scaler, 'gmm': gmm, 'k': k}, a.out)
print('saved', a.out)

clf = joblib.load('models/classifier.joblib')
p_ai = clf.predict_proba(X)[:, 1]
print('\nMean classifier P(AI) per cluster:')
print(pd.Series(p_ai).groupby(hard).mean().round(3))

# unseen HUMAN sentences placed into the AI-fitted clusters (never used for fitting)
hum = pd.read_csv(a.train); hum = hum[hum['is_ai'] == 0].sample(5000, random_state=9)
hs = [s for t in hum['text'] for s in split_sentences(t)[:8] if is_scorable(s)]
HZ = np.clip(scaler.transform(np.array([sentence_features(s) for s in hs], dtype=np.float32)), -5, 5)
print('\nShare of AI vs human sentences per cluster:')
print(pd.DataFrame({'AI': pd.Series(hard).value_counts(normalize=True),
                    'human': pd.Series(gmm.predict(HZ)).value_counts(normalize=True)}).round(3))
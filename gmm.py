"""Model 2 - Gaussian Mixture Model on AI-written sentences only (unsupervised).

The GMM is fitted on AI sentences from the TRAIN split. Every check that involves the
classifier uses the VALIDATION split instead, because the classifier was trained on
train sentences and would look over-confident there.

"""
import argparse, os
import joblib, numpy as np, pandas as pd
pd.set_option('display.width', 200)          # print wide tables in full
pd.set_option('display.max_columns', 40)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from features import split_sentences, is_scorable, sentence_features, FEATURE_NAMES

ap = argparse.ArgumentParser()
ap.add_argument('--train', default='data/splits/train.csv')
ap.add_argument('--val', default='data/splits/val.csv')
ap.add_argument('--classifier', default='models/classifier.joblib')
ap.add_argument('--n-docs', type=int, default=20000)        # AI documents used to fit
ap.add_argument('--n-check-docs', type=int, default=5000)   # val documents per class for checks
ap.add_argument('--max-sent-per-doc', type=int, default=8)
ap.add_argument('--k', type=int, default=None)
ap.add_argument('--kmax', type=int, default=12)
ap.add_argument('--elbow', type=float, default=0.10)        
ap.add_argument('--out', default='models/gmm.joblib')
ap.add_argument('--results-dir', default='results')
a = ap.parse_args()
os.makedirs(a.results_dir, exist_ok=True)
rng = np.random.default_rng(3)


# ---------------------------------------------------------------- helpers
def sample_sentences(texts, max_per_doc, rng):
    """Up to max_per_doc RANDOM scorable sentences per document.
    Used for every sample in this script so the AI and human sides are drawn the same
    way (taking the first N sentences would over-represent titles and openings).
    Returns the sentences and, for each one, the position of its source document."""
    sents, doc_pos = [], []
    for i, text in enumerate(texts):
        s = [x for x in split_sentences(text) if is_scorable(x)]
        if len(s) > max_per_doc:                         # cap so long essays don't dominate
            s = [s[j] for j in rng.choice(len(s), max_per_doc, replace=False)]
        sents += s
        doc_pos += [i] * len(s)
    return sents, np.array(doc_pos, dtype=int)


def to_features(sents):
    return np.array([sentence_features(s) for s in sents], dtype=np.float32)


def standardise(X, scaler):
    return np.clip(scaler.transform(X), -5, 5)           # clip outliers at +/-5 sd


def choose_k(bics, frac):
    """Elbow rule: the smallest k that already gets within `frac` of the total BIC
    improvement available between the smallest k and the best k. Taking the plain
    minimum tends to pick the largest k on big datasets, which gives clusters too
    fine-grained to describe."""
    ks = sorted(bics)
    best = min(bics.values())
    total = bics[ks[0]] - best
    if total <= 0:
        return ks[0]
    for kk in ks:
        if bics[kk] - best <= frac * total:
            return kk


# ---------------------------------------------------------------- 1. AI sentences from train
train = pd.read_csv(a.train)
ai_docs = train[train['is_ai'] == 1]
ai_docs = ai_docs.sample(min(a.n_docs, len(ai_docs)), random_state=3)
texts, _ = sample_sentences(ai_docs['text'], a.max_sent_per_doc, rng)
X = to_features(texts)
print(f'{len(X):,} AI sentences from {len(ai_docs):,} documents')

# ---------------------------------------------------------------- 2. standardise
scaler = StandardScaler().fit(X)
Z = standardise(X, scaler)

# ---------------------------------------------------------------- 3. choose k (BIC, lower = better)
k = a.k
if k is None:
    sub = Z[rng.choice(len(Z), min(40000, len(Z)), replace=False)]
    bics = {kk: GaussianMixture(kk, covariance_type='full', reg_covar=1e-4,
                                random_state=0).fit(sub).bic(sub)
            for kk in range(2, a.kmax + 1)}
    for kk, b in bics.items():
        print(f'  k={kk:2d}  BIC={b:,.0f}')
    pd.Series(bics, name='bic').rename_axis('k').to_csv(f'{a.results_dir}/gmm_bic.csv')
    k = choose_k(bics, a.elbow)
    print(f'elbow rule picked k={k} (plain minimum would be k={min(bics, key=bics.get)}); '
          f're-run with --k {k} to fix it')
print('using k =', k)

gmm = GaussianMixture(k, covariance_type='full', reg_covar=1e-4, n_init=2, random_state=0).fit(Z)
post = gmm.predict_proba(Z)
hard = post.argmax(axis=1)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
joblib.dump({'scaler': scaler, 'gmm': gmm, 'k': k}, a.out)
print('saved', a.out)

# ---------------------------------------------------------------- 4. describe each cluster
# Compare each cluster's average feature values with the overall AI average (z-scores),

prof = pd.DataFrame(X, columns=FEATURE_NAMES)
mu, sd = prof.mean(), prof.std()
prof['cluster'] = hard
# "missing" signals: share of sentences where a feature is exactly zero
zero_feats = ['first_person_ratio', 'comma_per_word', 'long_word_ratio']
overall_zero = (prof[zero_feats] == 0).mean()

rows, examples = [], []
for c, g in prof.groupby('cluster'):
    z = (g[FEATURE_NAMES].mean() - mu) / sd
    top = z.sort_values(key=abs, ascending=False).head(3)
    zero_share = (g[zero_feats] == 0).mean()
    print(f'\nCluster {c}: n={len(g):,} ({len(g) / len(prof):.1%})')
    print('  top features (z vs AI average):', top.round(2).to_dict())
    print('  share of sentences with value 0 (cluster vs overall):',
          {f: f'{zero_share[f]:.0%} vs {overall_zero[f]:.0%}' for f in zero_feats})
    # most typical members = highest posterior probability for this cluster
    idx = g.index[np.argsort(-post[g.index, c])][:3]
    for i in idx:
        print(f'   e.g. ({post[i, c]:.2f}) {texts[i][:200]}')
        examples.append({'cluster': c, 'posterior': post[i, c], 'sentence': texts[i]})
    rows.append({'cluster': c, 'n': len(g), 'share': len(g) / len(prof),
                 **{f'mean_{f}': g[f].mean() for f in FEATURE_NAMES},
                 **{f'z_{f}': z[f] for f in FEATURE_NAMES},
                 **{f'zero_share_{f}': zero_share[f] for f in zero_feats}})
pd.DataFrame(rows).round(4).to_csv(f'{a.results_dir}/gmm_cluster_profile.csv', index=False)
pd.DataFrame(examples).to_csv(f'{a.results_dir}/gmm_cluster_examples.csv', index=False)

# ---------------------------------------------------------------- 5. generator composition
# Only the DAIGT rows carry a real generator name; every AI_Human row is just 'AI_Human',
# so including them would make every cluster look ~100% the same.
named = train[(train['is_ai'] == 1) & (train['generator'] != 'AI_Human')]
if len(named):
    ns, pos = sample_sentences(named['text'], a.max_sent_per_doc, rng)
    gen = named['generator'].to_numpy()[pos]
    hc = gmm.predict(standardise(to_features(ns), scaler))
    print(f'\nWhere each named generator lands ({len(ns):,} sentences, '
          f'{len(named):,} documents; columns sum to 1):')
    tab = pd.crosstab(pd.Series(hc, name='cluster'), pd.Series(gen, name='generator'),
                      normalize='columns').round(2)
    tab['all AI (fit sample)'] = pd.Series(hard).value_counts(normalize=True).round(2)
    print(tab)
    tab.to_csv(f'{a.results_dir}/gmm_generator_composition.csv')
else:
    print('\nNo AI rows with a named generator in train; skipping generator composition.')

# ---------------------------------------------------------------- 6. checks on VALIDATION data
# The fitted clusters are 100% AI by construction. To check the clusters are real, we
# place unseen AI and human sentences from val into them, sampled the same way.
val = pd.read_csv(a.val)
clf = joblib.load(a.classifier) if os.path.exists(a.classifier) else None
if clf is None:
    print(f'\n{a.classifier} not found: run classifier.py first for the P(AI) check.')

assigned = {}
for label, name in [(1, 'AI'), (0, 'human')]:
    d = val[val['is_ai'] == label]
    d = d.sample(min(a.n_check_docs, len(d)), random_state=9)
    s, _ = sample_sentences(d['text'], a.max_sent_per_doc, rng)
    F = to_features(s)
    c = gmm.predict(standardise(F, scaler))
    p = clf.predict_proba(F)[:, 1] if clf is not None else np.full(len(F), np.nan)
    assigned[name] = (c, p)

counts = pd.DataFrame({n: pd.Series(c).value_counts() for n, (c, _) in assigned.items()})
counts = counts.reindex(range(k)).fillna(0).astype(int)
check = pd.DataFrame({
    'AI share of AI sents': counts['AI'] / counts['AI'].sum(),
    'human share of human sents': counts['human'] / counts['human'].sum(),
    'label composition (% AI)': counts['AI'] / counts.sum(axis=1),
})
check['human/AI ratio'] = check['human share of human sents'] / check['AI share of AI sents']
if clf is not None:
    for n, (c, p) in assigned.items():
        check[f'mean P(AI), {n} sents'] = pd.Series(p).groupby(c).mean()
print('\nValidation check (unseen sentences placed into the AI-fitted clusters):')
print(pd.concat([counts.add_prefix('n '), check], axis=1).round(3))
print('Note: label composition depends on how many AI vs human sentences were sampled;\n'
      'the human/AI ratio is the fairer comparison (1 = no difference, <1 = more AI-typical).')
pd.concat([counts.add_prefix('n_'), check], axis=1).round(4).to_csv(
    f'{a.results_dir}/gmm_validation_check.csv')
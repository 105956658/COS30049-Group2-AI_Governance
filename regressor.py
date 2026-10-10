"""Model 3 - RandomForestRegressor: document-level score + uncertainty range.

Fit on the VALIDATION split: the classifier and GMM never saw those documents, so
their sentence outputs are honest (not over-confident, as they would be on train data).
Evaluate on the untouched test split and the external 2026 holdout.

The range reported for each document is the 5th-95th percentile of the individual trees'
predictions. It shows how much the trees disagree (model uncertainty). It is NOT a
calibrated 90% confidence interval.

"""
import argparse, os
import joblib, numpy as np, pandas as pd
pd.set_option('display.width', 200)          
pd.set_option('display.max_columns', 40)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error
from features import doc_feature_names
from pipeline import build_doc_matrix, predict_with_interval

ap = argparse.ArgumentParser()
ap.add_argument('--fit', default='data/splits/val.csv')
ap.add_argument('--evals', nargs='+', default=['data/splits/test.csv',
                                               'data/splits/holdout_2026.csv'])
ap.add_argument('--n-fit-docs', type=int, default=20000)
ap.add_argument('--n-eval-docs', type=int, default=10000)
ap.add_argument('--n-errors', type=int, default=15)   # FPs and FNs saved per eval set
ap.add_argument('--model-dir', default='models')
ap.add_argument('--results-dir', default='results')
a = ap.parse_args()
os.makedirs(a.results_dir, exist_ok=True)

clf = joblib.load(f'{a.model_dir}/classifier.joblib')
gb = joblib.load(f'{a.model_dir}/gmm.joblib')
k = gb['k']
names = doc_feature_names(k)

# ---------------------------------------------------------------- fit
df = pd.read_csv(a.fit)
df = df.sample(min(a.n_fit_docs, len(df)), random_state=4).reset_index(drop=True)
X, keep = build_doc_matrix(df['text'], clf, gb)
y = df['is_ai'].to_numpy()[keep].astype(float)
print(f'fitting on {len(y):,} documents, {X.shape[1]} features')

rfr = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                            n_jobs=-1, random_state=0).fit(X, y)
imp = pd.Series(rfr.feature_importances_, index=names).sort_values(ascending=False)
print('Feature importances:')
print(imp.round(3).to_string())
imp.round(4).to_csv(f'{a.results_dir}/regressor_feature_importance.csv', header=['importance'])

# ---------------------------------------------------------------- evaluate + error analysis
show_cols = ['n_sentences', 'mean_sent_len', 'sent_len_cv', 'mean_p', 'std_p',
             'frac_p_gt_0.8', 'mean_cluster_entropy'] + [f'mean_cluster_{i}' for i in range(k)]
summary = []
for path in a.evals:
    name = os.path.splitext(os.path.basename(path))[0]
    d = pd.read_csv(path)
    d = d.sample(min(a.n_eval_docs, len(d)), random_state=5).reset_index(drop=True)
    Xe, kp = build_doc_matrix(d['text'], clf, gb)
    dd = d.iloc[kp].reset_index(drop=True)              # rows that had scorable sentences
    ye = dd['is_ai'].to_numpy().astype(int)
    score, lo, hi = predict_with_interval(rfr, Xe)
    pred = (score > .5).astype(int)
    feat = pd.DataFrame(Xe, columns=names)

    res = {'set': name, 'n': len(ye),
           'auc': roc_auc_score(ye, score), 'acc@0.5': (pred == ye).mean(),
           'mae': mean_absolute_error(ye, score),
           'baseline_mean_p_auc': roc_auc_score(ye, feat['mean_p']),
           'tree_band_width': np.mean(hi - lo)}
    summary.append(res)
    print(f'\n[{name}] n={len(ye):,}  (skipped {len(d) - len(kp)} docs with no scorable sentences)')
    print(f"  RFR score      AUC={res['auc']:.3f}  acc@0.5={res['acc@0.5']:.3f}  MAE={res['mae']:.3f}")
    print(f"  baseline mean sentence prob  AUC={res['baseline_mean_p_auc']:.3f}")
    print(f"  tree spread (5th-95th pct)  mean width={res['tree_band_width']:.3f}")

    # accuracy by document length: training essays are long, the 2026 holdout is short
    bucket = pd.cut(feat['n_sentences'], [0, 2, 5, 10, 20, np.inf],
                    labels=['1-2', '3-5', '6-10', '11-20', '20+'])
    by_len = (pd.DataFrame({'bucket': bucket, 'correct': pred == ye, 'is_ai': ye})
              .groupby('bucket', observed=True)
              .agg(n=('correct', 'size'), accuracy=('correct', 'mean'), ai_share=('is_ai', 'mean')))
    print('  accuracy by number of scorable sentences:')
    print('    ' + by_len.round(3).to_string().replace('\n', '\n    '))

    # ---- error analysis
    out = pd.concat([dd[['text']], feat[show_cols]], axis=1)
    out['is_ai'], out['score'], out['low'], out['high'] = ye, score, lo, hi
    out['band_width'] = hi - lo
    out['outcome'] = np.select([(pred == 1) & (ye == 1), (pred == 0) & (ye == 0),
                                (pred == 1) & (ye == 0)], ['TP', 'TN', 'FP'], 'FN')
    print('  outcomes:', out['outcome'].value_counts().to_dict())
    # what do errors have in common? compare average features across the four outcomes
    group_means = out.groupby('outcome')[show_cols + ['score', 'band_width']].mean()
    print('  average document features by outcome (compare FP with TN, FN with TP):')
    print('    ' + group_means.T.round(3).to_string().replace('\n', '\n    '))
    group_means.round(4).to_csv(f'{a.results_dir}/outcome_means_{name}.csv')

    # most confident mistakes: highest-scoring humans (FP), lowest-scoring AI texts (FN)
    fp = out[out['outcome'] == 'FP'].nlargest(a.n_errors, 'score')
    fn = out[out['outcome'] == 'FN'].nsmallest(a.n_errors, 'score')
    errs = pd.concat([fp, fn])
    errs.to_csv(f'{a.results_dir}/errors_{name}.csv', index=False)
    print(f'  saved {len(fp)} FPs and {len(fn)} FNs to {a.results_dir}/errors_{name}.csv')
    for tag, g in [('FP', fp), ('FN', fn)]:
        for _, r in g.head(2).iterrows():
            print(f"    {tag} score={r['score']:.2f}: {r['text'][:150].replace(chr(10), ' ')}")

pd.DataFrame(summary).round(4).to_csv(f'{a.results_dir}/regressor_results.csv', index=False)

os.makedirs(a.model_dir, exist_ok=True)
joblib.dump({'model': rfr, 'k': k, 'feature_names': names}, f'{a.model_dir}/regressor.joblib')
print('\nsaved', f'{a.model_dir}/regressor.joblib')
"""Adversarial validation: can a model tell training rows from test rows?
AUC near 0.5: same distribution, plain CV is trustworthy. Well above 0.5: a domain shift, so CV scores will be too
optimistic; validate on test-like data (the most test-like training rows, saved below, or simulated test noise)."""
from kit.common import train, test
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import comp

X = pd.concat([comp.features(train()), comp.features(test())], ignore_index=True)
if not X.shape[1]:
    raise SystemExit('comp.features() returns no columns: add features first (image tasks: compare simple image statistics)')
is_test = np.r_[np.zeros(len(train())), np.ones(len(test()))]
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1)
p = cross_val_predict(m, X, is_test, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
auc = roc_auc_score(is_test, p)
print(f'train-vs-test AUC {auc:.3f}', '-> no meaningful shift' if auc < 0.6 else '-> SHIFT: build a test-like validation')
imp = pd.Series(m.fit(X, is_test).feature_importances_, X.columns).sort_values(ascending=False)
print('features that give the test set away:\n' + imp.head(15).to_string())
np.save('preds/adv_testlikeness_train.npy', p[:len(train())])  # high = training row that looks like test data

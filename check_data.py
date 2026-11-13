"""Day-0 look at the data: sizes, submission format, target balance, missing values, and train->test feature shifts."""
from kit.common import train, test
import pandas as pd
import comp

tr, te, ss = train(), test(), comp.load_sample()
print(f'train {tr.shape}  test {te.shape}  sample_submission {ss.shape}, columns {list(ss.columns)}')
if comp.ID in te:
    print('sample ids == test ids:', set(ss[comp.ID]) == set(te[comp.ID]))
if comp.TASK == 'regression':
    print('target:', tr[comp.TARGET].describe().round(3).to_dict())
else:
    print('target balance (train):\n' + tr[comp.TARGET].value_counts(normalize=True).round(3).to_string())
miss = pd.DataFrame({'train': tr.isna().mean(), 'test': te.isna().mean()})
miss = miss[(miss.train > 0) | (miss.test > 0)]
print('missing values:\n' + (miss.round(3).to_string() if len(miss) else '  none'))
F, Ft = comp.features(tr), comp.features(te)
if F.shape[1]:
    shift = ((Ft.mean() - F.mean()) / (F.std() + 1e-9)).abs().sort_values(ascending=False)
    print('largest train->test mean shifts, in train standard deviations:\n' + shift.head(10).round(2).to_string())

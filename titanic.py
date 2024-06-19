# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pickle
import gc

# 分布確認
import ydata_profiling as pdp
# 可視化
import matplotlib.pyplot as plt
# 前処理
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
# モデリング
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore")

# matplotlibの日本語表示
!pip install japanize-matplotlib
import japanize_matplotlib
%matplotlib inline

df_train = pd.read_csv("../input/titanic/train.csv")
df_train.head(5)

print(df_train.shape)
print(len(df_train))
print(len(df_train.columns))

df_train.info()

df_train.isnull().sum()

# 試しにPclass, Fareを説明変数に使う

x_train, y_train, id_train = df_train[["Pclass", "Fare"]], df_train[["Survived"]], df_train[["PassengerId"]]
print(x_train.shape, y_train.shape, id_train.shape)

# ホールドアルト法

x_tr, x_va, y_tr, y_va = train_test_split(x_train,
                                         y_train,
                                         test_size=0.2,
                                         shuffle=True,
                                         stratify=y_train,
                                         random_state=123)

print(x_tr.shape, y_tr.shape)
print(x_va.shape, y_va.shape)

# クロスバリデーション

n_splits = 5
cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(x_train, y_train))

for nfold in np.arange(n_splits):
    print("-"*20, nfold, "-"*20)
    idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
    x_tr, y_tr = x_train.loc[idx_tr, :], y_train.loc[idx_tr, :]
    x_va, y_va = x_train.loc[idx_va, :], y_train.loc[idx_va, :]
    print(x_tr.shape, y_tr.shape)
    print(x_va.shape, y_va.shape)

# ハイパーパラメータ

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.1,
    'num_leaves':16,
    'n_estimators': 100000,
    "random_state": 123,
    "importance_type": "gain",
}

model = lgb.LGBMClassifier(**params)
model.fit(x_tr,
         y_tr,
         eval_set=[(x_tr, y_tr), (x_va, y_va)],
         early_stopping_rounds=100,
         verbose=10,
         )

# 精度の評価

y_tr_pred = model.predict(x_tr)
y_va_pred = model.predict(x_va)

metric_tr = accuracy_score(y_tr, y_tr_pred)
metric_va = accuracy_score(y_va, y_va_pred)

print("[accuracy] tr: {:.2f}, va: {:.2f}".format(metric_tr, metric_va))

# 説明変数の重要度の算出

imp = pd.DataFrame({"col":x_train.columns, "imp": model.feature_importances_})
imp.sort_values("imp", ascending=False, ignore_index=True)

# クロスバリデーションの場合

metrics = []
imp = pd.DataFrame()

n_splits = 5
cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(x_train, y_train))

for nfold in np.arange(n_splits):
    print("-"*20, nfold, "-"*20)
    idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
    x_tr, y_tr = x_train.loc[idx_tr, :], y_train.loc[idx_tr, :]
    x_va, y_va = x_train.loc[idx_va, :], y_train.loc[idx_va, :]
    print(x_tr.shape, y_tr.shape)
    print(x_va.shape, y_va.shape)
    model = lgb.LGBMClassifier(**params)
    model.fit(x_tr,
         y_tr,
         eval_set=[(x_tr, y_tr), (x_va, y_va)],
         early_stopping_rounds=100,
         verbose=100,
         )
    y_tr_pred = model.predict(x_tr)
    y_va_pred = model.predict(x_va)
    metric_tr = accuracy_score(y_tr, y_tr_pred)
    metric_va = accuracy_score(y_va, y_va_pred)
    print("[accuracy] tr: {:.2f}, va: {:.2f}".format(metric_tr, metric_va))
    metrics.append([nfold, metric_tr, metric_va])
    
    _imp = pd.DataFrame({"col":x_train.columns, "imp":model.feature_importances_, "nfold":nfold})
    imp = pd.concat([imp, _imp], axis=0, ignore_index=True)

imp = imp.groupby("col")["imp"].agg(["mean", "std"])
imp.columns = ["imp", "imp_std"]
imp = imp.reset_index(drop=False)

imp.sort_values("imp", ascending=False, ignore_index=True)

# 推論用データセットの作成
df_test = pd.read_csv("../input/titanic/test.csv")
x_test = df_test[["Pclass", "Fare"]]
id_test = df_test[["PassengerId"]]

# 学習モデルによる推論
y_test_pred = model.predict(x_test)

# 提出用ファイルの作成
df_submit = pd.DataFrame({"PassengerId":id_test["PassengerId"], "Survived":y_test_pred})
display(df_submit.head(5))
df_submit.to_csv("submission_baseline.csv", index=None)

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

data_train = pd.read_csv('/kaggle/input/titanic/train.csv')
data_test = pd.read_csv('/kaggle/input/titanic/test.csv')
data_gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

data_all = pd.concat([data_train, data_test], sort=False)
data_all

# 男性，女性に特徴量を分ける
data_all['Sex_male'] = data_all['Sex'].replace({'male': 1, 'female': 0})
data_all['Sex_female'] = data_all['Sex'].replace({'male': 0, 'female': 1})
data_all['Sex'] = data_all['Sex'].replace({'male': 0, 'female': 1})
data_all.head(10)

# 単身者が多いので別の特徴量として分離
data_all['Alone'] = 0
data_all.loc[data_all['Families'] == 1, 'Alone'] = 1
data_all.head(20)

# 年齢　欠損値は mean +- std の範囲のランダム値で埋める
ave = data_all['Age'].mean()
std = data_all['Age'].std()
data_all['Age'].fillna(np.random.randint(ave - std, ave + std), inplace=True)
data_all.isnull().sum()

data_all['Embarked'].fillna('S', inplace=True)
data_all['Embarked'] = data_all['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
data_all.head(10)

categorical_features = ['Sex', 'Embarked', 'Pclass']
# Fare 欠損値補完
data_all['Fare'].fillna(np.mean(data_all['Fare']), inplace=True)
# 影響の小さい特徴量を削除
drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data_all.drop(drop_columns, axis=1, inplace=True)
# 結合データを再度分離
data_train = data_all[:len(data_train)]
data_test = data_all[len(data_train):]
# 特徴量と目的変数を分離
y_data_train = data_train['Survived']
X_data_train = data_train.drop('Survived', axis=1)
X_data_test = data_test.drop('Survived', axis=1)
# Cross Validation 交差検証
from sklearn.model_selection import KFold
import lightgbm as lgb

y_data_preds = []
# 各分割されたデータで学習した各モデルを格納
models = []
# 各検証用データの予測値を格納
oof_data_train = np.zeros((len(X_data_train),))
# データをどのように分割するか
cv = KFold(n_splits=5, shuffle=True, random_state=0)

param = {
    'objective' : 'binary',
    'max_bin' : 300,
    'learning_rate' : 0.05,
    'num_leaves' : 40
}

for fold_id, (train_index, valid_index) in enumerate(cv.split(X_data_train)):
    X_tr = X_data_train.loc[train_index, :]
    X_val = X_data_train.loc[valid_index, :]
    y_tr = y_data_train[train_index]
    y_val = y_data_train[valid_index]
    
    # LightGBM
    lgb_data_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_features)
    lgb_data_eval = lgb.Dataset(X_val, y_val, reference=lgb_data_train, categorical_feature=categorical_features)
    
    model = lgb.train(param, lgb_data_train, valid_sets=lgb_data_eval,
                      # 学習回数ごとに表示する 過学習を止める
                      callbacks=[lgb.log_evaluation(period=10), lgb.early_stopping(10)],
                      # 学習回数
                      num_boost_round=1000
                     )
    # best_iteration 最も正解率が高い学習結果が利用
    oof_data_train[valid_index] = model.predict(X_val, num_iteration=model.best_iteration)
    y_data_pred = model.predict(X_data_test, num_iteration=model.best_iteration)

    # 予測結果とモデルを格納
    y_data_preds.append(y_data_pred)
    models.append(model)

# 各モデルのベストスコアと平均
scores = [
    m.best_score['valid_0']['binary_logloss'] for m in models
]

score = sum(scores) / len(scores)

display(scores)
display(score)

# 検証用データに対する正解率
from sklearn.metrics import accuracy_score

# 0.5 < なら 1, それ以外は 0
y_data_pred_oof = (oof_data_train > 0.5).astype(int)
accuracy_score(y_data_train, y_data_pred_oof)

# 予測
y_pred_submit = sum(y_data_preds) / len(y_data_preds)
y_pred_submit = (y_pred_submit > 0.5).astype(int)
y_pred_submit

submit = data_gender_submission
# map 関数でint型にキャスト
submit['Survived'] = list(map(int, y_pred_submit))
submit.to_csv('kfold_submit.csv', index=False)

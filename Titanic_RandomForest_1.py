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

import matplotlib.pyplot as plt
import seaborn as sns

data_train = pd.read_csv('/kaggle/input/titanic/train.csv')
data_test = pd.read_csv('/kaggle/input/titanic/test.csv')
data_gender_submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

# EDA 探索的データ分析
data_train.describe()
from ydata_profiling import ProfileReport
profile = ProfileReport(data_train, title="Profiling Report")
display(profile)

# Feature Engineering 特徴量エンジニアリング
data_all = pd.concat([data_train, data_test], sort=False)
data_all.isnull().sum()
data_all['Sex'].replace({'male': 0, 'female': 1}, inplace=True)
data_all.head(10)
data_all['Embarked'].fillna('S', inplace=True)
data_all['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
data_all.head(10)
data_all['Fare'].fillna(np.mean(data_all['Fare']), inplace=True)
data_all['Age'].fillna(np.mean(data_all['Age']), inplace=True)

# 影響の小さい特徴量を削除
drop_columns = ['PassengerId', 'Name', 'Parch', 'SibSp', 'Ticket', 'Cabin']
data_all.drop(drop_columns, axis=1, inplace=True)

# 結合データを再度分離
print(len(data_train))
data_train = data_all[:len(data_train)]
data_test = data_all[len(data_train):]

# 特徴量と目的変数を分離
y_data_train = data_train['Survived']
X_data_train = data_train.drop('Survived', axis=1)
X_data_test = data_test.drop('Survived', axis=1)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

# 学習
clf.fit(X_data_train, y_data_train)

# 予測 デフォルトの閾値は0.5
y_data_pred = clf.predict(X_data_test)
y_data_pred

# Submit 提出
submit = data_gender_submission
# map 関数でint型にキャスト
submit['Survived'] = list(map(int, y_data_pred))
submit.to_csv('randomForest_submit.csv', index=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

DATA_DIR = '../datasets/bike-sharing-demand'
train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))

train['datetime'] = pd.to_datetime(train['datetime'])

train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['dayofweek'] = train['datetime'].dt.dayofweek
train['quarter'] = train['datetime'].dt.quarter
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second


test['datetime'] = pd.to_datetime(test['datetime'])

test['year'] = test['datetime'].dt.year
test['month'] = test['datetime'].dt.month
test['day'] = test['datetime'].dt.day
test['dayofweek'] = test['datetime'].dt.dayofweek
test['quarter'] = test['datetime'].dt.quarter
test['hour'] = test['datetime'].dt.hour
test['minute'] = test['datetime'].dt.minute
test['second'] = test['datetime'].dt.second

sns.countplot(x='year', data=train)
plt.show()

sns.boxplot(x='dayofweek', y='count', data=train)

train.groupby('month')['temp'].mean().plot(kind='bar')
sns.barplot(x='month', y='temp', hue='year', data=train)
sns.barplot(x='month', y='count', hue='year', data=train)
plt.show()

train = train.drop(['minute', 'second'], axis=1)
test= test.drop(['minute', 'second'], axis=1)

train['weekend'] = train['dayofweek'].apply(lambda x: int(x in [5, 6]))
test['weekend'] = test['dayofweek'].apply(lambda x: int(x in [5, 6]))

fig, axes = plt.subplots(2, 1)
fig.set_size_inches(10, 8)
sns.boxplot(x='dayofweek', y='count', data=train.loc[train['year'] == 2011], ax=axes[0])
sns.boxplot(x='dayofweek', y='count', data=train.loc[train['year'] == 2012], ax=axes[1])
plt.show()

fig, axes = plt.subplots(3, 1)
fig.set_size_inches(10, 8)
sns.lineplot(x='hour', y='count', data=train.loc[train['year']==2011], ax=axes[0])
sns.lineplot(x='hour', y='casual', data=train.loc[train['year']==2011], ax=axes[1])
sns.lineplot(x='hour', y='registered', data=train.loc[train['year']==2011], ax=axes[2])
plt.show()

plt.figure(figsize=(10, 6))
sns.pointplot(x='hour', y='count', hue='dayofweek', data=train.loc[train['year']==2011])
plt.show()

plt.figure(figsize=(10, 6))
sns.pointplot(x='hour', y='count', hue='workingday', data=train)
plt.show()

plt.figure(figsize=(10, 6))
sns.pointplot(x='hour', y='count', hue='season', data=train)
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(x='hour', y='casual', hue='year', data=train)
sns.lineplot(x='hour', y='registered', hue='year', data=train)
plt.show()

train[['temp','atemp','weather','count','casual','registered']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(abs(train.corr()), annot=True, cmap='Greens')
plt.show()

train.corr().style.background_gradient(cmap='coolwarm')

from sklearn.decomposition import PCA
pca = PCA(n_components=1, random_state=123)
train_temp = pca.fit_transform(train[['temp', 'atemp']])
test_temp = pca.transform(test[['temp', 'atemp']])

assert train_temp[10].round(3) == 6.063
assert test_temp[10].round(3) == 15.526

train = train.drop(['temp', 'atemp'], axis=1)
test = test.drop(['temp', 'atemp'], axis=1)

train['temp_pca'] = train_temp
test['temp_pca'] = test_temp

pd.DataFrame(abs(train.corr())['count'].sort_values()).plot.barh(figsize=(10,6))
pd.DataFrame(abs(train.corr())['casual'].sort_values()).plot.barh(figsize=(10,6))

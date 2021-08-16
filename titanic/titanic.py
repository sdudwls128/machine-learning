# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()
test.head()
train.shape
test.shape
train.info
test.info

def pie_chart(feature):
    feature_ratio = train[feature].value_counts(sort=False)
    feature_size = feature_ratio.size
    feature_index = feature_ratio.index
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()

    plt.plot(aspect='auto')
    plt.pie(feature_ratio, labels=feature_index, autopct='%1.1f%%')
    plt.title(feature + '\'s ratio in total')
    plt.show()

    for i, index in enumerate(feature_index):
        plt.subplot(1, feature_size + 1, i + 1, aspect='equal')
        plt.pie([survived[index], dead[index]], labels=['Survived', 'Dead'], autopct='%1.1f%%')
        plt.title(str(index) + '\'s ratio')

    plt.show()

def bar_chart(feature):
    survived = train[train['Survived'] == 1][feature].value_counts()
    dead = train[train['Survived'] == 0][feature].value_counts()
    df = pd.DataFrame([survived, dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked=True, figsize=(10, 5))

train_and_test = [train, test]

for dataset in train_and_test:
#     print(dataset['Passenger'])
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.')
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Countess', 'Don','Dona', 'Dr', 
                                                 'Jonkheer', 'Lady','Major', 'Rev', 'Sir'], 'Other')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

for dataset in train_and_test:
    dataset['Title'] = dataset['Title'].astype(str)

# Sex Freutre를 String Data로 변경
for dataset in train_and_test:
    dataset['Sex'] = dataset['Sex'].astype(str)
    
# 배를 탑승한 선착장을 나타내는 Embarked 처리 
train.Embarked.value_counts(dropna=False)

for dataset in train_and_test:
    dataset['Embarked'] = dataset['Embarked'].fillna('S') # Embarked feature 'S'가 대부분. Nan data 2개도 'S'에 속할 확률이 높으므로 'S'로 넣음
    dataset['Embarked'] = dataset['Embarked'].astype(str) # String Data로 변형

# Age Nan 값을 평균 나이로 채움. pd.cut으로 같은 길이의 구간을 가지는 다섯 개의 AgeBand 그룹을 만듦
for dataset  in train_and_test:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
    train['AgeBand'] = pd.cut(train['Age'], 5)
print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
    
for dataset in train_and_test:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = dataset['Age'].map({0: 'Child', 1: 'Young', 2:'Middle', 3:'Prime', 4:'Old'}).astype(str)

# Pclass와 Fare가 어느 정도 연관성이 있는 것 같아 Fare의 Nan값을 해당 Pclass사람들의 평균 Fare값으로 넣음
print(train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).mean())
print("")
print(test[test['Fare'].isnull()]['Pclass'])
for dataset in train_and_test:
    dataset['Fare'] = dataset['Fare'].fillna(13.675)

# Age에서 한 것처럼 Fare도 Binning
for dataset in train_and_test:
    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare']   = 3
    dataset.loc[ dataset['Fare'] > 39.688, 'Fare'] = 4
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    
# SibSp와 parch Feature를 합쳐 Family라는 Feature로 만듦
# for dataset in train_and_test:
#     dataset["Family"]
# train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# train.head()
# pd.crosstab(train['Title'], train['Sex'])

# -















import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train.head()
test.head()

# print('train data shape: ', train.shape)
# print('test data shape: ', test.shape)
# print('----------[train information]----------')
# print(train.info())
# print(test.info())
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

pie_chart('Pclass')
# bar_chart("SibSp")
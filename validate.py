#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, linear_model, metrics
from joblib import load


# In[14]:


# чтение данных: валидационная выборка
df = pd.read_csv('val.tsv',sep='\t')
libs = df.libs
df.head()


# In[15]:


# размер валидационной выборки
df.shape


# In[16]:


# наличие пропусков в данных
df.isnull().values.any()


# In[22]:


# предобработка данных
# преобразуем исходные данные в матрицу

df_train = pd.read_csv('train.tsv',sep='\t')
libs_train = df_train.libs
vectorizer = CountVectorizer(tokenizer = lambda x: x.split(','))
vectorizer.fit_transform(libs_train)

V = vectorizer.transform(libs)
X_val = V.toarray()
y_val = df['is_virus'].to_numpy()


# In[7]:


# оценка качества модели на валидационной выборке
classifier = load('SGD_Model.joblib')
prediction_val = classifier.predict(X_val)
metrics.classification_report(y_val, prediction_val)


# In[8]:


# вывод матрицы ошибок валидационной выборки
metrics.confusion_matrix(y_val, prediction_val)


# In[9]:


tn, fp, fn, tp = metrics.confusion_matrix(y_val, classifier.predict(X_val)).ravel()
accuracy = metrics.accuracy_score(y_val, classifier.predict(X_val))
precision = metrics.precision_score(y_val, classifier.predict(X_val))
recall = metrics.recall_score(y_val, classifier.predict(X_val))
f1 = metrics.f1_score(y_val, classifier.predict(X_val))


# In[10]:


lines = []
lines.append('True positive: {}'.format(tp))
lines.append('False positive {}'.format(fp))
lines.append('False negative: {}'.format(fn))
lines.append('True negative: {}'.format(tn))
lines.append('Accuracy: {:.4}'.format(accuracy))
lines.append('Precision: {:.4}'.format(precision))
lines.append('Recall: {:.4}'.format(recall))
lines.append('F1: {:.4}'.format(f1))


# In[11]:


with open('validation.txt','w') as file:
    file.writelines(line + '\n' for line in lines)


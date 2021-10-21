#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, linear_model, metrics
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
from joblib import dump


# In[2]:


# чтение данных
df = pd.read_csv('train.tsv',sep='\t')
libs = df.libs
df.head()


# In[3]:


# размер обучающей выборки
df.shape


# In[4]:


# наличие пропусков в данных
df.isnull().values.any()


# In[8]:


# предобработка данных
# преобразуем исходные данные в матрицу
vectorizer = CountVectorizer(tokenizer = lambda x: x.split(','))
V = vectorizer.fit_transform(libs)

X_train = V.toarray()
y_train = df['is_virus'].to_numpy()


# In[19]:


# проверим баланс классов в обучающей выборке
# доля класса 1
round(len(y_train[y_train==1])/len(y_train),2)


# In[20]:


# доля класса 0
round(len(y_train[y_train==0])/len(y_train),2)


# ## Создание и обучение модели

# In[10]:


# задание модели
classifier = linear_model.SGDClassifier(loss='log',random_state=0,max_iter=200)


# In[11]:


# обучение модели
classifier.fit(X_train,y_train)


# In[12]:


# вывод средней абсолютной ошибки модели на обучающей выборке
metrics.mean_absolute_error(y_train, classifier.predict(X_train))


# In[22]:


# вывод среднего значения целевой переменной
round(np.mean(y_train),2)


# In[17]:


dump(classifier, 'SGD_Model.joblib')


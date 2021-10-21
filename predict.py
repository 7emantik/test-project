#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, linear_model, metrics
from joblib import load


# In[2]:


# чтение файла
df = pd.read_csv('test.tsv',sep='\t')
libs = df.libs
df.head()


# In[3]:


# размер тестовой выборки
df.shape


# In[14]:


# предобработка данных
# преобразуем исходные данные в матрицу

df_train = pd.read_csv('train.tsv',sep='\t')
libs_train = df_train.libs
vectorizer = CountVectorizer(tokenizer = lambda x: x.split(','))
vectorizer.fit_transform(libs_train)

V = vectorizer.transform(libs)
X_test = V.toarray()


# In[15]:


# построение прогноза для тестовой выборки

classifier = load('SGD_Model.joblib')
prediction_test = classifier.predict(X_test)
prediction_test[:10]


# In[16]:


with open ('prediction.txt','w') as file:
    file.writelines('prediction' + '\n')
    file.writelines(str(i) + '\n' for i in prediction_test)


# ## Дополнительное задание

# In[17]:


# пояснение к построенному прогнозу модели
predict_proba = classifier.predict_proba(X_test)[:,1]
ind_predict_proba = np.where(predict_proba > 0.5)


# In[18]:


lines = []
for i in range(X_test.shape[0]):
    if i in ind_predict_proba[0]:
        line = 'с вероятностью {:.2%} файл зловреден'.format(predict_proba[i])
        lines.append(line)
    else:
        lines.append('')


# In[19]:


with open ('explain.txt','w') as file:
    file.writelines(line + '\n' for line in lines)


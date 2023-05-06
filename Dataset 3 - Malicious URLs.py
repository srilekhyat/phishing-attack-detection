#!/usr/bin/env python
# coding: utf-8

# # Malicious URLs Dataset Preprocessing

# In[28]:


import pandas as pd
import numpy as np


# In[3]:


# reading dataset 'Malicious-URLs'

df = pd.read_csv('Malicious-URLs.csv')
df.head()


# ### Grouping all malicious categories into a common category

# In[4]:


type_dict = {'phishing': 1, 'defacement': 1, 'malware': 1, 'benign': 0}
df.replace({'type': type_dict}, inplace = True)

# df['label'] = np.where(df['type']=='benign', 0, 1)
df.head()


# In[5]:


df.describe()


# ### Imbalanced data

# In[6]:


# getting count of labels - benign (0) and malicious (1)
df.groupby('type').count()


# In[7]:


import seaborn as sn
import matplotlib.pyplot as plt

count = df.type.value_counts()

plt.figure(figsize=(2,3))
sn.barplot(x=count.index, y=count)
plt.xlabel('Type')
plt.ylabel('Count')


# ### Extracting Special Characters count and important parts of the URL

# In[8]:


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits += 1
    return digits

def char_count(url):
    chars = 0
    for i in url:
        if i.isalpha():
            chars += 1
    return chars

# URL Length
df['url_length'] = df['url'].apply(lambda x: len(str(x)))

# Special characters
feature = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
for a in feature:
    df[a+'_count'] = df['url'].apply(lambda i: i.count(a))
    

# Existence of HTTPS/HTTP
df['https'] = np.where('https' in df['url'], 1, 0)

# Digit count
df['digit_count'] = df['url'].apply(lambda x: digit_count(x))

# Char count
df['char_count'] = df['url'].apply(lambda x: char_count(x))

# Get Tokens
def getTokens(url):
    url = url.lower()
    url = url.replace(' ', '/')
    url = url.replace('.', '/')
    url = url.replace('=', '/')
    url = url.replace('&', '/')
    url = url.replace('?', '/')
    url = url.replace('-', '/')
    url = url.replace('@', '/')
    url = url.replace(':', '/')
    url = url.replace(';', '/')
    url = url.replace('%2f', '/')
    url = url.replace('%', '/')
    url = url.replace('+', '/')
    url = url.replace('$', '/')
    url = url.replace('#', '/')
    url = url.replace('~', '/')
    url = url.replace(',', '/')
    url = url.replace('(', '/')
    url = url.replace(')', '/')
    url = url.replace('[', '/')
    url = url.replace(']', '/')
    url = url.replace('{', '/')
    url = url.replace('}', '/')
    url = url.replace('_', '/')
    url = url.replace('!', '/')
    url = url.replace('*', '/')
    url = url.replace("'", '/')
    url = url.replace('|', '/')
    url = url.replace('>', '/')
    url = url.replace('<', '/')
    url = url.replace('\\', '/')
    urls = url.split('/')
    urls = list(filter(lambda a: a != ' ' and a != '', urls))
    urls = list([str(u) for u in urls])
    return urls

df['url'] = df['url'].apply(lambda x : str(x))
df['url_tokens'] = df['url'].apply(lambda x: getTokens(x))
#df['url_tokens'] = pd.Series(df['url_tokens'], dtype="string")

df.head()


# ### Histogram on URL Length frequency for Benign and Malicious URLs

# In[9]:


df[df.type==0].url_length.plot(
    bins=20, kind='hist', color='blue', 
    label='Benign', alpha=0.6)
df[df.type==1].url_length.plot(bins=20,
    kind='hist', color='red', 
    label='Malicious', alpha=0.4)

plt.legend()
plt.xlabel("URL Length")


# ### Comparing the different special character count features

# In[10]:


x = []
xlim = [10, 20, 87, 51, 42, 6, 231, 37, 6, 5, 54, 15, 9]
for a in feature:
    x.append(a + '_count')
    
count = 0
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15,25))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for i in range(4):
    for j in range(3):
        axes[i,j].hist(df[df.type==0][x[count]], color='blue', alpha=0.6, label='Benign')
        axes[i,j].hist(df[df.type==1][x[count]], color='red', alpha=0.6, label='Malicious')
        axes[i,j].set_xlabel(x[count])
        axes[i,j].set_ylabel('Frequency')
        axes[i,j].set_xlim([0,xlim[count]])
        axes[i,j].legend()
        count += 1
        
axes[4,0].hist(df[df.type==0][x[count]], color='blue', alpha=0.6, label='Benign')
axes[4,0].hist(df[df.type==1][x[count]], color='red', alpha=0.6, label='Malicious')
axes[4,0].set_xlabel(x[count])
axes[4,0].set_ylabel('Frequency')
axes[4,0].set_xlim([0,xlim[count]])
axes[4,0].legend()

fig.delaxes(axes[4,1])
fig.delaxes(axes[4,2])

plt.show()


# ## Inferences
# - Malicious URLs have a greater URL length
# 
# 
# - Malicious URLs tend to have a greater **//** count
# 
# 
# - Malicious URLs tend to have a greater **=** count
# 
# 
# - Benign URLs tend to have a greater **?** count
# 
# 
# - Benign URLs tend to have a greater **-** count

# In[11]:


df[df['url_tokens'].str.len() == 0]


# In[12]:


feature = ['@','?','-','=','.','#','%','+','$','!','*',',','//']
df[df.type==0]['url_tokens'].str.len().plot(
    bins=20, kind='hist', color='blue', 
    label='Benign', alpha=0.6)
df[df.type==1]['url_tokens'].str.len().plot(bins=20,
    kind='hist', color='red', 
    label='Malicious', alpha=0.4)

plt.legend()
plt.xlabel("URL Tokens")


# ### Applying Word2vec

# In[13]:


from gensim.models import Word2Vec

w2vmodel = Word2Vec(vector_size=300, min_count=1, window=5, workers=4)
w2vmodel.build_vocab(df['url_tokens'])
w2vmodel.train(df['url_tokens'], total_examples=w2vmodel.corpus_count, epochs=2)


# In[14]:


count = 0
vectors = []
for word in w2vmodel.wv.key_to_index:
    index = w2vmodel.wv.key_to_index[word]
    vectors.append(w2vmodel.wv[index])

vectors[:3]


# ### Getting the TFIDF Weights

# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ' '.join([u for url in df['url_tokens'] for u in url])
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([corpus])

tfidf_dict = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))


# In[17]:


tfidf_matrix.toarray()


# In[29]:


words = vectorizer.get_feature_names_out()
scores = tfidf_matrix.toarray()[0]
d = {'Words': words, 'TFIDF Scores': scores}
tfidf_dataframe = pd.DataFrame(d)
tfidf_dataframe[tfidf_dataframe['Words']=='html']


# ### Creating the Weighted Vector using TFIDF
# 
# - For each word in the corpus, we multiplied the corresponding word vector with its pre-calculated tfidf score

# In[30]:


tfidf_weighted_vector = {}
errors = []
count = 0
for i in range(len(vectors)):
    try:
        v = vectors[i]
        word = w2vmodel.wv.index_to_key[i]
        tfidf = tfidf_dict[str(word)]
#         weighted_vector[word] = v*tfidf_dict[str(word)]
        tfidf_weighted_vector[word] = np.array(list(map(lambda e: e * tfidf, v)))
    except KeyError:
        tfidf_weighted_vector[word] = np.random.uniform(0,1,300)


# In[31]:


tfidf_weighted_vector['html']


# ### Creating paragraph vectors for each URL
# 
# - We created paragraph vectors by taking the average of every $word_i$ at $index_j$
# ![image info](ParagraphVector.png)

# In[51]:


def create_paragraph_vectors(tokens):
    temp = []
    for i in range(300):
        sum = 0
        for token in tokens:
            try:
                sum += tfidf_weighted_vector[token][i]
            except KeyError as e:
                sum += np.random.uniform(0,1,300)
        temp.append(sum/len(tokens))
    paragraph_vector = np.asarray(temp, dtype=np.float32)
    return paragraph_vector

example_url = ['https', 'www', 'google', 'com']
res = create_paragraph_vectors(example_url)
res


# In[33]:


df['paragraph_vector'] = df['url_tokens'].apply(lambda x: create_paragraph_vectors(x))
df.head()


# ### Generating Paragraph Vectors for every URL

# In[34]:


new_df = df[['url', 'type', 'url_tokens', 'paragraph_vector']]
new_df.head()


# In[35]:


cols = ['v'+str(i) for i in range(300)]
cols


# ### Separating the vector
# 
# - We are separating the vector of 300 dimensions such that the new dataframe will have 300 columns of 1 element each instead of 1 column of 300 dimension vector
# 
# - This is so that we can apply oversampling techniques and produce more data

# In[36]:


import warnings
warnings.filterwarnings('ignore')

new_df[cols] = pd.DataFrame(new_df['paragraph_vector'].to_list(), index=new_df.index)
new_df.head()


# ## Imbalanced Data

# In[37]:


X = new_df.iloc[:, 4:]
Y = new_df['type']


# In[38]:


X.shape


# In[39]:


Y.shape


# ### Splitting the dataset into training and testing set

# In[40]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# ### Logistic Regression

# In[69]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logRegClf = LogisticRegression(random_state=0)
logRegClf.fit(X_train, Y_train)

Y_pred = logRegClf.predict(X_test)
score = accuracy_score(Y_test, Y_pred)
print(score)
print(classification_report(Y_test, Y_pred))


# In[77]:


cmLR = confusion_matrix(Y_test, Y_pred)
plt.figure(figsize=(4,3))
sn.heatmap(cmLR, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# ### KMeans

# In[38]:


from sklearn.cluster import KMeans

# Elbow Method
distortions = []
for i in range(1, 11):
    print(i)
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_train)
    distortions.append(kmeans.inertia_)

plt.plot(range(1, 11), distortions)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# In[81]:


kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_train)
Y_kmeans_pred = kmeans.predict(X_test)
kmeans_score = accuracy_score(Y_test, Y_kmeans_pred)
print(kmeans_score)
print(classification_report(Y_test, Y_kmeans_pred))


# In[82]:


cmKM = confusion_matrix(Y_test, Y_kmeans_pred)
plt.figure(figsize=(4,3))
sn.heatmap(cmKM, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# ### Cat Boost Classifier

# In[42]:


from catboost import CatBoostClassifier

cbmodel = CatBoostClassifier()
cbmodel.fit(X_train, Y_train)

Y_cb_pred = cbmodel.predict(X_test)


# In[79]:


cbscore = accuracy_score(Y_test, Y_cb_pred)
print(cbscore)
print(classification_report(Y_test, Y_cb_pred))


# In[83]:


cmCB = confusion_matrix(Y_test, Y_cb_pred)
plt.figure(figsize=(4,3))
sn.heatmap(cmCB, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# ### CNN Model

# In[74]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Input, Embedding, GlobalMaxPooling1D, LeakyReLU, Dropout, MaxPool1D, Bidirectional, TimeDistributed


# In[75]:


X_cnn_train = X_train.values.reshape(520952, 300, 1)
X_cnn_train.shape


# In[76]:


X_cnn_test = X_test.values.reshape(130239, 300, 1)
X_cnn_test.shape


# In[115]:


cnnmodel = Sequential()
cnnmodel.add(Conv1D(filters=32, kernel_size=(3,), padding='same', activation='relu', input_shape = (X_cnn_train.shape[1],1)))
cnnmodel.add(Conv1D(filters=64, kernel_size=(3,), padding='same', activation='relu'))
cnnmodel.add(Conv1D(filters=128, kernel_size=(3,), padding='same', activation='relu'))
cnnmodel.add(MaxPool1D(pool_size=(3,), strides=2, padding='same'))
cnnmodel.add(Dropout(0.5))
cnnmodel.add(Flatten())
cnnmodel.add(Dense(units = 256, activation=LeakyReLU(alpha=0.001)))
cnnmodel.add(Dense(units = 512, activation=LeakyReLU(alpha=0.001)))
cnnmodel.add(Dense(units = 2, activation='sigmoid'))

cnnmodel.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
cnnmodel.summary()


# In[97]:


cnnmodel_history = cnnmodel.fit(X_cnn_train, Y_train, epochs=2, batch_size = 100, validation_data = (X_cnn_test, Y_test))


# In[98]:


Y_cnn_pred = cnnmodel.predict(X_cnn_test)
Y_cnn_pred = np.argmax(Y_cnn_pred, axis=1)
print(classification_report(Y_test, Y_cnn_pred))

CM_CNN = confusion_matrix(Y_test, Y_cnn_pred)
plt.figure(figsize=(4,3))
sn.heatmap(CM_CNN, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# <!-- ### Summary:
# - **Logistic Regression**: 0.8890731654880643
# 
# - **KMeans**: 0.46559786239144957
# 
# - **Cat Boosting**: 0.8890731654880643
# 
# 
# For imbalanced data, LR and Cat Boosting gave the same results. -->

# ## Rebalancing using SMOTE

# In[86]:


count_before = Y.value_counts()

plt.figure(figsize=(2,3))
sn.barplot(x=count_before.index, y=count_before)
plt.title('Before Oversampling')
plt.xlabel('Type')
plt.ylabel('Count')  


# In[87]:


from imblearn.over_sampling import SMOTE

smote_oversample = SMOTE(random_state=2)
X_smote_oversampled, Y_smote_oversampled = smote_oversample.fit_resample(X, Y)

count = Y_smote_oversampled.value_counts()

plt.figure(figsize=(2,3))
sn.barplot(x=count.index, y=count)
plt.title('After Oversampling')
plt.xlabel('Type')
plt.ylabel('Count')                           


# In[88]:


X_smote_train, X_smote_test, Y_smote_train, Y_smote_test = train_test_split(X_smote_oversampled, Y_smote_oversampled, test_size=0.2, random_state=42)

print(X_smote_train.shape)
print(X_smote_test.shape)
print(Y_smote_train.shape)
print(Y_smote_test.shape)


# ### Logistic Regression

# In[47]:


logRegClfSM = LogisticRegression(random_state=0)
logRegClfSM.fit(X_smote_train, Y_smote_train)

Y_smote_pred_LR = logRegClfSM.predict(X_smote_test)
smote_LR_score = accuracy_score(Y_smote_test, Y_smote_pred_LR)
print(smote_LR_score)


# In[84]:


print(classification_report(Y_smote_test, Y_smote_pred_LR))
smote_cmLR = confusion_matrix(Y_smote_test, Y_smote_pred_LR)
plt.figure(figsize=(4,3))
sn.heatmap(smote_cmLR, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# ### KMeans

# In[48]:


# Elbow Method
distortions = []
for i in range(1, 11):
    print(i)
    kmeansSM = KMeans(n_clusters=i, random_state=42)
    kmeansSM.fit(X_smote_train)
    distortions.append(kmeansSM.inertia_)

plt.plot(range(1, 11), distortions)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# In[85]:


kmeansSM = KMeans(n_clusters=2, random_state=42)
kmeansSM.fit(X_smote_train)
Y_kmeans_pred_SM = kmeansSM.predict(X_smote_test)
kmeans_smote_score = accuracy_score(Y_smote_test, Y_kmeans_pred_SM)
print(kmeans_smote_score)


# In[88]:


print(classification_report(Y_smote_test, Y_kmeans_pred_SM))

smote_cmKM = confusion_matrix(Y_smote_test, Y_kmeans_pred_SM)
plt.figure(figsize=(4,3))
sn.heatmap(smote_cmKM, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# ### Cat Boost Classifier

# In[102]:


cbmodelSM = CatBoostClassifier()
cbmodelSM.fit(X_smote_train, Y_smote_train)

Y_smote_pred_CB = cbmodel.predict(X_smote_test)
CB_smote_score = accuracy_score(Y_smote_test, Y_smote_pred_CB)
print(CB_smote_score)


# In[101]:


print(classification_report(Y_smote_test, Y_smote_pred_CB))

smote_cmCB = confusion_matrix(Y_smote_test, Y_smote_pred_CB)
plt.figure(figsize=(4,3))
sn.heatmap(smote_cmCB, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# ### CNN Model on SMOTE data

# In[91]:


X_cnn_smote_train = X_smote_train.values.reshape(684964, 300, 1)
X_cnn_smote_train.shape


# In[92]:


X_cnn_smote_test = X_smote_test.values.reshape(171242, 300, 1)
X_cnn_smote_test.shape


# In[93]:


cnnmodel_SMOTE = Sequential()
cnnmodel_SMOTE.add(Conv1D(filters=32, kernel_size=(3,), padding='same', activation='relu', input_shape = (X_cnn_smote_train.shape[1],1)))
cnnmodel_SMOTE.add(Conv1D(filters=64, kernel_size=(3,), padding='same', activation='relu'))
cnnmodel_SMOTE.add(Conv1D(filters=128, kernel_size=(3,), padding='same', activation='relu'))
cnnmodel_SMOTE.add(MaxPool1D(pool_size=(3,), strides=2, padding='same'))
cnnmodel_SMOTE.add(Dropout(0.5))
cnnmodel_SMOTE.add(Flatten())
cnnmodel_SMOTE.add(Dense(units = 256, activation=LeakyReLU(alpha=0.001)))
cnnmodel_SMOTE.add(Dense(units = 512, activation=LeakyReLU(alpha=0.001)))
cnnmodel_SMOTE.add(Dense(units = 2, activation='softmax'))

cnnmodel_SMOTE.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
cnnmodel_SMOTE.summary()


# In[94]:


cnnmodel_history_SMOTE = cnnmodel.fit(X_cnn_smote_train, Y_smote_train, epochs=5, batch_size = 10, validation_data = (X_cnn_smote_test, Y_smote_test))


# In[99]:


Y_cnn_smote_pred = cnnmodel.predict(X_cnn_smote_test)
Y_cnn_smote_pred = np.argmax(Y_cnn_smote_pred, axis=1)
print(classification_report(Y_smote_test, Y_cnn_smote_pred))

CM_CNN = confusion_matrix(Y_smote_test, Y_cnn_smote_pred)
plt.figure(figsize=(4,3))
sn.heatmap(CM_CNN, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# ### Summary

# ## Rebalancing using ADASYN

# In[103]:


from imblearn.over_sampling import ADASYN

adasyn_oversample = ADASYN(random_state=2)
X_adasyn_oversampled, Y_adasyn_oversampled = adasyn_oversample.fit_resample(X, Y)


# In[106]:


X_adasyn_train, X_adasyn_test, Y_adasyn_train, Y_adasyn_test = train_test_split(X_adasyn_oversampled, Y_adasyn_oversampled, test_size=0.2, random_state=42)

print(X_adasyn_train.shape)
print(X_adasyn_test.shape)
print(Y_adasyn_train.shape)
print(Y_adasyn_test.shape)


# ### Logistic Regression

# In[57]:


logRegClfAD = LogisticRegression(random_state=0)
logRegClfAD.fit(X_adasyn_train, Y_adasyn_train)

Y_adasyn_pred_LR = logRegClfAD.predict(X_adasyn_test)
adasyn_LR_score = accuracy_score(Y_adasyn_test, Y_adasyn_pred_LR)
print(adasyn_LR_score)


# In[90]:


print(classification_report(Y_adasyn_test, Y_adasyn_pred_LR))

adasyn_cmLR = confusion_matrix(Y_adasyn_test, Y_adasyn_pred_LR)
plt.figure(figsize=(4,3))
sn.heatmap(adasyn_cmLR, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# ### KMeans

# In[58]:


# Elbow Method
distortions = []
for i in range(1, 11):
    print(i)
    kmeansAD = KMeans(n_clusters=i, random_state=42)
    kmeansAD.fit(X_smote_train)
    distortions.append(kmeansAD.inertia_)

plt.plot(range(1, 11), distortions)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()


# In[91]:


kmeansAD = KMeans(n_clusters=2, random_state=42)
kmeansAD.fit(X_adasyn_train)
Y_kmeans_pred_AD = kmeansAD.predict(X_adasyn_test)
kmeans_adasyn_score = accuracy_score(Y_adasyn_test, Y_kmeans_pred_AD)
print(kmeans_adasyn_score)


# In[92]:


print(classification_report(Y_adasyn_test, Y_kmeans_pred_AD))

adasyn_cmKM = confusion_matrix(Y_adasyn_test, Y_kmeans_pred_AD)
plt.figure(figsize=(4,3))
sn.heatmap(adasyn_cmKM, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# ### Cat Boost Classifier

# In[60]:


cbmodelAD = CatBoostClassifier()
cbmodelAD.fit(X_adasyn_train, Y_adasyn_train)

Y_adasyn_pred_CB = cbmodel.predict(X_adasyn_test)
CB_adasyn_score = accuracy_score(Y_adasyn_test, Y_adasyn_pred_CB)
print(CB_adasyn_score)


# In[93]:


print(classification_report(Y_adasyn_test, Y_adasyn_pred_CB))

adasyn_cmCB = confusion_matrix(Y_adasyn_test, Y_adasyn_pred_CB)
plt.figure(figsize=(4,3))
sn.heatmap(adasyn_cmCB, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# ### CNN Model on ADASYN Data

# In[118]:


X_cnn_adasyn_train = X_adasyn_train.values.reshape(687358, 300, 1)
X_cnn_adasyn_train.shape


# In[117]:


X_cnn_adasyn_test = X_adasyn_test.values.reshape(171840, 300, 1)
X_cnn_adasyn_test.shape


# In[119]:


cnnmodel_SMOTE = Sequential()
cnnmodel_SMOTE.add(Conv1D(filters=32, kernel_size=(3,), padding='same', activation='relu', input_shape = (X_cnn_adasyn_train.shape[1],1)))
cnnmodel_SMOTE.add(Conv1D(filters=64, kernel_size=(3,), padding='same', activation='relu'))
cnnmodel_SMOTE.add(Conv1D(filters=128, kernel_size=(3,), padding='same', activation='relu'))
cnnmodel_SMOTE.add(MaxPool1D(pool_size=(3,), strides=2, padding='same'))
cnnmodel_SMOTE.add(Dropout(0.5))
cnnmodel_SMOTE.add(Flatten())
cnnmodel_SMOTE.add(Dense(units = 256, activation=LeakyReLU(alpha=0.001)))
cnnmodel_SMOTE.add(Dense(units = 512, activation=LeakyReLU(alpha=0.001)))
cnnmodel_SMOTE.add(Dense(units = 2, activation='sigmoid'))

cnnmodel_SMOTE.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
cnnmodel_SMOTE.summary()


# In[156]:


cnnmodel_history_ADASYN = cnnmodel.fit(X_cnn_adasyn_train, Y_adasyn_train, epochs=2, batch_size = 100, validation_data = (X_cnn_adasyn_test, Y_adasyn_test))


# In[159]:


Y_cnn_pred_adasyn = cnnmodel.predict(X_cnn_adasyn_test)
print(Y_cnn_pred_adasyn[:5])
Y_cnn_pred_adasyn = np.argmax(Y_cnn_pred_adasyn, axis=1)
print(classification_report(Y_adasyn_test, Y_cnn_pred_adasyn))

CM_ADASYN = confusion_matrix(Y_adasyn_test, Y_cnn_pred_adasyn)
plt.figure(figsize=(4,3))
sn.heatmap(CM_ADASYN, annot=True,fmt=".1f")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[153]:


df_example = new_df[['v10', 'v200', 'type']]
df_example.head()
sn.scatterplot(data = df_example, x ='v10', y = 'v200', hue = 'type')


# In[154]:


df_example2 = X_smote_oversampled
df_example2['type'] = Y_smote_oversampled
sn.scatterplot(data = df_example2, x ='v10', y = 'v200', hue = 'type')


# In[155]:


df_example3 = X_adasyn_oversampled
df_example3['type'] = Y_adasyn_oversampled
sn.scatterplot(data = df_example3, x ='v10', y = 'v200', hue = 'type')


# ## Testing with HTTP Traffic

# In[41]:


normal_traffic_train = 'HTTPtraffic/normalTrafficTraining.txt'
normal_traffic_test = 'HTTPtraffic/normalTrafficTest.txt'
anomalous_traffic_test = 'HTTPtraffic/anomalousTrafficTest.txt'


# In[42]:


def parseFile(filename):
    fptr = open(filename)
    block = ""

    request_type = []
    urls = []
    hosts = []
    user_agent = []
    pragma = []
    cache_control = []
    accept = []
    accept_encoding = []
    accept_charset = []
    accept_language = []
    cookie = []
    conn = []
    content_type = []
    content_len = []
    
    count = 0

    for line in fptr.readlines():
        req = 'NULL'
        url = 'NULL'
        http = 'NULL'
        uag = 'NULL'
        p = 'NULL'
        cc = 'NULL'
        a = 'NULL'
        ae = 'NULL'
        ac = 'NULL'
        al = 'NULL'
        ckie = 'NULL'
        c = 'NULL'
        cont_type = 'NULL'
        cont_len = 'NULL'
        block += line
        if line == "\n":
            count += 1
            if count == 2:
                count = 0
                block = ""
                continue
            data = block.split("\n")
            for i in range(len(data)):
                d = data[i].strip()
                if d.startswith("GET") or d.startswith("POST") or d.startswith("PUT"):
                    req, url, http = d.split(" ")
                elif d.startswith("User-Agent: "):
                    uag = d[12:]
                elif d.startswith("Pragma: "):
                    p = d[8:]
                elif d.startswith("Cache-Control: "):
                    cc = d[15:]
                elif d.startswith("Accept: "):
                    a = d[8:]
                elif d.startswith("Accept-Encoding: "):
                    ae = d[17:]
                elif d.startswith("Accept-Charset: "):
                    ac = d[16:]
                elif d.startswith("Accept-Language"):
                    al = d[17:]
                elif d.startswith("Cookie: "):
                    ckie = d[8:]
                elif d.startswith("Connection: "):
                    c = d[12:]
                elif d.startswith("Content-Type: "):
                    cont_type = d[14:]
                elif d.startswith('Content-Length: '):
                    cont_len = d[16:]
            
            request_type.append(req)
            urls.append(req + " " + url)
            host = data[8][6:]
            
            hosts.append(host)
            user_agent.append(uag)
            pragma.append(p)
            cache_control.append(cc)
            accept.append(a)
            accept_encoding.append(ae)
            accept_charset.append(ac)
            accept_language.append(al)
            cookie.append(ckie)
            conn.append(c)
            content_type.append(cont_type)
            content_len.append(cont_len)
            
            block = ""
            
    http_traffic_data = pd.DataFrame(data=list(zip(request_type, urls, hosts, user_agent, pragma, cache_control, accept, accept_encoding, accept_charset, accept_language, cookie, conn, content_type, content_len)), columns=["method", "path", "host", "user_agent", "pragma", "cache_control", "accept", "accept_enc", "accept_char", "accept_lang", "cookie", "conn", "content_type", "content_len"])
    
    return http_traffic_data


# In[43]:


df_normal_traffic = parseFile(normal_traffic_train)
df_normal_traffic


# In[44]:


df_normal_traffic["label"] = 0
df_normal_traffic


# In[45]:


df_anomalous_traffic = parseFile(anomalous_traffic_test)
df_anomalous_traffic["label"] = 1
df_anomalous_traffic


# In[46]:


df_all_requests = pd.concat([df_normal_traffic, df_anomalous_traffic], ignore_index=True, sort='False')
df_all_requests


# In[47]:


df_all_requests['path'] = df_all_requests['path'].apply(lambda x : str(x))
df_all_requests['url_tokens'] = df_all_requests['path'].apply(lambda x: getTokens(x))

df_all_requests.head()


# In[49]:


df_all_requests = df_all_requests[['path', 'label', 'url_tokens']]
df_all_requests.head()


# In[61]:


def create_paragraph_vectors(tokens):
    temp = []
    for i in range(300):
        sum = 0
        for token in tokens:
            try:
                sum += tfidf_weighted_vector[token][i]
            except KeyError as e:
                v = np.random.uniform(0,1,300)
                sum += v[i]
        temp.append(sum/len(tokens))
    paragraph_vector = np.asarray(temp, dtype=np.float32)
    return paragraph_vector

tokens = ['get', 'http', 'localhost', '8080', 'tienda1', 'index', 'jsp']
res = create_paragraph_vectors(tokens)
print(res.shape)

example_url = ['https', 'www', 'google', 'com']
res = create_paragraph_vectors(example_url)
print(res.shape)


# In[62]:


df_all_requests['paragraph_vector'] = df_all_requests['url_tokens'].apply(lambda x: create_paragraph_vectors(x))
df_all_requests.head()


# In[63]:


import warnings
warnings.filterwarnings('ignore')

df_all_requests[cols] = pd.DataFrame(df_all_requests['paragraph_vector'].to_list(), index=df_all_requests.index)
df_all_requests.head()


# In[64]:


http_X = df_all_requests.iloc[:, 4:]
http_Y = df_all_requests['label']


# In[65]:


http_X.shape


# In[66]:


http_Y.shape


# In[67]:


from sklearn.model_selection import train_test_split

http_X_train, http_X_test, http_Y_train, http_Y_test = train_test_split(http_X, http_Y, test_size=0.2, random_state=42)

print(http_X_train.shape)
print(http_X_test.shape)
print(http_Y_train.shape)
print(http_Y_test.shape)


# ### Logistic Regression

# In[68]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logRegClfHTTP = LogisticRegression(random_state=0)
logRegClfHTTP.fit(http_X_train, http_Y_train)

http_Y_pred = logRegClfHTTP.predict(http_X_test)
score = accuracy_score(http_Y_test, http_Y_pred)
print(score)
print(classification_report(http_Y_test, http_Y_pred))


# In[70]:


http_Y_pred = logRegClf.predict(http_X_test)
score = accuracy_score(http_Y_test, http_Y_pred)
print(score)
print(classification_report(http_Y_test, http_Y_pred))


# ### Cat Boost

# In[71]:


from catboost import CatBoostClassifier

cbmodelHTTP = CatBoostClassifier()
cbmodelHTTP.fit(http_X_train, http_Y_train)

http_Y_cb_pred = cbmodelHTTP.predict(http_X_test)


# In[72]:


http_cbscore = accuracy_score(http_Y_test, http_Y_cb_pred)
print(http_cbscore)
print(classification_report(http_Y_test, http_Y_cb_pred))



# coding: utf-8

# In[1]:


import os
#print(os.listdir("../input"))

import pandas as pd
import numpy as np
from scipy.stats import randint
import seaborn as sns # used for plot interactive graph. 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import re
import collections
 
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict


# In[2]:


#read data
df_is_comments= pd.read_csv ("C:/Garima/AI/AIBugTriaging/csvdata/issue_comment.csv", encoding ='latin1') 
df_is= pd.read_csv ("C:/Garima/AI/AIBugTriaging/csvdata/issue.csv", encoding ='latin1') 
df_is_comp= pd.read_csv ("C:/Garima/AI/AIBugTriaging/csvdata/issue_component.csv", encoding ='latin1')

df_is_comments= df_is_comments[["issue_id","message"]]

#Concatenate issue comments message along the issue id
df_is_comments=df_is_comments.groupby('issue_id')['message'].apply(' '.join).reset_index()
print(df_is_comments.head())
df_is =df_is[['issue_id','summary','description']]
df_is['summary']=df_is['summary'] + ' ' + df_is['description']
df_is =df_is[['issue_id','summary']]
df_is_wth_comp=pd.merge(df_is,df_is_comp,on="issue_id")
df_is_wth_comm= pd.merge(df_is_wth_comp, df_is_comments, on="issue_id")


# In[3]:


print("***Count NaN in each column of a DataFrame with comp***")

print("Nan in each columns" , df_is_wth_comp.isnull().sum(), sep='\n')

print("***Count NaN in each column of a DataFrame with comp and comments***")

print("Nan in each columns" , df_is_wth_comm.isnull().sum(), sep='\n')


# In[4]:


#Make data ready by concatenating and selecting required columns also copy data in csv'
df_is_wth_commf=df_is_wth_comm
df_is_wth_commf['summary']=df_is_wth_commf['summary'] + df_is_wth_commf['message']
df_is_wth_commf =df_is_wth_commf[['issue_id','component','summary']]
df_is_wth_comm.to_csv('C:/Garima/AI/AIBugTriaging/csvdata/df_is_wth_comm.csv')


# In[5]:


len(df_is_wth_comm)


# In[6]:


print(df_is_wth_comm.shape)
df_is_wth_comm.head(5).T


# In[7]:


#Data exploration
df_is_wth_commf = df_is_wth_commf[df_is_wth_commf['summary'].notna()].reset_index(drop=True)
print("***Count NaN in each column of a DataFrame with comp and comments***")

print("Nan in each columns" , df_is_wth_commf.isnull().sum(), sep='\n')

#Drop some of the categories
indexNames = df_is_wth_commf[ (df_is_wth_commf['component'] == 'Selenium tests') | (df_is_wth_commf['component'] == 'Audit Logging') | (df_is_wth_commf['component'] == 'Problem Reporting')
                            | (df_is_wth_commf['component'] == 'Repository Statistics')].index

df_is_wth_commf.drop(indexNames , inplace=True)
df_is_wth_commf.reset_index(inplace=True)
len(df_is_wth_commf)


# In[8]:


df_is_wth_commf.head(5).T


# In[9]:


#Data exploration continued
pd.DataFrame(df_is_wth_commf.component.unique())


# In[10]:


#combine component types
df_is_wth_commf.replace({'component': 
             {'browser': 'browser and search',
              'search' : 'browser and search',
              'indexing' : 'browser and search',
              'Design' : 'system',
              'scheduling' : 'system',
              'Maven 2 Support' : 'others',
              'redback' : 'others',
              'WebDAV Interface' : 'web',
              'Metadata Repository' : 'repository',
              'repository converter' : 'repository',
              'repository interface' : 'repository',
              'repository scanning' : 'repository',
              'rest services' : 'web',
              'Web Interface' : 'web',
              'reporting' : 'reporting and logging',
              }},inplace= True)


# In[11]:


#Data exploration continued
pd.DataFrame(df_is_wth_commf.component.unique())


# In[12]:


# Create a new column 'category_id' with encoded categories 
df_is_wth_commf['category_id'] = df_is_wth_commf['component'].factorize()[0]
category_id_df = df_is_wth_commf[['component', 'category_id']].drop_duplicates()


# Dictionaries for future use
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'component']].values)

# New dataframe
df_is_wth_commf.head()


# In[13]:


print(category_to_id)
print(id_to_category)


# In[14]:


#number of entries per category
fig = plt.figure(figsize=(8,6))
colors = ['grey','grey','grey','grey','grey','grey','grey','grey','grey',
    'grey','darkblue','darkblue','darkblue']
df_is_wth_commf.groupby('component').category_id.count().sort_values().plot.barh(
    ylim=0, color=colors, title= 'NUMBER OF BUGS IN EACH CATEGORY\n')
plt.xlabel('Number of ocurrences', fontsize = 10);


# In[15]:


#clean data
data = df_is_wth_commf.summary.values.tolist()
data = [re.sub("^https?:\/\/.*[\r\n]*", " ", str(sent)) for sent in data]
data = [re.sub(r'\s+', ' ', str(sent)) for sent in data]
#data = [re.sub("\'", "", str(sent)) for sent in data]
#remove all non alpha numeric characters
#data = [re.sub("[^a-zA-Z0-9 -]", " ", str(sent)) for sent in data]
#find multiple occurence of a dinle character in a string, or look at regx table
#data = [re.sub("\w+(\_|\.)\w+", " ", str(sent)) for sent in data]
data = [re.sub("[^a-zA-Z]", " ", str(sent)) for sent in data]
#replace multiple white spaces
data = [re.sub("/\s\s+/g", " ", str(sent)) for sent in data]


# In[16]:


df_data= pd.DataFrame(data, columns =['description'])
len(df_data)


# In[17]:


df_data= pd.concat([df_is_wth_commf,df_data], axis=1)
df_data.to_csv("C:/Garima/AI/AIBugTriaging/csvdata/test.csv")
#df_data = df_data[df_data['description'].notna()].reset_index(drop=True)
len(df_data)


# In[18]:


len(df_is_wth_commf)


# In[19]:


df_data=df_data[['component','category_id','description']]
#df_data.head()
print("Nan in each columns" , df_data.isnull().sum(), sep='\n')


# In[20]:


tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,
                        ngram_range=(1, 2), 
                        stop_words='english')

# We transform each dexcription into a vector
features = tfidf.fit_transform(df_data.description).toarray()

labels = df_data.category_id

print("Each of the %d bugs is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))


# In[21]:


# Finding the three most correlated terms with each of the component categories
N = 3
for Product, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("\n==> %s:" %(Product))
  print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
  print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))


# In[23]:


#Spliting the data into train and test sets
X = df_data['description'] # Collection of documents
y = df_data['component'] # Target or the labels we want to predict (i.e., the 13 different complaints of products)

#model evaluation
X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(features, 
                                                               labels, 
                                                               df_data.index, test_size=0.25, 
                                                               random_state=1)
model = LinearSVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[25]:


#Merge the predicted result into actual dataframe
df= pd.DataFrame(columns =['y'], index=indices_test)
df['y']=y_pred
df1= pd.DataFrame(columns =['y0'], index=indices_train)
df1['y']=y_train
df_result= pd.concat([df,df1])
df_result = df_result[['y']]
df_result= pd.concat([df_result,df_data],axis=1)
df_result=df_result[['description','category_id','y']]
df_result= pd.concat([df_is_wth_commf,df_result],axis=1)
df_result=df_result[['issue_id','component','description','category_id','y']]
df_result.to_csv("C:/Garima/AI/AIBugTriaging/csvdata/results.csv")
df_result.head()


# In[26]:


y_predtrain = model.predict(X_train)


# In[27]:


# Classification report
print('\t\t\t\tCLASSIFICATIION METRICS\n')
print(metrics.classification_report(y_train, y_predtrain, 
                                    target_names= df_data['component'].unique()))


# In[28]:


#confusion matrix
conf_mat = confusion_matrix(y_train, y_predtrain)
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt='d',
            xticklabels=category_id_df.component.values, 
            yticklabels=category_id_df.component.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title("CONFUSION MATRIX - LinearSVC\n", size=16);


# In[29]:


#most correlated term with each category
model.fit(features, labels)

N = 4
for Product, category_id in sorted(category_to_id.items()):
  indices = np.argsort(model.coef_[category_id])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
  bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
  print("\n==> '{}':".format(Product))
  print("  * Top unigrams: %s" %(', '.join(unigrams)))
  print("  * Top bigrams: %s" %(', '.join(bigrams)))


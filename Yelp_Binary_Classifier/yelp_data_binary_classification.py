# For this first part, we will build a parser for extracting tokens from the **review text** only. First, you should tokenize each review using **whitespaces and punctuations as delimiters**. Do not remove stopwords. You should apply casefolding (lower case everything) and use the [nltk Porter stemmer](http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.porter) ... you may need to install nltk if you don't have it already. 

import json
import re

from nltk.stem import PorterStemmer

import json
import re

review_text_list = []

for line in open('training_data.json', 'r'):
    review_text_list.append(json.loads(line)["text"].encode("utf-8"))


stemmer = PorterStemmer()

list_of_list_text_review = []

#TOkenizing the review text
for string in review_text_list:
    string = string.lower()
    training_data_tokenized = re.split("[^A-Za-z0-9#]",string)
    training_data_tokenized = filter(None,training_data_tokenized)
    stemmed = [stemmer.stem(item) for item in training_data_tokenized]
    list_of_list_text_review.append(stemmed)


training_data_stem = [item for sub_list in list_of_list_text_review for item in sub_list]
#Unique tokens
unique_set = set(training_data_stem)
print("Unique words Length", len(unique_set))


# Great, now we can tokenize the documents. Let's make a list of the most popular words in our reviews. For this step, you should maintain a count of how many times each word occurs. Then you should print out the top-20 words in your reviews.
# 
import math as math

#Most popular words
popular_dict = {}

#
for item in training_data_stem:
    if item not in popular_dict:
        popular_dict[item] = 1
    else:
        popular_dict[item] += 1
        
X = []
Y = []
#popular_dict_rev_sorted = sorted(popular_dict.items(), key=lambda x:x[1],reverse=True)
popular_dict_rev_sorted = sorted(popular_dict, key=popular_dict.get, reverse=True)
for i,ele in enumerate(popular_dict_rev_sorted):
    if(i+1 <= 20):
        print i+1, ele, popular_dict[ele]
    X.append(math.log(i+1, 10))
    Y.append(math.log(popular_dict[ele],10))


# ### Zipf's Law

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt  

plt.plot(X, Y)
plt.xlabel('RANK in log-base10')
plt.ylabel('Term Count in log-base10')
plt.show()


# Zipfs law states that, frequency decreases very rapidly with rank. i.e. the most frequent word will occur approximately twice as often as the second most frequent word, three times as often as the third most frequent word, etc. As we can see from the above graph and from the frequency count  table above, the graph follows Zipf's law closely

# In this part you will build feature vectors for each review. This will be input to our ML classifiers. You should call your parser from earlier, using all the same assumptions (e.g., casefolding, stemming). Each feature value should be the term count for that review.

from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer()
list_of_text_reviews =  [' '.join(sub_list) for sub_list in list_of_list_text_review]
#list_of_text_reviews =  [xxx for sub_list in list_of_list_text_review for xxx in sub_list]

vector_counts = vector.fit_transform(list_of_text_reviews)
#print vector_counts.shape

y = []
for line in open('training_data.json', 'r'):
    y.append(json.loads(line)["label"].encode("utf-8"))


# 
# ### Setting 1: Splitting data into train-test 
# 
# In the first setting, you should treat the first 70% of your data as training. The remaining 30% should be for testing. 
# 
# ### Setting 2: Using 5 fold cross-validation
# 
# In the second setting, use 5-fold cross-validation. 

#### Setting 1, Splitting data into train-test  ###
from sklearn import preprocessing
label_process = preprocessing.LabelEncoder()
feature_label_y = label_process.fit_transform(y)

import numpy as np

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


from sklearn.model_selection import train_test_split
import scipy

#vector_counts = scipy.sparse.csr_matrix(vector_counts)
X_train, X_test, y_train, y_test = train_test_split(vector_counts, feature_label_y,
                                                stratify=feature_label_y, 
                                                test_size=0.3)


# In[7]:


# First classifier kNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
clf = neigh.fit(X_train, 
                y_train)

predicted_y = clf.predict(X_test)
print "Accuracy of k Nearest neighbor Classifier is", np.mean(predicted_y == y_test)  
print 'k Nearest neighbor Precision score of Food relevant class:', precision_score(y_test, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(y_test, predicted_y, average='macro',labels=[0]), 'Recall score of Food relevant class:' , recall_score(y_test, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(y_test, predicted_y, average='macro',labels=[0])

# Second classifier Decision Tree
from sklearn import tree
tree_handle = tree.DecisionTreeClassifier()
clf = tree_handle.fit(X_train, 
                      y_train)
predicted_y = clf.predict(X_test)
print "Accuracy of Decision Tree Classifier is", np.mean(predicted_y == y_test)  
print 'Decision Tree Classifier Precision score of Food relevant class:', precision_score(y_test, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(y_test, predicted_y, average='macro',labels=[0]), 'Recall score of Food relevant class:' , recall_score(y_test, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(y_test, predicted_y, average='macro',labels=[0])

# Third classifier Naive Bayes
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
bayes_clf = naive_bayes.fit(X_train, 
                            y_train)
predicted_y = bayes_clf.predict(X_test)
print "Accuracy of Naive Bayes Classifier is", np.mean(predicted_y == y_test)  
print 'Naive Bayes Classifier Precision score of Food relevant class:', precision_score(y_test, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(y_test, predicted_y, average='macro',labels=[0]), 'Recall score of Food relevant class:' , recall_score(y_test, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(y_test, predicted_y, average='macro',labels=[0])

# Fourth classifier SVM
from sklearn import svm
clf = svm.SVC()
svm_clf = clf.fit(X_train, 
                  y_train)
predicted_y = svm_clf.predict(X_test)
print "Accuracy of SVM Classifier is",np.mean(predicted_y == y_test)  
print 'SVM Classifier Precision score of Food relevant class:', precision_score(y_test, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(y_test, predicted_y, average='macro',labels=[0]), 'Recall score of Food relevant class:' , recall_score(y_test, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(y_test, predicted_y, average='macro',labels=[0])


### Setting 2: Using 5 fold cross-validation ###
from sklearn.model_selection import cross_val_predict

print "---------------------------------------------------------------------------------------------"

from sklearn.model_selection import cross_val_score

neigh = KNeighborsClassifier(n_neighbors=5)
predicted_y = cross_val_predict(neigh,  vector_counts, feature_label_y, cv=5)
print "Accuracy of k Nearest neighbor Classifier is", np.mean(predicted_y == feature_label_y)  
print 'k Nearest neighbor Precision score of Food relevant class:', precision_score(feature_label_y, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(feature_label_y, predicted_y, average='macro',labels=[0]), 'Recall score of Food relevant class:' , recall_score(feature_label_y, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(feature_label_y, predicted_y, average='macro',labels=[0])




handle = tree_handle
predicted_y = cross_val_predict(handle,  vector_counts, feature_label_y, cv=5)
print "Accuracy of Decision Tree Classifier Classifier is", np.mean(predicted_y == feature_label_y)  
print 'Decision Tree Classifier Precision score of Food relevant class:', precision_score(feature_label_y, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(feature_label_y, predicted_y, average='macro',labels=[0]), 'Recall score of Food relevant class:' , recall_score(feature_label_y, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(feature_label_y, predicted_y, average='macro',labels=[0])


handle = naive_bayes
predicted_y = cross_val_predict(handle,  vector_counts, feature_label_y, cv=5)
print "Accuracy of Naive Bayes Classifier is", np.mean(predicted_y == feature_label_y)  
print 'Naive Bayes Classifier Precision score of Food relevant class:', precision_score(feature_label_y, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(feature_label_y, predicted_y, average='macro',labels=[0]), 'Recall score of Food relevant class:' , recall_score(feature_label_y, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(feature_label_y, predicted_y, average='macro',labels=[0])


# ## Below cell takes a long time to run, please run if needed.

# In[ ]:


clf = svm.SVC()
handle = clf
predicted_y = cross_val_predict(handle,  vector_counts, feature_label_y, cv=5)
print "Accuracy of SVM Classifier is", np.mean(predicted_y == feature_label_y)  
print 'SVM Classifier Precision score of Food relevant class:', precision_score(feature_label_y, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(feature_label_y, predicted_y, average='macro',labels=[0]), 'Recall score of Food relevant class:' , recall_score(feature_label_y, predicted_y, average='macro',labels=[1]), 'Food irrelvant class is', precision_score(feature_label_y, predicted_y, average='macro',labels=[0])


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score


from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()

#n = 10000
vectorizer = TfidfVectorizer(sublinear_tf=False, max_df=0.72,stop_words='english',
                            ngram_range=(1,3),
                            use_idf=True,
                            min_df=4,
                            )

vector_counts = vectorizer.fit_transform(list_of_text_reviews)

X_train, X_test, y_train, y_test = train_test_split(vector_counts, feature_label_y,
                                                stratify=feature_label_y, 
                                                test_size=0.3)

model = naive_bayes
clf = model.fit(X_train, 
                y_train)

predicted_y = clf.predict(X_test)
print "Accuracy of Naive Bayes Classifier is", np.mean(predicted_y == y_test)  
bayes_result = predicted_y

from sklearn.ensemble import RandomForestClassifier 
ran_forest_model = RandomForestClassifier()
model = ran_forest_model
clf = model.fit(X_train, 
                y_train)
predicted_y = clf.predict(X_test)
print "Accuracy of Ran forest Classifier is", np.mean(predicted_y == y_test)
ran_forest_result = predicted_y


# In[11]:


from sklearn import svm
    
clf = svm.SVC(kernel='linear',class_weight = 'balanced')

svm_clf = clf.fit(X_train, 
                  y_train)
predicted_y = svm_clf.predict(X_test)
print "Accuracy of SVM Classifier is", np.mean(predicted_y == y_test)


# In[12]:


fin_pred = []
for index,item in enumerate(bayes_result):
    if(bayes_result[index] == ran_forest_result[index]):
        fin_pred.append(bayes_result[index])
    elif(bayes_result[index] == predicted_y[index]):
        fin_pred.append(bayes_result[index])
    elif(ran_forest_result[index] == predicted_y[index]):
        fin_pred.append(ran_forest_result[index])
        
print "Accuracy of best ensemble Classifier is", np.mean(fin_pred == y_test)



lot = []
#Keep in mind that this will take a lot of time to run. Maybe more than an hour. on 32 GB ram machine.
xxx = list_of_text_reviews[0:6400]

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

yyy = []
for sen in xxx:
    fil_sen = []
    for w in sen.split(" "):
        if w not in stop_words:
            fil_sen.append(w)
    yyy.append(' '.join(fil_sen))
    
#print yyy[1]
#print xxx[1]

for index,item in enumerate(yyy):
    lot.append(tuple([item] + [y[index]]))
#print lot[0]

from nltk.tokenize import word_tokenize # or use some other tokenizer
all_words = set(word.lower() for passage in lot for word in (passage[0].split(" ")))
t = [({word: (word in (x[0].split(" "))) for word in all_words}, x[1]) for x in lot]
#print t[8]
import nltk
classifier = nltk.NaiveBayesClassifier.train(t)
classifier.show_most_informative_features()


# In[13]:


def print_top10(vectorizer, clf, class_labels):
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s" % (",".join(feature_names[j] for j in top10)))

model = naive_bayes
clf = model.fit(vector_counts, 
                feature_label_y)


print "Most 10 informative features for Food-relevant class is"
print_top10(vectorizer, clf, [0])


# In[14]:


def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%-15s\t\t%-15s" % ( fn_1, fn_2)
        
model = naive_bayes
clf = model.fit(vector_counts, 
                feature_label_y)


print "Most nformative features are: Left column is for irrelavant and right for relavant"
show_most_informative_features(vectorizer, clf)

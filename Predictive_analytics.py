#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Import Statements for part 1
import math
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import operator
import sys
import copy

# Import statements for part 2
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report # remove


# In[5]:


def Accuracy(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    
    """
    count = 0
    for i in range(len(y_pred)):
        if(y_pred[i] == y_true[i]):
            count += 1
    return count/len(y_pred)


# In[6]:


def Recall(y_true,y_pred):
#     :type y_true: numpy.ndarray
#     :type y_pred: numpy.ndarray
#     :rtype: float
    
    conf1 = ConfusionMatrix(y_pred, y_true)
    conf = (np.transpose(conf1))
    leng = len(set(y_true))
    recall = np.zeros(leng)
    for i in range(len(recall)):
        val = 0
        #print(conf[i][i])
        #print(np.sum(conf[i]))
        val = conf[i][i]/ np.sum(conf[i])
        #print(val)
        recall[i] = val
#     print(recall)
       
    return (np.sum(recall)/len(recall))    


# In[7]:


def Precision(y_true,y_pred):
    """
    :type y_true: numpy.ndarray
    :type y_pred: numpy.ndarray
    :rtype: float
    """
    conf = ConfusionMatrix(y_pred, y_true)

    leng = len(set(y_true))
    precision = np.zeros(leng)
   
    for i in range(len(precision)):
        val = 0
        #print(conf[i][i])
        #print(np.sum(conf[i]))
        val = conf[i][i]/ np.sum(conf[i])
        #print(val)
        precision[i] = val
    #print(precision)
       
    return (np.sum(precision)/len(precision))    


# In[8]:


def WCSS(clusters, centroids):
    """
    :Clusters List[numpy.ndarray]
    :rtype: float
    """
    my_wcss = 0
    no_of_clusters = len(clusters)
    for cluster in range(no_of_clusters):
        my_wcss += np.sum((clusters[cluster] - centroids[cluster, :]) ** 2) 
    return my_wcss


# In[9]:


def ConfusionMatrix(y_pred, y_true):
    result = []
    leng = len(set(y_true))
    #print(set(y_true))
    for i in range(leng):
        result.append(np.zeros(leng))
       
    for i in range(len(y_true)):
        pred = y_pred[i]
        actual = y_true[i]
       
        result[pred - 1][actual - 1] += 1
       
    return np.asarray(result)


# In[10]:


def KMeans(x_train, no_of_clusters):
#     References
#     https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42
#     https://mubaris.com/posts/kmeans-clustering/
    no_of_rows = x_train.shape[0]
    no_of_feat = x_train.shape[1]

    # init centroid randomly from the dataset
    centroid_arr = np.array([]).reshape(no_of_feat, 0)
    for i in range(no_of_clusters):
        centroid_arr = np.c_[centroid_arr, x_train[random.randint(0, no_of_rows - 1)]]
    centroids_init = centroid_arr.T
    
    centroids_old = np.zeros((no_of_clusters, no_of_feat))
    centroids_new = copy.deepcopy(centroids_init)
    
    centroid_diff = np.linalg.norm(centroids_new - centroids_old)
    
    distance_arr = np.zeros((no_of_rows, no_of_clusters)) # tracks distance
    
    while centroid_diff != 0:
        
        for cluster in range(no_of_clusters):
            distance_arr[:, cluster] = np.linalg.norm(x_train - centroids_new[cluster], axis = 1)
    
        labels = np.argmin(distance_arr, axis = 1) # gives labels
        
        # updating cluster centers
        centroids_old = copy.deepcopy(centroids_new)
            
        # Recalculate new centroid
        for cluster in range(no_of_clusters):
            centroids_new[cluster, :] = np.mean(centroids_old[cluster], axis = 0)
            
        centroid_diff = np.linalg.norm(centroids_new - centroids_old)
       
            
    predictions = set(labels)
    results_arr = []
    
    for prediction in predictions:
        my_index = []
        for label in labels:
            if label == prediction:
                my_index = np.where(labels == label)
        temp = x_train[my_index]
        results_arr.append(temp)
        
    return results_arr, centroids_new


# In[11]:


def KNN(X_train,X_test,Y_train,n):
#     scaler = StandardScaler()
#     #X_norm = scaler.fit_transform(X)
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.fit_transform(X_test)
    final = []
    #Reference: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    dist = -2 * np.dot(X_test, X_train.T) + np.sum(X_train**2,    axis=1) + np.sum(X_test**2, axis=1)[:, np.newaxis]
    sortDist = np.argsort(dist, axis=1)
    filtered = sortDist[:,:n]
    for i in filtered:
        votes={}
        for j in i:
            z=Y_train.take(j)
            if z in votes:
                votes[z]+=1
            else:
                votes[z]=1
        sortedVotes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
        final.append(sortedVotes[0][0])
        #np.array(final)
    return np.asarray(final)


# In[9]:


def get_distance(X_train, X):
    #Reference: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
    dists = -2 * np.dot(X, X_train.T) + np.sum(X_train**2,    axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
    return dists


# In[34]:


def RandomForest(X_train, y_train, X_test, y_test):
#     References:
#     https://www.youtube.com/watch?v=LDRbO9a6XPU
#     https://www.youtube.com/watch?v=D_2LkhMJcfY&t=286s
#     https://towardsdatascience.com/decision-tree-from-scratch-in-python-46e99dfea775
    
    trees = []
    
    X_train2 = []
    for i in range(len(X_train)):
        arr = np.append(X_train[i], y_train[i])
        X_train2.append(arr)

    X_test2 = []
    for i in range(len(X_test)):
        arr = np.append(X_test[i], y_train[i])
        X_test2.append(arr)

    X_train2 = np.asarray(X_train2)    
    X_test2 = np.asarray(X_test2) 
        
    parameters = math.ceil(len(X_train2[0]) ** 0.5)
        
    for i in range(30):
        index = random.sample(range(0,X_train2.shape[1] - 2), parameters)
        index.append(X_train2.shape[1] - 1)
        data = np.asarray(random.sample(range(0,X_train2.shape[0]), 1000))
        trees.append(build_tree(X_train2[data], index))
        
    pred = []
    for row in X_test2:
        local_pred = []
        for tree_node in trees:
            local_pred.append(predict(tree_node, row))
            counts = np.bincount(local_pred)
        pred.append(np.argmax(counts))
    return pred       


# In[11]:


def predict(mytree, row):
    if(mytree.label is not None):
        return mytree.label
    col = mytree.col
    if(row[col] >= mytree.value):
        return predict(mytree.right, row)
    else:
        return predict(mytree.left, row)


# In[12]:


def build_tree(training_data, index):

    if(count(training_data)):
        return Node(None, None, None, None, training_data[0][-1])
            
    true_part, false_part, info_gain, value, column = data_split(training_data, index)
    if(info_gain == 0):
        return Node(None, None, None, None, training_data[0][-1])
    true_tree = build_tree(true_part, index)
    false_tree = build_tree(false_part, index)
    return Node(true_tree, false_tree, column, value, None)


# In[13]:


def count(rows):

    count = []
    for row in rows:
        if(row[-1] not in count):
            count.append(row[-1])    
    if (len(count) == 1):
        return True
    else:
        return False


# In[14]:


class Node:
    def __init__(self,truebranch, falsebranch, col, value, label = None):
        self.left = falsebranch
        self.right = truebranch
        self.col = col
        self.value = value
        self.label = label


# In[15]:


def data_split(rows, indices):
    
    current_impurity = gini_impurity(rows)
    best_gain = -1    
    part1 = []
    part2 = []
    v = 0
    c = 0   
    for i in indices:
            if(indices.index(i) == len(indices) - 1):
                pass
            else:
                data = []
                for entry in rows:
                    data.append(entry[i])            
                for val in data:
                    true_row, false_row = data_partition(rows, val, i)
                    if(len(true_row) == 0 or len(false_row) == 0):
                        continue
                    else:
                        gini1 = gini_impurity(true_row)
                        gini2 = gini_impurity(false_row)
                        total = len(true_row) + len(false_row)
                        gini3 = ((gini1 * len(true_row))/total) + ((gini2 * len(false_row))/total)
                        info_gain = current_impurity - gini3

                        if info_gain > best_gain:
                            best_gain = info_gain
                            part1 = true_row
                            part2 = false_row
                            v = val
                            c = i    
    return part1, part2, best_gain, v, c   


# In[16]:


def data_partition(rows, value, index):
    true_rows = []
    false_rows = []
    for row in rows:
        if row[index] >= value:
            true_rows.append(row)
        else:
            false_rows.append(row)

    return true_rows, false_rows


# In[17]:


def gini_impurity(rows):
    impurity = 1
    count = {}
    for row in rows:
        val = row[-1]
        if val in count.keys():
            count[val] += 1
        else:
            count[val] = 1
    
    for key, value in count.items():
        impure = value/len(rows)
        impurity = impurity - impure ** 2
    return impurity


# In[18]:


def PCA(X_train, N):
    # Reference
    # https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
    c_matrix = np.cov(X_train.T)
    solution= np.linalg.eig(c_matrix)
    e_vals = solution[0]
    e_vectors = solution[1]
    index = e_vals.argsort()[::-1]
    e_vals = e_vals[index]
    e_vectors = e_vectors[:,index]
    s=np.argsort(e_vals)[-N:]
    return (np.dot(X_train,e_vectors[s].T))


# In[17]:


def SklearnSupervisedLearning(X_train,y_train,X_test):
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    # Logistic Regression
    clf_lr = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred_lr = clf_lr.predict(X_test)

    
    # SVM
    clf_svm = svm.SVC(kernel = 'linear')
    clf_svm.fit(x_train, y_train)
    y_pred_svm = clf_svm.predict(x_test)

    
    # Decision Tree
    clf_dt = tree.DecisionTreeClassifier()
    clf_dt = clf_dt.fit(X_train, y_train)
    y_pred_dt = clf_dt.predict(X_test)

    
    # KNN
#     scaler = preprocessing.StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.fit_transform(X_test)
    knn_model = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean', p=2)
    knn_model.fit(X_train, y_train)
    y_pred_knn= knn_model.predict(X_test)

    return y_pred_lr, y_pred_svm, y_pred_dt, y_pred_knn


# In[27]:


def SklearnVotingClassifier(X_train,y_train,X_test):
    
    """
    :type X_train: numpy.ndarray
    :type X_test: numpy.ndarray
    :type Y_train: numpy.ndarray
    
    :rtype: List[numpy.ndarray] 
    """
    clf1 = LogisticRegression(solver = 'liblinear')
    clf2 = tree.DecisionTreeClassifier(max_depth = 10, criterion = 'entropy')
    clf3 = svm.SVC(kernel = 'linear')
    clf4 = KNeighborsClassifier(n_neighbors=11, metric = 'euclidean', p = 2)
    labels = ['Logistic Regression', 'Decision Tree', 'SVM', 'KNN']
    hard_voting_classifier = VotingClassifier(estimators = [(labels[0], clf1),
                                                        (labels[1], clf2),
                                                        (labels[2], clf3),
                                                        (labels[3], clf4)], voting = 'hard')
    hard_voting_classifier.fit(X_train, y_train)
    y_pred_ensemblemodel = hard_voting_classifier.predict(X_test)

    return y_pred_ensemblemodel


# In[13]:

# Functions to be called here
# dataset = pd.read_csv('data.csv')
# X = dataset.iloc[:, :-1]
# Y = dataset.iloc[:, 48]

# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 41)

# # Convert to NP array
# x_train = x_train.values
# x_test = x_test.values
# y_train = y_train.values
# y_test = y_test.values


# # In[39]:


# # K Means
# kmeans_cluster, centroids = KMeans(x_train, 11)
# my_wcss = WCSS(kmeans_cluster, centroids)
# print(my_wcss)


# # In[14]:


# # KNN

# y_pred = KNN(x_train, x_test, y_train, n=5)
# acc_knn = Accuracy(y_test, y_pred)
# print("Accuracy : ", acc_knn*100)
# print("Recall: ", Recall(y_test, y_pred) * 100)
# print("Precision: ", Precision(y_test, y_pred) * 100)


# print(ConfusionMatrix(y_pred, y_test))


# # In[28]:


# # PCA

# pca_result = PCA(x_train, 5)
# print("Result of PCA new shape: ", pca_result.shape)


# # In[35]:


# # Random Forest
# rf_pred = RandomForest(x_train, y_train, x_test, y_test)
# acc_rf = Accuracy(y_test, rf_pred)
# print("Accuracy : ", acc_rf * 100)
# print("Recall: ", Recall(y_test, rf_pred) * 100)
# print("Precision: ", Precision(y_test, rf_pred) * 100)

# print(ConfusionMatrix(rf_pred, y_test))


# # In[19]:


# # SkLearn Supervised Learning
# pred_lr, pred_svm, pred_dt, pred_knn = SklearnSupervisedLearning(x_train, y_train, x_test)
# print("Accuracy for LR: ", metrics.accuracy_score(y_test, pred_lr))
# print("Accuracy for SVM: ", metrics.accuracy_score(y_test, pred_svm))
# print("Accuracy for DT: ", metrics.accuracy_score(y_test, pred_dt))
# print("Accuracy for KNN: ", metrics.accuracy_score(y_test, pred_knn))


# # In[38]:


# # SkLearn Ensemble Model
# pred_ensemble = SklearnVotingClassifier(x_train, y_train, x_test)
# print("Accuracy for Ensemble Model: ", metrics.accuracy_score(y_test, pred_ensemble))


# In[ ]:





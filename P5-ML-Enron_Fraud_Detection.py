#This code is developed to process the Enron fraud in 2000
# This code was developed as part of a project while taking the Udacity Nano Degree program in Data Analyst in 4th Quarter of 2016
# coding: utf-8



from feature_format import featureFormat
from feature_format import targetFeatureSplit
from sklearn.feature_selection import SelectKBest


from copy import copy
import matplotlib.pyplot as plt
import sys
import pickle
sys.path.append("../tools/")

import pandas as pd
import matplotlib

import numpy as np
from tester import test_classifier, dump_classifier_and_data


from time import time
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC


from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


from sklearn.cluster import KMeans



# features_list is a list of strings, each of which is a feature name
# first feature must be "poi", as this will be singled out as the label
target_label = 'poi'
email_features_list = [
    # 'email_address', # remit email address; informational label
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages', 
    ]
financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]

features_list = [target_label] + financial_features_list + email_features_list 
NAN_value = 'NaN'

# load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

#total number of data points
print "Number of data point :" ,len(data_dict)
##allocation across classes (POI/non-POI)

count_poi=0
for name in data_dict:
    count_poi=count_poi+data_dict[name]['poi']  ## since poi is either 1 or 0, sum of it is the total numnber of poi
count_non_poi=len(data_dict)-count_poi
#print "Number of POI : ", count_poi
#print "Number of non_POI : ", count_non_poi
#print "This number of POI and non-POI is before removing outlier TOTAL"
#print "Number of financial_features_list",len(financial_features_list)
#print "Number email feature",len(email_features_list)

##number of features used
print"Number of features used :", len(features_list)-1 ### poi is not included for features used.
all_features = data_dict['ALLEN PHILLIP K'].keys()
print('Each person has %d features available' %  len(all_features))
### Evaluate dataset for completeness
missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
for person in data_dict.keys():
    records = 0
    for feature in all_features:
        if data_dict[person][feature] == 'NaN':
            missing_values[feature] += 1
        else:
            records += 1

# checking missing values for all financial features

financial_info_incomplete = []
for individual in data_dict.keys():  
    if data_dict[individual]['total_payments'] == 'NaN' and     data_dict[individual]['total_stock_value'] == 'NaN':
        financial_info_incomplete.append(individual)
print
if len(financial_info_incomplete) > 0:
    print('Individuals with missing data for payments and stock value:')
    records = 0
    for individual in financial_info_incomplete:
        print individual        
        records += data_dict[individual]['poi']
    print('Out of these %d individual %d are POIs' % (len(financial_info_incomplete), 
          records))
else:
    print('No individual with missing data for payments and stock value.')
print

data_dict.pop("CHAN RONNIE",0)
data_dict.pop("POWERS WILLIAM",0)
data_dict.pop("LOCKHART EUGENE E",0)


### Is anyone missing all email information?
incomplete_email = []
for person in data_dict.keys():
    if data_dict[person]['to_messages'] == 'NaN' and        data_dict[person]['from_messages'] == 'NaN':
        incomplete_email.append(person)    
if len(incomplete_email) > 0:
    print('The following people have no message data for emails:')
    counter = 0
    for person in incomplete_email:
        print("%s, POI: %r" % (person, data_dict[person]['poi']))        
        counter += data_dict[person]['poi']
    print('Of these %d people, %d of them are POIs' % (len(incomplete_email), counter))
else:
    print('No person is missing both to and from message records.')


# ### Handling outlires
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
### plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

#remove outliers
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)
data_dict.pop('TOTAL', 0)
print len(data_dict)

##total number of data points after removal of outliers
#print "Number of data point :" ,len(data_dict)
##allocation across classes (POI/non-POI)

count_poi=0
for name in data_dict:
    count_poi=count_poi+data_dict[name]['poi']  ## since poi is either 1 or 0, sum of it is the total numnber of poi
count_non_poi=len(data_dict)-count_poi
#print "Number of POI : ", count_poi
#print "Number of non_POI : ", count_non_poi
#print "This number of POI and non-POI is before removing outlier TOTAL"
#print "Number of financial_features_list",len(financial_features_list)
#print "Number email feature",len(email_features_list)

##number of features used
print"Number of features used :", len(features_list)-1 ### poi is not included for features used.

# instantiate copies of dataset and features for grading purposes
my_dataset = copy(data_dict)
my_features_list = copy(features_list)


# ### Creating new features
# I engineered a feature, poi_interaction which was a ratio of the total number of emails to and from a POI to the total number of emails sent or received.
# add new features
def add_features(my_dataset, my_features_list):
    """
    Given the data dictionary of people with features, adds some features to
    """
    for name in data_dict:

        # Add ratio of POI messages to total.
        try:
            total_messages = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
            poi_related_messages = data_dict[name]["from_poi_to_this_person"] +                                    data_dict[name]["from_this_person_to_poi"] +                                    data_dict[name]["shared_receipt_with_poi"]
            poi_ratio = 1.* poi_related_messages / total_messages
            data_dict[name]['poi_ratio_messages'] = poi_ratio
        except:
            data_dict[name]['poi_ratio_messages'] = NAN_value
# print "finished"
    return data_dict
data_dict = add_features(data_dict, my_features_list)

my_features_list= my_features_list+['poi_ratio_messages']

### Decision Tree importance feature selection 
# Currently not used
#from sklearn.tree import DecisionTreeClassifier
# pipeline_clf = DecisionTreeClassifier(min_samples_split=30, criterion='entropy', random_state=42)
#clf = DecisionTreeClassifier(min_samples_split=30, criterion='entropy', random_state=42)

## Determine DecisionTree Feature Importances
#clf = DecisionTreeClassifier()
#clf.fit(features, labels)
#tree_scores = zip(my_features_list[1:],clf.feature_importances_)
#sorted_dict = sorted(tree_scores, key=lambda feature: feature[1], reverse = True)
#for item in sorted_dict:
    #print item[0], item[1]

# SelectKBest for the new feature using f_regression
#new_feature_list = [target_label]+ ['poi_ratio_messages']
#from sklearn.feature_selection import SelectKBest, f_regression
#selector = SelectKBest(f_regression, k=1)
#selector.fit(features, labels)
#features = selector.transform(features)
#print features.shape
#pprint.pprint(selector.scores_)
#feature_scores = zip(new_feature_list[1:],selector.scores_)
#sorted_dict = sorted(feature_scores, key=lambda feature: feature[1], reverse = True)
#for item in sorted_dict:
    #print item[0], item[1]

# ### Feature slection
# score function
def score_func(y_true,y_predict):
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0

    for prediction, truth in zip(y_predict, y_true):
        if prediction == 0 and truth == 0:
            true_negatives += 1
        elif prediction == 0 and truth == 1:
            false_negatives += 1
        elif prediction == 1 and truth == 0:
            false_positives += 1
        else:
            true_positives += 1
    if true_positives == 0:
        return (0,0,0)
    else:
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        return (precision,recall,f1)



# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

import pprint
from sklearn.feature_selection import SelectKBest, f_classif
### Use SelectKBest to determine the optimum features to use
selector = SelectKBest(k=14)
selector.fit(features, labels)
features = selector.transform(features)
print features.shape
pprint.pprint(selector.scores_)
feature_scores = zip(my_features_list[1:],selector.scores_)
sorted_dict = sorted(feature_scores, key=lambda feature: feature[1], reverse = True)
for item in sorted_dict:]
    print item[0], item[1

# selecting features by combining ranked features using GaussianNB
def selectKBest(sorted_dict, data):
	result = []
	_k = 20
	for k in range(0,_k):
		my_features_list = ['poi']
		for n in range(0,k+1):
			my_features_list.append(sorted_dict[n][0])
        data = featureFormat(my_dataset, my_features_list, sort_keys = True, remove_all_zeroes = False)
        labels, features = targetFeatureSplit(data)
        features = [abs(x) for x in features]
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(features, labels)
        predictions = clf.predict(features)
        score = score_func(labels,predictions)
        result.append((k+1,score[0],score[1],score[2]))
        return result

# The top selecKbest 4 features 
my_feature_list = [target_label]+[
'exercised_stock_options', 
'total_stock_value', 
'bonus',
'salary']

# extract the features specified in features_list
data = featureFormat(my_dataset, my_feature_list)
# split into labels and features
labels, features = targetFeatureSplit(data)
# scale features via min-max
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
# ### Trying a varity of classifiers
# Classifier list
classifers_dict = {'AdaBoost': AdaBoostClassifier(n_estimators=100,
                                                  learning_rate=1),
                    'Random Forest': RandomForestClassifier(n_estimators=10,
                                                            # max_depth=None,
                                                            min_samples_split=50,
                                                            random_state=0),
                    'Naive Bayes': GaussianNB(),
                    'SVM': SVC(C=1, kernel='rbf', degree=2),
                    'Tree': DecisionTreeClassifier(),
                    'KNN': KNeighborsClassifier(n_neighbors=20, weights='distance', p=3)
                    }


for name, clf in classifers_dict.iteritems():
    print name
    t0 = time()
    clf.fit(features, labels)
    print "\tTime to train: ", time() - t0, "s"

    pred = clf.predict(features)
    acc = accuracy_score(labels, pred)
    print "\tPrediction accuracy: ", acc
    print "Precision: ", precision_score(labels, pred)
    print "Recall: ", recall_score(labels, pred)


# # cross validation of classifiers using tester.py
# GaussianNB
from sklearn.naive_bayes import GaussianNB
GN= GaussianNB()
t0 = time()
test_classifier(GN, my_dataset, my_feature_list)

# DecisionTree
from sklearn import tree
DT = tree.DecisionTreeClassifier()
test_classifier(DT, my_dataset, my_feature_list)

# Adaboost
from sklearn.ensemble import AdaBoostClassifier 
ab = AdaBoostClassifier(random_state=42) 
t0 = time() 
test_classifier(ab, my_dataset, my_feature_list, folds = 100) 
print("AdaBoost fitting time: %rs" % round(time()-t0, 3)) 


# ### Tuning classifiers with GridsearchCV
# Cross validation train test split
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn import grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import grid_search
from sklearn import grid_search

# Tuning Adaboost classfier
# Build pipeline for the algorithm
Pipeline2 = Pipeline([
        ('imp', Imputer(missing_values='NaN')),
        ('std', MinMaxScaler()),
        ('selection', SelectKBest()),
        ('pca', PCA()),
        ('clf', AdaBoostClassifier(DecisionTreeClassifier(max_depth=6),random_state=0))
    ])

# estimator parameters
k = [2,3,4]
#c = [1,2]
e = [100,50,25]
r = [0.001,0.01,0.1,1]

param_grid = {'selection__k': k,
              #'pca__n_components': c,
              #'imp__strategy': strategy = 0,
              'clf__n_estimators': e,
              'clf__learning_rate': r,
              
             }

# set model parameters to grid search object
gridCV_object = GridSearchCV(estimator = Pipeline2, 
                             param_grid = param_grid,
                             scoring = 'f1',
                             cv = StratifiedShuffleSplit(labels_train,100, test_size=0.3,random_state=42))

# train the model
gridCV_object.fit(features_train, labels_train)

print gridCV_object.best_params_
print gridCV_object.scorer_
pred_CV = gridCV_object.predict(features_test)
from sklearn.metrics import classification_report
print classification_report(labels_test, pred_CV)

## send the best parameter to clf
t0=time()
clf=gridCV_object.best_estimator_
test_classifier(clf, my_dataset, my_feature_list, folds = 1000)
t1=time()-t0
print t1

# ### Dumping classifier, dataset, and features_list
dump_classifier_and_data(clf, my_dataset, my_feature_list)
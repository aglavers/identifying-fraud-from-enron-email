#!/usr/bin/python

import sys
import pickle
from pprint import pprint
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# make selections for scaling and new features
use_new_features = False
use_scaling = False
features_list = ['poi', 
                 'shared_receipt_with_poi', 
                 'deferred_income', 
                 'salary', 
                 'long_term_incentive', 
                 'expenses', 
                 'exercised_stock_options']

### Load the dictionary containing the dataset
if use_new_features:
    data_dict = pickle.load(open("final_project_dataset_augmented_email_poi.pkl", "r") )
    features_list += ['email_poi']
else:
    data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
print '\nFeatures: '
pprint(features_list)
print "use_new_features: {}, use_scaling: {}".format(use_new_features, use_scaling)

### Task 2: Remove outliers
data_dict.pop( 'TOTAL', 0 ) 
data_dict.pop("LAY KENNETH L", 0)
data_dict.pop("FREVERT MARK A", 0)
data_dict.pop("BHATNAGAR SANJAY", 0)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# scaling
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(features)
scaled_features = min_max_scaler.transform(features)
                
# split test and train  using either scaled or unscaled features             
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(scaled_features if use_scaling else features,
                     labels, 
                     test_size=0.25, 
                     random_state=42)
 
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# instantiate DeciosionTree, fit,predict and score
from sklearn.metrics import f1_score
from sklearn import tree
opts = {'random_state' : 42}
clf = tree.DecisionTreeClassifier(**opts)
clf.fit(features_train, labels_train)
pred_labels_test = clf.predict(features_test)
print "DecisionTree: F1 score on test: {}".format(f1_score(labels_test, pred_labels_test))

# instantiate Kmeans, fit,predict and score
from sklearn.cluster import KMeans
opts = {'random_state' : 42}
clf = KMeans(**opts)
clf.fit(features_train, labels_train)
pred_labels_test = clf.predict(features_test)
print "KMeans: F1 score on test: {}".format(f1_score(labels_test, pred_labels_test))
 
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.

# set up a classifier and grid parameters
base_clf  = KMeans()
parameters = {'random_state' : [42], 
              'n_clusters'   : [4, 8, 16],
              'max_iter'     : [300, 1000, 10000],
              'init'         : ['k-means++', 'random'] }

# do the grid search and print results
print "KMeans Grid search ..."
from sklearn import grid_search
gs_clf = grid_search.GridSearchCV(base_clf, parameters, scoring='f1')
gs_clf.fit(features_train, labels_train)
clf =  gs_clf.best_estimator_
best_parameters = clf.get_params()
print "Best score: {:0.3f} with parameters:".format(gs_clf.best_score_)
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))  

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)


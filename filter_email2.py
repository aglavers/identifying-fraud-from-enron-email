#!/usr/bin/python
import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
##from parse_out_email_text import parseOutText

"""
The new feature, named email_poi, is built as follows:
- Email sent by all persons is munged into a single dictionary. 
  Trials showed that using the sent emails only produced 
  better predictions than all the email
- Each message is split to remove forwarded/replied text using 
  "-----" as delimiter. This makes the text more personal to the sender.
- Email text is pipelined through a CountVectorizer, TfidTransformer, 
  and SGDClassifier. Trials showed the SGD Classifier producing better 
  results than Naive Bayes.
- GridSearchCV is used to tune the classifier and find an optimal result

  Adapted from starter code to process the emails from Sara and Chris.

  output is to "final_project_dataset_augmented_email_poi.pkl" which
  is "final_project_dataset.pkl" with one new feature "email_poi"

"""
os.chdir("/Users/alavers/Source/UdacityIntroMachineLearning/final_project")
base_path = "../final_project/emails_by_address/"
base_path_email = "../"
base_path_project = "../final_project/"

### EMAIL MUNGING -
### - Build a dict of all email text matched to a person via email address.
### - remove forwarded/replied to text.
### - only consider text sent by person

## build a dict of email for each person
all_email = {}
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

for person_name in data_dict:
    email_address = data_dict[person_name]['email_address']
    if email_address == "NaN":
        print "No email adddress for {}".format(person_name)
        continue
    
    # This directory has a .txt file per person from_ and to_ which in turn
    # contains a list of filenames, 1 per email message. 
    
    ## build a list of files - trials showed that from_ files only were more accurate
    email_files = []
    for to_from in ['from_']:
        txt_file = "{}{}{}.txt".format(base_path, to_from, email_address)
        try:
            with open("{}from_{}.txt".format(base_path, email_address)) as emf:
                email_files  = email_files + emf.read().splitlines()
        except IOError:
            print "Not found: {}".format(txt_file)
            
    ## munge the email text in each email file to form
    ## a single string of all the email text for this person
    email_count = 0    
    all_text = ""   
    for email_file in email_files :         
        full_email_file = base_path_email + email_file
         
        # read email int 'text'
        email = open(full_email_file, "r")
        try: 
            text = email.read()
            email_count += 1
        except Exception as e:
            print "Exception on {0} file {1}. {2}".format(email_address, full_email_file, e)
            continue
            
        # forwards, repies and footers often have dash separators. Split
        # to use only the text that is more closely attributable to this person     
        append_text = text.split("-----")[0]
        
        # append 
        all_text += append_text
                           
    # print result for this email address and append if there were any hits
    print "Email address {} has {} messages".format(email_address, email_count)
  
    # Add to the output dict 
    all_email.update({person_name : {'email_address' : email_address,
                                     'poi' : data_dict[person_name]['poi'],
                                     'all_text' : all_text}}) 

# write to file to ease rerunning            
with open('{}all_email_data.pkl'.format(base_path_project),'w') as aef:
   pickle.dump(all_email, aef)
print "Dumped {} names".format(len(all_email))

### CLASSIFICATION TO PREDICT POI FROM EMAIL TEXT 
### - use munged email text 
### - pipeline adapted from  http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html#example-model-selection-grid-search-text-feature-extraction-py 

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer    
from sklearn.pipeline import Pipeline    
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score  
from pprint import pprint
from time import time

# load email text and build fature & label lists
all_email = pickle.load(open('{}all_email_data.pkl'.format(base_path_project),'r') )
print "Loaded {} names".format(len(all_email))

labels = []
features_text = []
for person_name in all_email:
    labels.append(all_email[person_name]['poi'])
    features_text.append(all_email[person_name]['all_text'])

# test/training split
features_text_train, features_text_test, labels_train, labels_test = \
    train_test_split(features_text, labels, test_size=0.25, random_state=422)

clf = Pipeline([('vect', CountVectorizer(stop_words='english', max_df=0.5, token_pattern="[a-z][a-z]+")),
                ('tfidf', TfidfTransformer(sublinear_tf=True)),
                ('sgd', SGDClassifier(loss='log', random_state=42))
                     ])
# grid search parameters - each prefix ties back to the pipeline
# and the vector specifies the alternate values to try                     
parameters = {
    'vect__max_df': (0.25, 0.5, 0.75),
    'vect__max_features': (None, 5000, 10000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'sgd__loss': ('hinge', 'log', 'modified_huber', 'squared_hinge')
    #'clf__alpha': (0.00001, 0.000001),
    #'clf__penalty': ('l2', 'elasticnet'),
    #'clf__n_iter': (10, 50, 80),
}

# do the grid search on the pipeline using training data
grid_search = GridSearchCV(clf, parameters, scoring="f1", n_jobs=-1, verbose=1)

print "Performing grid search..."
print "pipeline: {}".format([name for name, _ in clf.steps])
print "parameters: {}"
pprint(parameters)
t0 = time()
grid_search.fit(features_text_train, labels_train)

# output some results of the best combination
print "done in {:0.3f}s".format(time() - t0)
print "Best score: {:0.3f}".format(grid_search.best_score_)
print "Best parameters:"
best_clf =  grid_search.best_estimator_
best_parameters = best_clf.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))  
  
## f1 scores
labels_test_pred = best_clf.predict(features_text_test)
print "F1 score on test: {}".format(f1_score(labels_test, labels_test_pred))
labels_train_pred = best_clf.predict(features_text_train)
print "F1 score on train: {}".format(f1_score(labels_train, labels_train_pred))


### ADD THIS AS A NEW FEATURE ON THE FULL PERSON DATASET
### A new feature email_poi is the prediction of the test classifier 
###   -1 for no email, 
###   0  for not indicating a poi
###   1  for poi indication

#predict on the whole dataset and derive a new feature named email_poi
labels_pred = best_clf.predict(features_text)  
print "F1 score on all persons with email: {}".format(f1_score(labels, labels_pred))

# mark the new feature as missing (no email) with a -1 to account 
# for persons who have no email at all
for person_name in data_dict:
    person_dict = data_dict[person_name]
    person_dict.update({"email_poi" : -1})
    data_dict.update({person_name : person_dict})

# now mark the matching names as 0 (not a poi) or 1 (poi)  
ii = 0
for person_name in all_email:
    email_poi = 1 if labels_pred[ii] else 0
    try:
        person_dict = data_dict[person_name]
        person_dict.update({"email_poi" : email_poi})
        data_dict.update({person_name : person_dict})
    except KeyError as e:
        print 'Key Error', e
    ii+=1
    
pickle.dump(data_dict, open("final_project_dataset_augmented_email_poi.pkl", "w") )

# create  adatframe for easier review of results
import pandas as pd    
dfx = pd.DataFrame(data_dict).transpose()
dfx.corr()   
           
           
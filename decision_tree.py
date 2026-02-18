#-------------------------------------------------------------------------
# AUTHOR: Arianna Tarki
# FILENAME: decision_tree.py
# SPECIFICATION: Program that displays a decision tree from a given data table from a .csv file
# FOR: CS 4210- Assignment #1
# TIME SPENT: ~3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#encode the original categorical training features into numbers and add to the 4D array X.

feature_encoders = []       # array to encode labels to features

for col in range(4):        # for each feature (Age, Spec, Astig, Tear)
    
    le = LabelEncoder()
    column_data = [row[col] for row in db]
    encoded_column = le.fit_transform(column_data)
    feature_encoders.append(le)
    
    if col == 0:
        
        for value in encoded_column:      # for the first spot in the row
            
            X.append([value])
            
    else:
        
        for i, value in enumerate(encoded_column):      # for each spot after the first
           
            X[i].append(value)


#encode the original categorical training classes into numbers and add to the vector Y.

class_map = {}

for row in db:
  
    label = row[4]
    
    if label not in class_map:
      
        class_map[label] = len(class_map)
    
    Y.append(class_map[label])

#fitting the depth-2 decision tree to the data using entropy as your impurity measure

clf = DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf.fit(X, Y)


#plotting decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()
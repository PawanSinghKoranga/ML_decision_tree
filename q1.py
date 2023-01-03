import math
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
import matplotlib.pyplot as plt

df = pd.read_csv('Dataset_E.csv')
header = list(df.columns)

def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set([row[col] for row in rows])

def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def max_label(dict):
    max_count = 0
    label = ""

    for key, value in dict.items():
        if dict[key] > max_count:
            max_count = dict[key]
            label = key

    return label

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)

class Question:
    # A Question is used to partition a dataset.

    def __init__(self, column, value, header):
        self.column = column
        self.value = value
        self.header = header

    def match(self, example):
        # Compare the feature value in an example to the
        # feature value in this question.
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # This is a method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.header[self.column], condition, str(self.value))
    

def partition(rows, question):
    #Partitions a dataset.
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    # Calculate the Gini Impurity for a list of rows
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

def entropy(rows):
    # compute the entropy.
    entries = class_counts(rows)
    avg_entropy = 0
    size = float(len(rows))
    for label in entries:
        prob = entries[label] / size
        avg_entropy = avg_entropy + (prob * math.log(prob, 2))
    return -1*avg_entropy


def info_gain(left, right, current_uncertainty):
    # Information Gain.
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)

def find_best_split(rows, header):
    # iterating over every feature / value
    best_gain = 0  # to track the best information gain
    best_question = None
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1  # number of columns

    for col in range(n_features):  # for each feature
        values = set([row[col] for row in rows])
        for val in values:  # for each value
            question = Question(col, val, header)
            true_rows, false_rows = partition(rows, question)

            # Skip if it doesn't divide the dataset.
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            # Calculate the information gain from this split
            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question

class Leaf:
    # A Leaf node classifies data

    def __init__(self, rows, id, depth):
        self.predictions = class_counts(rows)
        self.predicted_label = max_label(self.predictions)
        self.id = id
        self.depth = depth

class Decision_Node:
    #A Decision Node asks a question and holds a reference to the question and the two child nodes.

    def __init__(self,
                 question,
                 true_branch,
                 false_branch,
                 depth,
                 id,
                 rows):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.depth = depth
        self.id = id
        self.rows = rows


def build_tree(rows, header, depth=0, id=0):

    gain, question = find_best_split(rows, header)

    # Base case: no further info gain
    # we'll return a leaf.
    if gain == 0:
        return Leaf(rows, id, depth)

    true_rows, false_rows = partition(rows, question)
    # Recursively build the true branch.
    true_branch = build_tree(true_rows, header, depth + 1, 2 * id + 2)
    # Recursively build the false branch.
    false_branch = build_tree(false_rows, header, depth + 1, 2 * id + 1)
    return Decision_Node(question, true_branch, false_branch, depth, id, rows)

def prune_tree(node, prunedList):
    # Base case
    if isinstance(node, Leaf):
        return node
    # If we reach a pruned node, the nodes below it are not considered
    if int(node.id) in prunedList:
        return Leaf(node.rows, node.id, node.depth)

    node.true_branch = prune_tree(node.true_branch, prunedList)
    node.false_branch = prune_tree(node.false_branch, prunedList)
    return node

def classify(row, node):
    # we've reached a leaf
    if isinstance(node, Leaf):
        return node.predicted_label

    # Compare the feature stored in the node to decide whether to follow true-branch/false-branch
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

def print_tree(node, spacing=""):
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + "Leaf id: " + str(node.id) + " Predictions: " + str(node.predictions) + " Label Class: " + str(node.predicted_label))
        return

    # Print the question at this node
   
    print(spacing + str(node.question) + " id: " + str(node.id) + " depth: " + str(node.depth))
    
    # Call this function recursively on the true branch
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")

    # Call this function recursively on the false branch
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")


def print_leaf(counts):
    """A nicer way to print the predictions at a leaf."""
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


def getLeafNodes(node, leafNodes =[],depth=[]):

    # Base case
    if isinstance(node, Leaf):
        leafNodes.append(node)
        depth.append(node.depth)
        return

    # Recursive right call for true values
    getLeafNodes(node.true_branch, leafNodes,depth)

    # Recursive left call for false values
    getLeafNodes(node.false_branch, leafNodes,depth)

    return leafNodes,depth


def getInnerNodes(node, innerNodes =[],Nodeids=[],innerdepth=[]):
    # Base case
    if isinstance(node, Leaf):
        return

    innerNodes.append(node)
    if node.id!=0:
        Nodeids.append(node.id)
        innerdepth.append(node.depth)

    # Recursive right call for true values
    getInnerNodes(node.true_branch, innerNodes,Nodeids,innerdepth)

    # Recursive left call for false values
    getInnerNodes(node.false_branch, innerNodes,Nodeids,innerdepth)

    return innerNodes,Nodeids,innerdepth

def computeAccuracy(rows, node):

    count = len(rows)
    if count == 0:
        return 0

    accuracy = 0
    for row in rows:
        # last entry of the column is the actual label
        if row[-1] == classify(row, node):
            accuracy += 1
    return round(accuracy/count, 2)


#Function to label values
#if MIN_Value <=val < (MIN_Value + Mean_Value) / 2 then label a
#if (MIN_Value + Mean_Value) / 2 <=val < Mean_Value then label b
#if (Mean_Value) <=val < (Mean_Value + MAX_Value)/2 then label c
#if (Mean_Value + MAX_Value)/2 <=val <= MAX_Value  then label d

def label(val, *boundaries):
    if (val < boundaries[0]):
        return 'a'
    elif (val < boundaries[1]):
        return 'b'
    elif (val < boundaries[2]):
        return 'c'
    else:
        return 'd'

#Function to discretize attributes into labels
def toLabel(df, old_feature_name):
    second = df[old_feature_name].mean()
    minimum = df[old_feature_name].min()
    first = (minimum + second)/2
    maximum = df[old_feature_name].max()
    third = (maximum + second)/2
    return df[old_feature_name].apply(label, args= (first, second, third))

df['Preg_labeled'] = toLabel(df, 'Pregnancies')
df['Glucose_labeled'] = toLabel(df, 'Glucose')
df['BP_labeled'] = toLabel(df, 'BloodPressure')
df['SkinThickness_labeled'] = toLabel(df, 'SkinThickness')
df['Insulin_labeled'] = toLabel(df, 'Insulin')
df['BMI_labeled'] = toLabel(df, 'BMI')
df['DPF_labeled'] = toLabel(df, 'DiabetesPedigreeFunction')
df['Age_labeled'] = toLabel(df, 'Age')
df['Outcome1'] = df['Outcome']

df.drop(['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','DiabetesPedigreeFunction','Age','BMI','Outcome'], axis = 1, inplace = True)


header = list(df.columns)

lst = df.values.tolist()

# splitting the data set into train and test
trainDF, testDF = model_selection.train_test_split(lst, test_size=0.2)

# building the tree
t = build_tree(trainDF, header)

# get leaf and inner nodes
print("\nLeaf nodes ****************")
leaves,depth = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))

print("\nNon-leaf nodes ****************")
innerNodes,Nodeids,innerdepth = getInnerNodes(t)

for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

# print tree
maxAccuracy = computeAccuracy(testDF, t)
print("\nTree before pruning with accuracy: " + str(maxAccuracy*100) + "\n")
print_tree(t)



#Training and testesting the classifier using Information Gain for 10 random splits for the given Dataset
arr = []
brr = []
for i in range(10):
    header = list(df.columns)
    x_train, x_test=model_selection.train_test_split(lst, test_size=0.2)
    t=build_tree(x_train,header)
    mAccuracy= computeAccuracy(x_test, t)
    arr.append(mAccuracy*100)
    leaves,depth = getLeafNodes(t)
    maxdepth=0
    for i in depth:
        if i>maxdepth:
            maxdepth=i
    brr.append(maxdepth)


#Training and testing completed
maxAcc = 0
for i in range(10):
    if arr[i] > maxAcc:
        maxAcc = arr[i]
        j = i
print("After 10 random splitting of the Dataset")
print("The best test accuracy", maxAcc)
print("The depth of that tree", brr[j])


# Pruning
accuracy_at_depth=[]
nodeIdToPrune = -1
for node in innerNodes:
    if node.id != 0:
        prune_tree(t, [node.id])
        currentAccuracy = computeAccuracy(testDF, t)
        print("Pruned node_id: " + str(node.id) + " to achieve accuracy: " + str(currentAccuracy*100) + "%")
        accuracy_at_depth.append(int(currentAccuracy*100))

        if currentAccuracy > maxAccuracy:
            maxAccuracy = currentAccuracy
            nodeIdToPrune = node.id
        t = build_tree(trainDF, header)

        if maxAccuracy == 1:
            break

if nodeIdToPrune != -1:
    print("\nPrinting the pruned tree")
    t = build_tree(trainDF, header)
    prune_tree(t, [nodeIdToPrune])
    print("\nFinal node Id to prune (for max accuracy): " + str(nodeIdToPrune))
else:
    t = build_tree(trainDF, header)
    print("\nPruning strategy did'nt increased accuracy")

print("\n********************************************************************")
print("*********** Final Tree with accuracy: " + str(maxAccuracy*100) + "%  ************")
print("********************************************************************\n")
print_tree(t)


# plotting accuracy vs depth
maxdepth=0
for i in depth:
    if i>maxdepth:
        maxdepth=i

A=[]
A= [0 for i in range(maxdepth-1)]

for i in range(len(accuracy_at_depth)):
    if A[innerdepth[i]-1]<accuracy_at_depth[i]:
        A[innerdepth[i]-1]=accuracy_at_depth[i]
d=[]
d=[i+1 for i in range(0,maxdepth-1)]
plt.plot(d,A);
plt.xlabel('Depth');
plt.ylabel('Accuracy in %');
plt.title('Accuracy with varying depth');

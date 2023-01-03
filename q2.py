import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import math

df = pd.read_csv("Dataset_E.csv")

print("Total number of rows in the original Dataset: ",len(data))

def outlier(df, column_name):
    upper_bound = (df[column_name].mean()+ (3* df[column_name].std()))
    return upper_bound

columns = df.columns

for col in columns:
    if col!= "Outcome":
        upper_bound = outlier(df,col)
        df.drop(df[(df[col]>upper_bound)].index,inplace = True)
        
print("Total number of rows in the Dataset after dropping outlier samples: ",len(df))

print("Dataset before normalization:")
print(df)
for col in columns:
    if col!="Class_att":
        df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())

print("Dataset after normalization:")
print(df)


def accuracy_score(y_true, y_pred):
    return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)

def pre_processing(df):
    X = df.drop([df.columns[-1]], axis = 1)
    y = df[df.columns[-1]]
    return X, y

def train_test_split(x, y, test_size = 0.25, random_state = None):
    x_test = x.sample(frac = test_size, random_state = random_state)
    y_test = y[x_test.index]
    x_train = x.drop(x_test.index)
    y_train = y.drop(y_test.index)
    return x_train, x_test, y_train, y_test

class  NaiveBayes:
    def __init__(self):
        # likelihoods: Likelihood of each feature per class
        # class_priors: Prior probabilities of classes
        # features: All features of dataset
        self.features = list
        self.likelihoods = {}
        self.class_priors = {}
        self.X_train = np.array
        self.y_train = np.array
        self.train_size = int
        self.num_feats = int

    def fit(self, X, y):
        self.features = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape[0]
        self.num_feats = X.shape[1]
        for feature in self.features:
            self.likelihoods[feature] = {}
            for outcome in np.unique(self.y_train):
                self.likelihoods[feature].update({outcome:{}})
                self.class_priors.update({outcome: 0})

        self._calc_class_prior()
        self._calc_likelihoods()

    def _calc_class_prior(self):
        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size

    def _calc_likelihoods(self):
        for feature in self.features:
            for outcome in np.unique(self.y_train):
                self.likelihoods[feature][outcome]['mean'] = self.X_train[feature][self.y_train[self.y_train == outcome].index.values.tolist()].mean()
                self.likelihoods[feature][outcome]['variance'] = self.X_train[feature][self.y_train[self.y_train == outcome].index.values.tolist()].var()


    def predict(self, X):
        results = []
        X = np.array(X)

        for query in X:
            probs_outcome = {}
            for outcome in np.unique(self.y_train):
                prior = self.class_priors[outcome]
                likelihood = 1
                evidence_temp = 1

                for feat, feat_val in zip(self.features, query):
                    mean = self.likelihoods[feat][outcome]['mean']
                    var = self.likelihoods[feat][outcome]['variance']
                    likelihood *= (1/math.sqrt(2*math.pi*var)) * np.exp(-(feat_val - mean)**2 / (2*var))

                posterior_numerator = (likelihood * prior)
                probs_outcome[outcome] = posterior_numerator

            result = max(probs_outcome, key = lambda x: probs_outcome[x])
            results.append(result + 0.0001)

        return np.array(results)



inputs = df.drop('Outcome',axis='columns')
target = df.Outcome

cross_validity=[]
for t in range(10):
    from header_created_by_pawan_tamal import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)
    model = GaussianNB()
    model.fit(X_train,y_train)
    scores=accuracy_score(y_test,model.predict(X_test))
    cross_validity.append(scores)

print("Below are the final accuracies for ten-fold cross validation:")
print(cross_validity)


import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import joblib
import os

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Function to get relative paths
def get_relative_path(relative_path):
    return os.path.join(BASE_DIR, relative_path).replace('/', '\\')

col_names = ['buying','maint','doors' ,'persons','lug_boot','safety','class']

# Load dataset
df = pd.read_csv(get_relative_path("data/car.data"), names=col_names)

lb = LabelEncoder()
for i in df.columns:
    df[i] = lb.fit_transform(df[i])

Xfeatures = df[['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']]
ylabels = df['class']

X_train, X_test, Y_train, Y_test = train_test_split(Xfeatures, ylabels, test_size=0.30, random_state=7)

# Using - Logisitic Regression
logit = LogisticRegression()
logit.fit(X_train, Y_train)
logit_file = os.path.join("models", "LogisticRegression.pkl")
joblib.dump(logit, logit_file)
print("Logistic Regression Accuracy Score:", accuracy_score(Y_test, logit.predict(X_test)))

# Using - Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, Y_train)
nb_file = os.path.join("models", "NaiveBayes.pkl")
joblib.dump(nb, nb_file)
print("Naive Bayes Accuracy Score:", accuracy_score(Y_test, nb.predict(X_test)))

# Using Neural Network
nn_clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
nn_clf.fit(X_train,Y_train)
nn_clf_file = os.path.join("models", "MLPClassifier.pkl")
joblib.dump(nn_clf, nn_clf_file)
print("Neural Network Accuracy Score:", accuracy_score(Y_test, nn_clf.predict(X_test)))


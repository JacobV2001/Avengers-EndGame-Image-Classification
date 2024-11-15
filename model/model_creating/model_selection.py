import pandas as pd
import joblib
import json

# libraries for model training, grid search, and evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from model_training import prepare_data


# function to define the models and their hyperparameters
def get_model_params():
    return [
        {
            'name': 'svm',
            'model': SVC(gamma='auto', probability=True),
            'params': {'svc__C': [1, 10, 100, 1000], 'svc__kernel': ['rbf', 'linear']}
        },
        {
            'name': 'random_forest',
            'model': RandomForestClassifier(),
            'params': {'randomforestclassifier__n_estimators': [1, 5, 10]}
        },
        {
            'name': 'logistic_regression',
            'model': LogisticRegression(solver='liblinear'),
            'params': {'logisticregression__C': [1, 5, 10]}
        }
    ]

# function to select best model & save it
def model_selection_and_storage():
    
    # prepare x adn y features and get class dictionary
    X, y, class_dict = prepare_data()

    # split data into train & test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # collect results of GridSearch
    scores = [] # list to store performance of each model
    best_estimators = {} # dictionary to store the best model
    
    # loop through each algorithm anf perform GridSearchCV
    for model_info in get_model_params():
        pipe = make_pipeline(StandardScaler(), model_info['model']) # standardize data before model fitting
        clf = GridSearchCV(pipe, model_info['params'], cv=5, return_train_score=False) #implement GridSearch w/ 5 folds cross-validation
        clf.fit(X_train, y_train)

        scores.append({
            'model': model_info['name'], # name of model
            'best_score': clf.best_score_, # best score from GridSearch
            'best_params': clf.best_params_ # best params for model
        })
        best_estimators[model_info['name']] = clf.best_estimator_ # store the best model for the current algorithm

    # print scores
    df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
    print(df)

    # select and evaluate model
    best_model = best_estimators['svm']
    print(f"SVM Test Accuracy: {best_model.score(X_test, y_test)}")
    print("Classification Report for SVM:")
    print(classification_report(y_test, best_model.predict(X_test)))

    # save model and class dictionary
    joblib.dump(best_model, 'saved_model.pkl')
    with open('class_dictionary.json', 'w') as f:
        json.dump(class_dict, f)

    print("Model and class dictionary saved successfully.")

# run function if script is executed directly 
if __name__ == "__main__":
    model_selection_and_storage()

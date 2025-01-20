from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

random_state = 42

def calculate_metrics(model, X_test, Y_test, threshold=0.5):
    predicted_probs = model.predict_proba(X_test)[:, 1]
    predicted_labels = [1 if prob >= threshold else 0 for prob in predicted_probs]
    return get_scores(predicted_labels, Y_test)

def get_scores(predicted_labels, Y_test):
    # Calculate metrics
    accuracy = accuracy_score(Y_test, predicted_labels)
    precision = precision_score(Y_test, predicted_labels)
    recall = recall_score(Y_test, predicted_labels)
    f1 = f1_score(Y_test, predicted_labels)
    cm = confusion_matrix(Y_test, predicted_labels)
    fpr, tpr, _ = roc_curve(Y_test, predicted_labels)
    roc_auc = auc(fpr, tpr)

    return accuracy, precision, recall, f1, cm, fpr, tpr, roc_auc

# Metric Printing Function
def print_metrics(accuracy, precision, recall, f1, cm, fpr, tpr, roc_auc):
    print("-" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    # print(f"AUC: {roc_auc:.4f}")
    # print("Confusion Matrix:")
    # print(cm)
    print("-" * 50)
    
    

def decision_tree_gs(param_grid, X_train, X_test, Y_train, Y_test):
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=random_state),
        param_grid=param_grid,
        scoring='accuracy',      
        cv=5,              
        n_jobs=-1,
                  
    )
    grid_search.fit(X_train, Y_train)

    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)

    # Calculate metrics
    metrics = calculate_metrics(best_clf, X_test, Y_test)
    
    print_metrics(*metrics)

    return best_clf, best_params, *metrics


# Logistic Regression with Grid Search
def logistic_regression_gs(param_grid, X_train, X_test, Y_train, Y_test, threshold=0.5):
    grid_search = GridSearchCV(
        estimator=LogisticRegression(max_iter=1000, random_state=random_state),
        param_grid=param_grid,
        scoring='accuracy',  
        cv=5,               
        n_jobs=-1,
                   
    )
    grid_search.fit(X_train, Y_train)
    
    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)

    # Calculate metrics
    metrics = calculate_metrics(best_clf, X_test, Y_test)

    print_metrics(*metrics)

    return best_clf, best_params, *metrics



# Random Forest Grid Search
def random_forest_gs(param_grid, X_train, X_test, Y_train, Y_test):
    
    # Perform GridSearchCV for fine-tuning
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=random_state),
        param_grid=param_grid,
        scoring='accuracy',      
        cv=5,              
        n_jobs=-1,
                  
    )
    grid_search.fit(X_train, Y_train)

    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)

    # Calculate metrics
    metrics = calculate_metrics(best_clf, X_test, Y_test)
    
    print_metrics(*metrics)

    return best_clf, best_params, *metrics



# Support Vector Machine with Grid Search
def svm_gs(param_grid, X_train, X_test, Y_train, Y_test):
    # Perform GridSearchCV for fine-tuning
    grid_search = GridSearchCV(
        estimator=SVC(probability=True, random_state=random_state),
        param_grid=param_grid,
        scoring='accuracy',      
        cv=5,              
        n_jobs=-1,
                  
    )
    grid_search.fit(X_train, Y_train)

    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)

    # Calculate metrics
    metrics = calculate_metrics(best_clf, X_test, Y_test)
    
    print_metrics(*metrics)

    return best_clf, best_params, *metrics



# Complement Naive Bayes with Grid Search
def cnb_gs(param_grid, X_train, X_test, Y_train, Y_test):
    grid_search = GridSearchCV(
        estimator=ComplementNB(),
        param_grid=param_grid,
        scoring='accuracy',      
        cv=5,              
        n_jobs=-1,
                  
    )
    grid_search.fit(X_train, Y_train)

    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)

    # Calculate metrics
    metrics = calculate_metrics(best_clf, X_test, Y_test)
    
    print_metrics(*metrics)

    return best_clf, best_params, *metrics



def neural_network_gs(param_grid, X_train, X_test, Y_train, Y_test):
    # Perform GridSearchCV for fine-tuning
    grid_search = GridSearchCV(
        estimator=MLPClassifier(random_state=random_state, max_iter=1000),
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        
    )
    grid_search.fit(X_train, Y_train)

    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)

    # Calculate metrics
    metrics = calculate_metrics(best_clf, X_test, Y_test)
    
    print_metrics(*metrics)

    return best_clf, best_params, *metrics


def knn_gs(param_grid, X_train, X_test, Y_train, Y_test):
    # Perform GridSearchCV for fine-tuning
    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        
    )
    grid_search.fit(X_train, Y_train)

    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_
    print("Best Hyperparameters:", grid_search.best_params_)

    # Calculate metrics
    metrics = calculate_metrics(best_clf, X_test, Y_test)
    
    print_metrics(*metrics)

    return best_clf, best_params, *metrics





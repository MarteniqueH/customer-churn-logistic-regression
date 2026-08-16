import pandas as pd
from src.preprocessing import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

data = pd.read_csv("data/CustomerChurnData.csv")
data = preprocessing(data)

#get mutiple feature columns 
X = data.iloc[:,:-1]
#Single traget column 
Y = data.iloc[:,-1]

train_X,test_X,train_y,test_y = train_test_split(
    X,
    Y,

    test_size = .2,

    random_state = 42,
)

def logistic_regression():
    
        #Hyperparameter Grid for search over

        hyperparameter_grid = {

            "C" : [0.01,0.1,1.0,10,100],
            "solver" : ["lbfgs"],
            "max_iter" : [5000],
            "class_weight" : [None,"balanced"]
            }

        
       
        grid_search = GridSearchCV(
              estimator = LogisticRegression(),
              param_grid=  hyperparameter_grid,
              cv = 5,
              scoring = "f1", 
              n_jobs = -1,
              verbose = 1
        )

            

        grid_search.fit(train_X,train_y)

        print("BEST HYPERPARAMETERS FOUND FROM GRID SEARCH: ")
        print("____________________________")
        print(grid_search.best_params_)
        print(f"The best Cross Validation F1 score: {grid_search.best_score_: .2%}")

        #The best model should be found with the grid search of the best hyperparamters 
        lr_model = grid_search.best_estimator_

        print("The cross validation scores: ")
        print("____________________________")

        #list of metrics that will be calculated 
        metrics = ["accuracy", "recall", "f1", "roc_auc"]
        #loop through metrcis 
        for metric in metrics:
                #complete a  5-fold cross validation 
                scores = cross_val_score(lr_model,train_X,train_y, cv = 5, scoring= metric)
                #Display the average score and standard deviation of the metric 
                print(f"{metric} : mean value = {scores.mean() :.2%} Standard Deviation: {scores.std() : .2%}")

        y_perdictions = lr_model.predict(test_X)

        accuracy = accuracy_score(test_y, y_perdictions)
        recall = recall_score(test_y, y_perdictions)
        f1Score = f1_score(test_y, y_perdictions)
        y_probabilities = lr_model.predict_proba(test_X)[:,1]
        rocAuc = roc_auc_score(test_y, y_probabilities)

        print("MODEL PERFORMANCE REPORT: ")
        print("____________________________")
        #Measures overall performance of model
        print(f"Accuracy Score: {accuracy: .2%}")
        #Measures how many positive cases correctly identified 
        print(f"Recall Score: {recall:.2%}")
        #Balances the percision and recall for imbalanced classes 
        print(f"F1 Score: {f1Score: .2%}")
        #Measures the model's ability to distinguish between classes 
        print(f"ROC-AUC: {rocAuc : .2%}" )


logistic_regression()




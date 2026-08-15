import pandas as pd
from src.preprocessing import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
    lr_model = LogisticRegression(
        #Controls regularization
        C=1.0,
        #L2 regularization
        l1_ratio = 0 ,
        #algorithm used to train model
        solver ="lbfgs",
        #gives the model many interations to converge
        max_iter= 1000,
        #all classes treated equally 
        class_weight = None
        

    )

    lr_model.fit(train_X,train_y)

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




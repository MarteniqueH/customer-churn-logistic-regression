
import pandas as pd
from src.preprocessing import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
 
data = pd.read_csv("data/CustomerChurnData.csv")
data = preprocessing(data)

#get mutiple feature columns 
X = data.iloc[:,:-1]
#Single traget column 
Y = data.iloc[:,-1]

#Data split between train and test
train_X,test_X, train_y ,test_y = train_test_split (
    X,
    Y,
    #Data split will be 80% train 20% test 
    test_size = 0.2,
    #makes random choice repeatable 
    random_state = 42
)




def gradient_boosting_classifier():
    gbc_model = GradientBoostingClassifier (
                #reduces the contribution of each tree
                    #reduces the contribution of each tree
                    learning_rate = 0.2, 
                    #number of trees to be built 
                    n_estimators = 100, 
                    #minimum number of samples required to split 
                    min_samples_split = 2,
        
                    #minimum number of samples required in a leaf 
                    min_samples_leaf = 3,
        
                    #maximum depth of each tree
                    max_depth = 3
        

               
                )

    gbc_model.fit(train_X, train_y)

    y_perdictions = gbc_model.predict(test_X)

    #Evaluation Metrics 

    accuracy = accuracy_score(test_y, y_perdictions)
    recall = recall_score(test_y, y_perdictions)
    f1Score = f1_score(test_y, y_perdictions)
    y_probabilities = gbc_model.predict_proba(test_X)[:,1]
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





    
   
gradient_boosting_classifier()




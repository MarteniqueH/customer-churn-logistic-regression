
import pandas as pd
from src.preprocessing import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, classification_report
 
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
    model = GradientBoostingClassifier (
        #reduces the contribution of each tree
                learning_rate = 0.2, 
                #number of trees to be built 
                n_estimators = 100, 
                #minimum number of samples required to split 
               min_samples_split = 2,
    
                #minimum number of samples required in a leaf 
               min_samples_leaf = 3,
    
                #maximum depth of each tree
                max_depth = 3)

    model.fit(train_X, train_y)

    y_perdictions = model.predict(test_X)

    accuracy = accuracy_score(test_y, y_perdictions)
    print(f"Accuracy Score: {accuracy:.4f}")
    print(classification_report(test_y, y_perdictions))
   
gradient_boosting_classifier()




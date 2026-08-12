import pandas as pd 


data = pd.read_csv("data/CustomerChurnData.csv")


def preprocessing(data):
#DATA PREPROCESSING 
    print("Data Preprocssing: ")
    print("______________________")


    # Pandas is viewing TotalCharges as an object (text)
    # even though the column contains numerical values.
    # Convert TotalCharges from object to a numerical data type.
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors = "coerce")
    # Confirm that TotalCharges was converted to a numerical data type.
    print(data["TotalCharges"].dtype)

    #Found 11 customers with NaN for total charges
    print(data[data["TotalCharges"].isnull()])
    #Replace the missing TotalCharges with 0 for customers with 0 months of tenture 
    data["TotalCharges"] = data["TotalCharges"].fillna(0)

    #prints column names, data types and non-null count
    print("Data Info")
    print("______________________")
    print(data.info())

    #checks the number of missing values in each column
    print("Data Missing Values:")
    print("______________________")
    print(data.isnull().sum())

    #checks if duplicate rows exist 
    print("Duplicate Row Count: ")
    print("______________________")
    print(data.duplicated().sum())

    #Statistical data for numberical columns (mean,min,max, and standard deviation)
    print("Statisical Values for Numerical Columns")
    print("______________________")
    print(data.describe())



    #Checking categorical columns 
    #All columns with data type as object are considered categorical columns 
    print("Categorical Variables and Frequencies")
    print("______________________")
    categorical_columns = data.select_dtypes(include ="object").columns.drop("customerID")
    print(categorical_columns )

    #For each categorical column print the values with numerical count 
    for column in categorical_columns: 
        print(data[column].value_counts())
data = preprocessing(data)
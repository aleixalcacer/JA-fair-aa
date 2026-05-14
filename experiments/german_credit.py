import pandas as pd
from ucimlrepo import fetch_ucirepo 
import fairlearn.datasets as ds


def get_german_credit_data():
    # fetch dataset 
    statlog_german_credit_data = fetch_ucirepo(id=144) 
    
    # data (as pandas dataframes) 
    X = statlog_german_credit_data.data.features 
    y = statlog_german_credit_data.data.targets 
    
    male = ["A91", "A93", "A94"]
    female = ["A92", "A95"]

    # change Attribute 9 to binary

    X['Attribute9'] = X['Attribute9'].apply(lambda x: 1 if x in male else 0)

    p_a = pd.DataFrame(X['Attribute9'])
    p_a.columns = ['gender']

    # remove Attribute 9 from features
    X = X.drop(columns=['Attribute9'])

    # convert all categorical features to numeric using one-hot encoding
    X = pd.get_dummies(X, drop_first=True)


    return X, y, p_a


def get_healthcare_data():
    # fetch dataset 
    healthcare_data = fetch_ucirepo(id=45) 
    
    # data (as pandas dataframes) 
    X = healthcare_data.data.features 
    y = healthcare_data.data.targets  # pd.DataFrame

    # convert y to binary (0/1)
    y = (y == 0).astype(int).squeeze()  # 1 if survived, 0 if died

    p_a = X["sex"]

    # remove sex from features
    X = X.drop(columns=["sex"])

    X = pd.get_dummies(X, drop_first=True)


    # remove rows with missing values
    X = X.dropna()
    y = y[X.index]
    p_a = p_a[X.index]

    return X, y, p_a


def get_adult_data():
    adult = ds.fetch_adult()

    X = adult.data
    y = adult.target
    y = (y == ">50K").astype(int)


    p_a = X["sex"]
    p_a = (p_a == "Male").astype(int)

    X = X.drop(columns=["sex"])
    X = pd.get_dummies(X, drop_first=True)

    return X, y, p_a





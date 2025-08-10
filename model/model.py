import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def get_cleaned_data():

    full_path = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_csv('./data/data.csv')

    df.columns

    df = df.drop(['id','Unnamed: 32'],axis=1)

    df.isnull().sum()
    
    df.dropna()
    
    df['diagnosis'] = df['diagnosis'].map({"M":1,"B":0})    
    return df


def model_data(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    
    x = data.drop(['diagnosis'],axis = 1)
    y = data['diagnosis']    
    # Scaling Data
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    
    model = LogisticRegression()
    model.fit(x_train,y_train)
    
    pred = model.predict(x_test)
    
    from sklearn.metrics import accuracy_score,classification_report
    
    print("Accuracy of the model:", accuracy_score(pred,y_test))
    print("Classification report of the model:", classification_report(pred,y_test))
    
    print("Train Score",model.score(x_train,y_train))
    print("Test Score",model.score(x_test,y_test))
    
    return model,scaler


def main():
    data = get_cleaned_data()
    
    model= model_data(data)
    scaler = model_data(data)
    
    import pickle
    
    with open('./model/model.pkl','wb') as f:
        pickle.dump(model,f)
    
    with open('./model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)
        

if __name__ == '__main__':
  main()


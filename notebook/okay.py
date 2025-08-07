import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

full_path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f"{full_path}/datasets/diabetes.csv")

df.info()
df.describe()
df.isnull().sum()

x = df.drop(columns=['Outcome'],axis=1)
y = df['Outcome']


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x)

standard = scaler.transform(x)

x = standard
y = df['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

from sklearn.svm import SVC

model = SVC(kernel='linear')

model.fit(x_train,y_train)

pred = model.predict(x_test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(pred,y_test)

print(acc)


sample = (1,85,66,29,0,26.6,0.351,31)


sample_array = np.asarray(sample)

sample_reshaped = sample_array.reshape(1,-1)


std_dat = scaler.transform(sample_reshaped)
# print(std_dat)

prediction = model.predict(sample_reshaped)

print(prediction)

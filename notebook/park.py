import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

full_path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f'{full_path}/datasets/parkinsons.csv')

df[df.duplicated()]
df.isnull().sum()
df.columns
df.head()

x = df.drop(columns=['status','name'],axis=1)
y = df['status']


from sklearn.preprocessing import StandardScaler

scalers = StandardScaler()

scalers.fit(x)

standard = scalers.transform(x)

x = standard
y = df['status']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)


from sklearn.svm import SVC

model = SVC(kernel='linear')

model.fit(x_train,y_train)

pred = model.predict(x_test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(pred,y_test)

print(acc)

sample = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

sample_array = np.asarray(sample)

sample_reshaped = sample_array.reshape(1,-1)

std_data = scalers.transform(sample_reshaped)

prediction = model.predict(std_data)

print(prediction)

import pickle

pickle.dump(model,open('park.wsv','wb'))
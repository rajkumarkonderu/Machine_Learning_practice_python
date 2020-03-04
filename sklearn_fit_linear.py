import numpy as np 
import pandas as pd 
#import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import preprocessing, svm 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error,mean_squared_error


import csv
import pdb
import matplotlib.pyplot as plt
import numpy as np
#pdb.set_trace()
X=[]
Y=[]
X_test=[]
with open('insurance.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print('Column names are',row)
            line_count += 1
        else:
            X.append(int(row[0]))
            join_seperator='.'
            Y.append(float(join_seperator.join(row[1].split(','))))
            line_count += 1
X_train1=np.array(X[:60])
X_train=X_train1.copy()
X_train=np.expand_dims(X_train,axis=1)
Y_train=np.array(Y[:60])
X_test=X[60:]
Y_test=Y[60:]
# Create linear regression object
regr =LinearRegression()
# Train the model using the training sets
regr.fit(X_train, Y_train)
#print(regr.fit)
plt.plot(X_train1,Y_train,'ro')

#plt.plot(list(X_train1),list(Y_train),'-r',label='Y=B0+B1*X')
B0=regr.intercept_
B1=regr.coef_
print (B0,B1)
Y_pred_train=B0+B1*X_train
#Y_pred_test=B0+B1*X_test
plt.plot(list(X_train1),list(Y_pred_train),'-r',label='Y=B0+B1*X')
plt.show()
X_test=np.array(X[60:])
#X_test=np.expand_dims(X_test,axis=1)
for i in X_test:
	temp=i.reshape(-1,1)
	print(regr.predict(temp))
for i in Y_test:
        #ste#mp1=i.reshape(-1,1)
	print(i) 
score=regr.score(X_train,Y_train.reshape(-1, 1))  
print(score) 
#mse1=regr.score(Y_test.reshape(-1,1),Y_test)
#print(mse1)
mse= mean_squared_error(Y_train.reshape(-1,1),Y_pred_train.reshape(-1,1))
print (mse)	
mae= mean_absolute_error(Y_train.reshape(-1,1),Y_pred_train.reshape(-1,1))
print (mae)
    

 







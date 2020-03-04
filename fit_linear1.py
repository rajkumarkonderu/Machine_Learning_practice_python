##Fitting linear regression for auto insurance
###The dataset is called the “Auto Insurance in Sweden” dataset and involves predicting the total payment for all the claims in thousands of Swedish Kronor (y) given the total number of claims (x).
import csv
import pdb
import matplotlib.pyplot as plt
import numpy as np
#pdb.set_trace()
X_train=[]
Y_train=[]
with open('insurance.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print('Column names are',row)
            line_count += 1
        else:
            X_train.append(int(row[0]))
            join_seperator='.'
            Y_train.append(float(join_seperator.join(row[1].split(','))))
            line_count += 1
X_train=np.array(X_train)
Y_train=np.array(Y_train)
#X_train=np.array(1,2,2)
def mean(X_train):
        return(np.mean(X_train))     
mean_x = mean(X_train)
mean_y=mean(Y_train)
print ("mean",mean_x)
print ("mean",mean_y)
plt.plot(X_train,Y_train,'ro')
#plt.show()
def variance(values, mean):
	return sum([(val-mean)**2 for val in values])
def covariance(x,mean_x,y,mean_y):
	return sum(np.multiply((x-mean_x),(y-mean_y)))
def predict(x,b0,b1):
	return (b0+b1*x)
def predict_x(y,b0,b1):
        return(int((y-b0)/b1))
def num_train_examples(values):
	return (values.shape[0])
def MSE(y,prd_y):  #MSE means mean square error, also called as l2 error
        m=num_train_examples(y) #m is number of training examples in a given dataset
        return (sum((y-prd_y)**2)/m)
def MAE(y,prd_y):
        m=num_train_examples(y)
        return (sum(np.absolute(y-prd_y))/m)
variance_x=variance(X_train,mean_x)
variance_y=variance(Y_train,mean_y)
covariance_xy=covariance(X_train,mean_x,Y_train,mean_y)
B1=covariance_xy/variance_x
B0=mean_y-B1*mean_x
Y_pred=B0+B1*X_train

print (B0,B1)
print ("The value of y when x=130:",predict(130,B0,B1))
print ("The value of y when x=50:",predict(50,B0,B1))
print ("The value of y when x=110:",predict(110,B0,B1))
print ("The value of y when x=80:",predict(80,B0,B1))
print (" the value of x when y=300:",predict_x(300,B0,B1)) 
print ("the Mean Square errors:",MSE(Y_train,Y_pred))
print ("the Mean absolute errors:",MAE(Y_train,Y_pred))


plt.plot(list(X_train),list(Y_pred),'-r',label='y=B0+B1*X')
plt.show()






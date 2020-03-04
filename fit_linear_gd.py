import numpy as np
import matplotlib.pyplot as plt
def preprocess_data(csv_file_name):
	f=open(csv_file_name,'r')
	data=f.readlines()
	X=[]
	Y=[]
	for i in data[1:]:
		split_line=i.split('\t')
		X.append(int(split_line[0]))
		Y.append(float('.'.join(split_line[1][:-1].split(','))))
	m=len(X)
	print(X)
	print (Y)
	plt.plot(np.array(X),np.array(Y),'ro')
	#plt.plot(np.array(X),np.array(Y),'ro')
	X=np.array(X)
	if X.ndim==1:
		X=np.array(X).reshape(-1,1)
	else:
		X=X
	#X=np.array(X).reshape(-1,1)
	Y=np.array(Y).reshape(-1,1)
	ones_array=np.ones((m,1))
	X=np.concatenate((ones_array,X),axis=1)
	return (X,Y,m)
def compute_cost(X,y,theta,m):
	y_pred=np.dot(X,theta)  #X*theta
	cost=(1/2*m)*sum(np.power((Y-y_pred),2))
	return y_pred,cost

def update_parameters(theta,learning_rate,y,y_pred,m):
	theta[0]=theta[0]-learning_rate*sum((y_pred-Y))/m
	theta[1]=theta[1]-learning_rate*np.sum(np.dot(np.transpose(y_pred-Y),X),axis=1)/m
	return theta
def predict(X,theta):
	return (sum(np.dot(X,theta)))#(theta[0]+theta[1]*X)
def plot_hypothesis(X,y):
	plt.plot(X,y,'-g',label='y=B0+B1*X')
	plt.show()
def learning_rate_check(num_iter,costs):
	plt.plot(range(num_iter),costs,'-r')
	plt.show()
#def draw_costs():

X,Y,m=preprocess_data('insurance.csv')
#Number of features =n
n=X.shape[1]
#Predicting y based on hypothesis h=theta0+theta1*x =X*theta
theta=np.zeros((n,1))
learning_rate=0.00001
num_iterations=1000
costs=[]
for i in range(num_iterations):
	y_pred,cost=compute_cost(X,Y,theta,m)
	costs.append(cost)
	theta=update_parameters(theta,learning_rate,Y,y_pred,m)
	#plt.plot(list(X[:,1]),y_pred,'-r',label='y=B0+B1*X')
print (cost,theta)
plotting=False
if plotting==True:
	plot_hypothesis(list(X[:,1]),y_pred)
check_learning=False
if check_learning==True:
	learning_rate_check(num_iterations,costs)

#h=theta[0]+theta[1]*X
print ("the hypothesisis h={}+{}*X".format(float(theta[0]),float(theta[1])))
print (theta[0],theta[1])
print ("The value of h when x=130:",predict(np.array([1,130]),theta))



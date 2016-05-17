import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes='true')
import sklearn.linear_model as sklm
import sklearn.cross_validation as skcv
import sklearn.metrics as skmetric

"""
Function to make scatter plot of the data colored according to species
"""
def scatter_plot(X, y, filename):
	x1 = []
	x2 = []
	for i in range(len(y)):
		if y[i] == -1:
			x1.append(X[i])
		else:
			x2.append(X[i])
	x1 = np.array(x1)
	x2 = np.array(x2)
	plt.scatter(x=x1[:,2], y=x1[:,1], color='r', label='Iris-setosa')
	plt.scatter(x=x2[:,2], y=x2[:,1], color='g', label='Iris-versicolor')
	plt.xlabel('Petal length')
	plt.ylabel('Sepal length')
	plt.legend(loc='lower right')
	plt.show()
	plt.savefig(filename)

"""
Main function
"""
if __name__ == '__main__':
	# loading and formatting the data
	df_all = pd.read_csv("iris_data.csv", names = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species'])
	df_two = df_all[df_all.Species.isin(['Iris-setosa','Iris-versicolor'])]
	m = len(df_two)
	df = pd.concat([pd.Series([1]*m), df_two[['SepalLength','PetalLength','Species']]], axis=1)
	df.columns = ['X0','X1','X2','y']
	x = df[['X0','X1','X2']].values
	# Assign the class -1 to Iris-setosa and 1 to Iris-versicolor
	y = []
	for val in df['y'].values:
		if val == 'Iris-setosa':
			y.append(-1)
		else:
			y.append(1)
	y = np.array(y)
	# divide the data into training and test data 
	x_train, x_test, y_train, y_test = skcv.train_test_split(x, y, test_size=0.4)
	# get the number of training examples (m) and number of features (n)
	m = x_train.shape[0]
	n = x_train.shape[1]
	
	# Perceptron algorithm
	eta = 0.01
	w = np.zeros(n)
	# iterate through the data and update the weights
	for i in range(m):
		z = np.dot(x_train[i],w)
		if z<0:
			h = -1
		else:
			h = 1
		for j in range(n):
			error = y_train[i] - h
			w[j] = w[j] + eta*error*x_train[i][j]
	# final weights	
	print("Weights: ", w)	
	
	# Calculate accuracy on training and test data
	predictions = []	
	for x in x_train:
		z = np.dot(x,w)
		if z<0:
			predictions.append(-1)
		else:
			predictions.append(1)
	print("Accuracy on training data: ", skmetric.accuracy_score(y_true=y_train, y_pred=predictions))
	
	predictions_test = []
	for x in x_test:
		z = np.dot(x,w)
		if z<0:
			predictions_test.append(-1)
		else:
			predictions_test.append(1)
	print("Accuracy on test data: ", skmetric.accuracy_score(y_true=y_test, y_pred=predictions_test))
	
	# Plot the results
	scatter_plot(x_train, predictions, "training_pred.png")
	scatter_plot(x_train, y_train, "training_labels.png")
	scatter_plot(x_test, predictions_test, "test_pred.png")
	scatter_plot(x_test, y_test, "test_labels.png")	
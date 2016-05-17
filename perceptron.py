import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes='true')
import sklearn.linear_model as sklm
import sklearn.cross_validation as skcv
import sklearn.metrics as skmetric

df_all = pd.read_csv("iris_data.csv", names = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species'])
df_two = df_all[df_all.Species.isin(['Iris-setosa','Iris-versicolor'])]
df = df_two[['SepalLength','PetalLength','Species']]

X_train,X_test,y_train,y_test = skcv.train_test_split(df[['SepalLength', 'PetalLength']], 
                                                     df.Species, test_size=0.4)
perceptron_model = sklm.Perceptron(n_iter=100).fit(X_train,y_train)
predictions = perceptron_model.predict(X_train)
print("Accuracy on training data: ", skmetric.accuracy_score(y_true=y_train, y_pred=predictions))
predictions_test = perceptron_model.predict(X_test)
print("Accuracy on test data: ", skmetric.accuracy_score(y_true=y_test, y_pred=predictions_test))

X_train['predictions'] = predictions
X_train['Labels'] = y_train
sns.lmplot(x='PetalLength', y='SepalLength', data=X_train, hue='predictions', fit_reg=False)
plt.savefig("train_pred.png")
sns.lmplot(x='PetalLength', y='SepalLength', data=X_train, hue='Labels', fit_reg=False)
plt.savefig("train_labels.png")

X_test['predictions'] = predictions_test
X_test['Labels'] = y_test
sns.lmplot(x='PetalLength', y='SepalLength', data=X_test, hue='predictions', fit_reg=False)
plt.savefig("test_pred.png")
sns.lmplot(x='PetalLength', y='SepalLength', data=X_test, hue='Labels', fit_reg=False)
plt.savefig("test_labels.png")

# References
# 
# 1. 'SI-370 Assignment 9'
# 2. sklearn.linear_model.Perceptron documentation
# 3. http://stamfordresearch.com/scikit-learn-perceptron/



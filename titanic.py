# Laura Niss
# Project1 Kaggle.com 'Titanic: Machine Learning from Disaster'

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import csv as csv
import sys

def import_data(train,test): 	# function 1
	'''
	Imports csv files `train` and `test` into pandas dataframes
	'''
	with open(train, "r") as train_filename:
		data = pd.read_csv(train_filename, header=0)
	with open(test, 'r') as test_filename:
		test_data = pd.read_csv(test_filename, header=0)
	return data, test_data

def median_fill(variable, new_variable, data):
	median_variable = np.zeros((2,3))
	for i in range (0,2): 	# for loop 1
		for j in range (0,3):	#for loop 2
			median_variable[i,j] = data[(data.Gender == i) & (data.Pclass == j+1)][variable].dropna().median() # 2x3 array with 6 numerical variables
	data[new_variable]= data[variable]
	for i in range (0,2):	# for loop 3
		for j in range (0,3): # for loop 4
			data.loc[(data[variable].isnull()) & (data.Gender == i) & (data.Pclass== j+1), new_variable] = median_variable[i,j]

def clean_data(data):	# function 2
	'''
	Converts varaibles with object parameters to ints and fills missing data. Drops unnecessary variables.
	'''
	data['Sex'] = data['Sex'].fillna(value='NA')
	data['Gender'] = data['Sex'].map({'female':1, 'male':0, 'NA':3}).astype(int)

	data['Embarked'] = data['Embarked'].fillna(value='NA')
	data['Port']= data['Embarked'].map({'S':0, 'C':1, 'Q':2, 'NA':3}).astype(int)

	median_fill('Age', 'Age_fill', data)
	median_fill('Fare', 'Fare_fill', data)

	data['Family_size']= data.SibSp + data.Parch

	data['Age_null'] = data['Age'].isnull()

	fare = 10
	while fare <=80:
		data[fare] = data['Fare_fill'].apply(lambda x: 1 if x <= fare else 0)
		fare = fare+10

	age = 10
	while age <=40:
		data[age] = data['Age_fill'].apply(lambda x: 1 if x <= age else 0)
		age = age+10

	df = data.drop(['Age', 'Embarked', 'Name', 'Ticket', 'Cabin', 'Sex','Fare'], axis=1)

	return df

def format_data(train, test): #function 3
	'''
	Makes train_data and test_data into same sized NumPy arrays.
	'''
	train_data = clean_data(train)
	train_df = train_data.drop(['PassengerId'], axis = 1)
	train_df = train_df.values
	test_df = clean_data(test)
	ids = test_df['PassengerId']
	test_df = test_df.drop(['PassengerId'], axis = 1)
	test_df = test_df.values
	return train_df, test_df, ids

if __name__ == '__main__':
	'''
	Use training data to predict survival of test data. Output csv submission file.
	'''
	train = 'train.csv'
	test = 'test.csv'

	data, test_data = import_data(train, test)
	train_df, test_df, ids = format_data(data, test_data)

	estimators = input('Number of estimators\n') # string 3
	int_est = int(estimators) # num variable

	forest = RandomForestClassifier(n_estimators=int_est)
	forest = forest.fit( train_df[:,1:], train_df[:,0] )

	output = forest.predict(test_df).astype(int)

	submission = open("submission.csv", "w") # file output
	open_file_object = csv.writer(submission)
	open_file_object.writerow(["PassengerId","Survived"])
	open_file_object.writerows(zip(ids, output))
	submission.close()

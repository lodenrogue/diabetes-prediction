import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

prediction_label = "Outcome"

def run():
	X_train, X_test, y_train, y_test = create_training_data()
	X2_train, X2_test, y2_train, y2_test = create_testing_data()

	X2 = get_testing_data_input_columns()

	# Scale data
	scaler = MinMaxScaler().fit(X2)
	X_train = scaler.transform(X_train)
	X2_train = scaler.transform(X2_train)

	models = [
		LinearSVC,
		SGDClassifier,
		BaggingClassifier,
		RandomForestClassifier,
		ExtraTreesClassifier,
		AdaBoostClassifier,
		GradientBoostingClassifier
	]

	for model in models:
		classifier = model()
		classifier.fit(X_train, y_train)

		# Score on testing data
		score = classifier.score(X2_train, y2_train)
		print(model)
		print("Score:", score)

		# Input values: Number of Pregnancies, Bloog Sugar Level, BMI, Age
		input = scaler.transform([[0,141,42.6,36]])
		print("Prediction:", classifier.predict(input))


def create_training_data():
	df = pd.read_csv("training_data.csv")
	X = np.array(df.drop(columns=[prediction_label, "BloodPressure", "SkinThickness", "Insulin", "DiabetesPedigreeFunction"]))
	y = np.array(df[prediction_label])
	
	return train_test_split(X, y, test_size=0.01)


def create_testing_data():
	df = pd.read_csv("diabetes.csv")
	X = np.array(df.drop(columns=[prediction_label, "BloodPressure", "SkinThickness", "Insulin", "DiabetesPedigreeFunction"]))
	y = np.array(df[prediction_label])

	return train_test_split(X, y, test_size=0.01)


def get_testing_data_input_columns():
	df = pd.read_csv("diabetes.csv")
	return np.array(df.drop(columns=[prediction_label, "BloodPressure", "SkinThickness", "Insulin", "DiabetesPedigreeFunction"]))


if __name__ == "__main__":
	run()

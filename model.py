import numpy as np
import pandas as pd
import pickle as pkl

import seaborn as sns
from matplotlib import style
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE


def load_dataset(path_to_dataset='./data/'):
	train_data = pd.read_csv(path_to_dataset + 'aug_train.csv')
	test_data = pd.read_csv(path_to_dataset + 'aug_test.csv')
	
	return train_data, test_data

def clean_data(data):
	data = data.copy()
	#city 
	data.city = data.city.apply(lambda city : int(city.split('_')[-1]))

	# gender
	data['gender'] = data['gender'].fillna('Unknown')
	data.gender = data.gender.map({'Male': 0, 'Female': 1, 'Other': 2, 'Unknown': 3})
	
	#relevant_experience
	data.relevent_experience = data.relevent_experience.map({'Has relevent experience': 1, 'No relevent experience': 0})

	#enrolled_university
	data.enrolled_university = data.enrolled_university.fillna('Unknown')
	data.enrolled_university = data.enrolled_university.map({'no_enrollment': 0, 'Full time course': 1, 'Part time course': 2, 'Unknown': 3})

	#education_level
	data.education_level = data.education_level.fillna('Unknown')
	data.education_level = data.education_level.map({'Graduate': 0, 'Masters': 1, 'High School': 2, 'Phd': 3, 'Primary School': 4, 'Unknown': 5})

	#major_discipline
	data.major_discipline = data.major_discipline.fillna('Unknown')
	data.major_discipline = pd.Categorical(data.major_discipline, categories=['STEM', 'Business Degree', 'Arts', 'Humanities', 'No Major',
		'Other', 'Unknown'], ordered=False)
	data.major_discipline = data.major_discipline.cat.codes

	#company_size
	def company_size_id( csz ):
		if csz == '<10':
			return 0
		if csz == '50-99':
			return 1
		if csz == '100-500':
			return 2
		if csz == '500-999':
			return 3
		if csz == '1000-4999':
			return 4
		if csz == '10/49':
			return 5
		if pd.isnull(csz):
			return 6
		if csz == '5000-9999':
			return 7
		if csz == '10000+':
			return 8

	data['company_size'] = data['company_size'].apply(company_size_id)

	#experience_id
	def experience_id( csz ):
		if pd.isnull(csz):
			return 0
		if csz == '>20':
			return 21
		if csz == '<1':
			return .5
		else:
			return int(csz)
	data['experience'] = data['experience'].apply(experience_id)

	#company_type
	data.company_type = data.company_type.fillna('Unknown')
	data.company_type = pd.Categorical(data.company_type, categories=['Unknown', 'Pvt Ltd', 'Funded Startup', 'Early Stage Startup', 'Other',
		'Public Sector', 'NGO'], ordered=False)
	data.company_type = data.company_type.cat.codes

	#last_new_job
	def last_new_job( csz ):
		if pd.isnull(csz) or csz == 'never':
			return 0
		if csz == '>4':
			return 5
		else:
			return int(csz)
	data['last_new_job'] = data['last_new_job'].apply(last_new_job)
 
	return data


def train_model():
	
	#load dataset
	train_data, test_data = load_dataset()
	
	train_clean_data = clean_data(train_data.drop(columns=['enrollee_id']))
	test_clean_data = clean_data(test_data.drop(columns=['enrollee_id']))
	
	X_train = train_clean_data.drop(columns=['target'])
	y_train = train_clean_data['target']
	
	#split dataset
	X_train, X_test, y_train, y_test = train_test_split(train_clean_data.drop('target', axis=1), train_clean_data['target'], test_size=0.2, random_state=42)

	#pipeline and model
	categorical_features = ['city', 'gender', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'company_size', 'company_type', 'last_new_job']
	numerical_features = ['city_development_index', 'experience', 'training_hours']

	categorical_transformer = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='most_frequent')),
		('onehot', OneHotEncoder(handle_unknown='ignore'))
	])

	numerical_transformer = Pipeline(steps=[
		('imputer', SimpleImputer(strategy='median')),
		('scaler', StandardScaler())
	])

	preprocessor = ColumnTransformer(transformers=[
		('cat', categorical_transformer, categorical_features),
		('num', numerical_transformer, numerical_features)
	])

	classifier = RandomForestClassifier(n_estimators=100,
													class_weight='balanced',
													random_state=42)

	model = make_pipeline(   
							preprocessor,
							SMOTE(random_state=42),
							classifier
						)
	
	#model training
	model.fit(X_train, y_train)
	
	#save model
	with open('model.pkl' , 'wb') as f:
  		pkl.dump(model, f)
	
 
def load_model(path_to_model='./model.pkl'):
	with open(path_to_model, 'rb') as f:
		model = pkl.load(f)

	return model
 
if __name__ == '__main__':
	train_model()


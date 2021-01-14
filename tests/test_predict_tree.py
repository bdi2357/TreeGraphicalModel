import pandas as pd 
import os,sys,re
import joblib
import pydotplus
import joblib
import time
from sklearn import tree
from os.path import expanduser
import pickle
sys.path.insert(0,"..")
from predict_tree import decsion_path_visualization
if __name__ == "__main__":
	input_model = os.path.join("..","test_input_data","iris_target_decision_tree_classifier.pkl")
	input_file = os.path.join("..","test_input_data","iris_sample.csv")

	home = expanduser("~")
	dest_dir = os.path.join(home,"test_decsion_path_visualization")
	if not os.path.isdir(dest_dir):
		os.mkdir(dest_dir)
	with open(input_model,'rb') as fp:
		clf = pickle.load(fp)
	f_path = os.path.abspath(input_model)[:-len(os.path.basename(input_model))]
	with open(os.path.join(f_path,"target.pkl"),'rb') as fp:
		class_names = pickle.load(fp)

	data = pd.read_csv(input_file)

	dot_data = tree.export_graphviz(clf, out_file=None,feature_names=data.columns,class_names = class_names ,filled=True, rounded=True,special_characters=True,max_depth=3)


	decsion_path_visualization(clf,dot_data,data,class_names,os.path.join(dest_dir,"test_prediction_iris.png"))


	
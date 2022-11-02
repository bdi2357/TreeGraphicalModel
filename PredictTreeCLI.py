import argparse
from predict_tree import decsion_path_visualization
import pickle
import pydotplus
import joblib
import time
import os,re,sys
from sklearn import tree
import pandas as pd
from label_transformer import transform_pred
"""
Example:
python PredictTreeCLI.py --InputFile iris_sample.csv --InputModel ../tree_test/iris_target_decision_tree_classifier.pkl --Dest ../tree_test/ --Name graphical_tree_pred3.png
python PredictTreeCLI.py --InputFile ../output_census2/sample.csv --InputModel ../output_census2/salary_target_decision_tree_classifier.pkl --Dest ../output_census2/ --Name graphical_tree_pred.png
python PredictTreeCLI.py --InputFile ../output_census3/sample.csv --InputModel ../output_census3/salary_target_decision_tree_classifier.pkl --Dest ../output_census3/ --Name graphical_tree_pred.png
python PredictTreeCLI.py --InputFile ../output_census4/samp_basic.csv --InputModel ../output_census4/salary_target_decision_tree_classifier.pkl --Dest ../output_census4/ --Name graphical_tree_pred.png

"""
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Interface to create graphical prediction of the tree model')
	parser.add_argument('--InputFile', dest='input_file',  help='<Required> The file destination if the input file')
	parser.add_argument('--InputModel', dest='input_model',  help='<Required> The file destination of the model')
	parser.add_argument('--Dest', dest='dest_dir',  help='<Required> destination directory' )
	parser.add_argument('--Name', dest='name',  help='<Required> name' )
	args = parser.parse_args()
	if not os.path.isdir(args.dest_dir):
		os.mkdir(args.dest_dir)
	if args.input_file and args.dest_dir  :
		print("input_file %s"%args.input_file )
		print("dest_dir %s"%args.dest_dir )
		
	else:
		print("ERROR !!! BAD INPUT ")
		exit(0)
	print(args.input_model)
	with open(args.input_model,'rb') as fp:
		clf = pickle.load(fp)
	f_path = os.path.abspath(args.input_model)[:-len(os.path.basename(args.input_model))]
	with open(os.path.join(f_path,"target.pkl"),'rb') as fp:
		class_names = pickle.load(fp)
	if os.path.isfile(os.path.join(f_path,"LabelTransform.pkl")):

		with open(os.path.join(f_path,"LabelTransform.pkl"),"rb") as fp:
			D_label_t = pickle.load(fp)
	else:
		D_label_t = {}


	data = pd.read_csv(args.input_file)
	data = transform_pred(data,D_label_t)

	print(clf.predict(data))

	print(list(data.columns))
	data.to_csv(os.path.join(args.dest_dir,"sample_transformed.csv"),index = False)

	dot_data = tree.export_graphviz(clf, out_file=None,feature_names=data.columns,class_names = class_names ,filled=True, rounded=True,special_characters=True)

	print(len(dot_data))
	decsion_path_visualization(clf,dot_data,data,class_names,os.path.join(args.dest_dir,args.name))


	
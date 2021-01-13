import argparse
from predict_tree import decsion_path_visualization,decsion_path_visualization2
import pickle
import pydotplus
import joblib
import time
import os,re,sys
from sklearn import tree
import pandas as pd
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
	data = pd.read_csv(args.input_file)
	dot_data = tree.export_graphviz(clf, out_file=None,feature_names=data.columns ,filled=True, rounded=True,special_characters=True,max_depth=3)


	decsion_path_visualization2(clf,dot_data,data,os.path.join(args.dest_dir,args.name))


from CreateTreeModel import create_tree
import argparse
import pydotplus
from sklearn.datasets import load_iris
import pandas as pd
import os,sys,re
if __name__ == "__main__":
	"""
	iris = load_iris()
	df = pd.DataFrame(iris.data)
	print(type(iris))
	df["target"] = iris.target
	print(df.head())
	df.to_csv("iris.csv",index=False)
	"""
	"""
	iris = pd.read_csv("iris.csv")
	data = iris
	target_col = "target"
	excluded_strs = []
	output_dir = ".."
	name = "iris"
	create_tree(data,target_col,excluded_strs,output_dir,name,max_depth=3)
	"""
	
	parser = argparse.ArgumentParser(description='Interface to create graphical tree model')
	parser.add_argument('--InputFile', dest='input_file',  help='<Required> The file destination if the input file')
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
	
	"""
	if args.input_file.find(".xlsx") > -1:
		args.input_file = excel2csv(args.input_file,tmp_dir)
	"""
	data = pd.read_csv(args.input_file)
	target_col = "target"
	excluded_strs = []
	output_dir = args.dest_dir
	name = args.name
	create_tree(data,target_col,excluded_strs,output_dir,name,max_depth=3)


from CreateTreeModel import create_tree
import argparse
from sklearn.datasets import load_iris
import pandas as pd
import os,sys,re
from label_transformer import prepare_data
import pickle
"""
usage example:
python CreateTreeCLI.py --InputFile iris.csv --Dest ../tree_test --Name iris
python CreateTreeCLI.py --InputFile ../output_census2/data.csv --Dest ../output_census2 --Name salary
python CreateTreeCLI.py --InputFile ../output_census3/transformed_input.csv --Dest ../output_census3 --Name salary
python CreateTreeCLI.py --InputFile ../output_census3/data.csv --Dest ../output_census4 --Name salary

"""
if __name__ == "__main__":
	
	
	parser = argparse.ArgumentParser(description='Interface to create graphical tree model')
	parser.add_argument('--InputFile', dest='input_file',  help='<Required> The file destination if the input file')
	parser.add_argument('--Dest', dest='dest_dir',  help='<Required> destination directory' )
	parser.add_argument('--Name', dest='name',  help='<Required> name' )
	parser.add_argument('--TreeMaxDepth', dest='max_depth',  help='<Recomended> tree maximal depth default value is 3' )

	args = parser.parse_args()
	if not os.path.isdir(args.dest_dir):
		os.mkdir(args.dest_dir)
	if args.input_file and args.dest_dir  :
		print("input_file %s"%args.input_file )
		print("dest_dir %s"%args.dest_dir )
		
	else:
		print("ERROR !!! BAD INPUT ")
		exit(0)
	if args.max_depth:
		max_depth = args.max_depth
	else:
		max_depth = 3
	
	
	data = pd.read_csv(args.input_file)
	target_col = "target"
	excluded_strs = []
	output_dir = args.dest_dir
	data = prepare_data(data,output_dir,tar_col = "target",save = True)
	if os.path.isfile(os.path.join(output_dir, "TargetLabel2Int.pkl")):
		with open(os.path.join(output_dir, "TargetLabel2Int.pkl"),"rb") as fp:
			tar_dict = pickle.load(fp)
		class_names = list(tar_dict.keys())
	else:
		class_names = []
	name = args.name
	create_tree(data,target_col,excluded_strs,output_dir,name,max_depth=max_depth,class_names=class_names)

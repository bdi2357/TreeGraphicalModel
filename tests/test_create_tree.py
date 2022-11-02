

import pydotplus
from sklearn.datasets import load_iris
import pandas as pd
import os,sys,re
from os.path import expanduser
import pickle
sys.path.insert(0,"..")
from CreateTreeModel import create_tree
if __name__ == "__main__":
	
	home = expanduser("~")
	dest_dir = os.path.join(home,"test_decsion_path_visualization")
	if not os.path.isdir(dest_dir):
		os.mkdir(dest_dir)	
	input_file = os.path.join("..","test_input_data","iris.csv") 
	
	max_depth = 4
	data = pd.read_csv(input_file)
	target_col = "target"
	excluded_strs = []
	output_dir = dest_dir
	name = "iris"
	create_tree(data,target_col,excluded_strs,output_dir,name,max_depth=max_depth)
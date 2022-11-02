import pandas as pd 
import numpy as np
import os
import time
from sklearn.tree import DecisionTreeClassifier
#from CreateTreeModel import create_tree
from collections import OrderedDict
import pickle
def fix_val(df,col,vals):
    for v in vals:
        df[str(v)] = df.apply(lambda r: r[col] == v,axis=1)

def pd_col_encoder(df,col):
	vals = df[col].drop_duplicates().tolist()
	fix_val(df,col,vals)
	return [str(v) for v in vals]
def objects_encoder(df,dest):
	cols_of_interest = [c for c in df.columns if df.dtypes[c] == np.dtype('object') and c!="target"]
	D_cols = OrderedDict()
	for c in cols_of_interest:
		D_cols[c] = pd_col_encoder(df,c)
	df = df.drop(cols_of_interest,axis=1)
	print("printing D_cols\n",D_cols)
	print(os.path.join(dest,"LabelTransform.pkl"))
	with open(os.path.join(dest,"LabelTransform.pkl"),'wb') as fp:
		pickle.dump(D_cols,fp)
		print("in dump")
	return df
def convert_target(df,tar_col = "target"):
	L = df[tar_col].drop_duplicates().tolist()
	D = OrderedDict([(x,L.index(x)) for x in L])
	df[tar_col] = df.apply(lambda r: D[r[tar_col]],axis=1)
	return df,D
def transform_pred(sample,label_t):
	for c in label_t.keys():
		print(c,label_t[c])
		for v in label_t[c]:
		    sample[v] = sample.apply(lambda r: v == r[c],axis=1)
	return sample.drop(list(label_t.keys()),axis=1)

def prepare_data(df,dest,tar_col = "target",save = False):
	df = objects_encoder(df,dest)
	df,D_tar = convert_target(df,tar_col)
	if save:
		df.to_csv(os.path.join(dest,"data_transformed.csv"),index=False)
	if not os.path.isfile(os.path.join(dest,"TargetLabel2Int.pkl")):
		with open(os.path.join(dest,"TargetLabel2Int.pkl"),'wb') as fp:
			pickle.dump(D_tar,fp)
	return df



if __name__ == "__main__":
	exit(0)
	output_dir = "../output_census"
	df_new = pd.read_csv("../Downloads/adult.data")
	df_new.head()
	fl = open("../Downloads/adult.columns","r")
	rd = fl.read()
	len(rd.split("\n"))
	rd.split("\n")[1]
	cols = [c.split(":")[0] for c  in rd.split("\n")]
	df_new.columns = cols 
	st = time.time()
	df_new = objects_encoder(df_new,output_dir)
	df_new,label_dict = convert_target(df_new,"target")
	print("objects_encoder time is %0.2f"%(time.time() - st))
	print(df_new.head())
	"""
	DTC = DecisionTreeClassifier(max_depth = 6)

	X = df_new[df_new.columns[:-1]]
	y = df_new["target"]
	DTC.fit(X,y)
	"""
	data = df_new
	target_col = "target"
	excluded_strs = []
	data.to_csv(os.path.join(output_dir,"data_transformed.csv"),index=False)
	name = "salary"
	class_names = [x[0] for x in label_dict.items()]
	create_tree(data,target_col,excluded_strs,output_dir,name,max_depth=5,class_names=class_names)

	




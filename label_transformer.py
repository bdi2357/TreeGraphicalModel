import pandas as pd 
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from CreateTreeModel import create_tree
def pd_col_encoder(df,col):
    vals = set(df[col])
    for v in vals:
        df["is "+str(v)] = df.apply(lambda r: r[col] == v,axis=1)

def objects_encoder(df):
	cols_of_interest = [c for c in df.columns if df.dtypes[c] == np.dtype('object') and c!="target"]
	for c in cols_of_interest:
		pd_col_encoder(df,c)
	df = df.drop(cols_of_interest,axis=1)
	return df
def convert_target(df,tar_col = "target"):
	L = df[tar_col].drop_duplicates().tolist()
	D = {x: L.index(x) for x in L}
	df[tar_col] = df.apply(lambda r: D[r[tar_col]],axis=1)
	return df,D

if __name__ == "__main__":

	df_new = pd.read_csv("../Downloads/adult.data")
	df_new.head()
	fl = open("../Downloads/adult.columns","r")
	rd = fl.read()
	len(rd.split("\n"))
	rd.split("\n")[1]
	cols = [c.split(":")[0] for c  in rd.split("\n")]
	df_new.columns = cols 
	st = time.time()
	df_new = objects_encoder(df_new)
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
	output_dir = "../output_census"
	name = "salary"
	create_tree(data,target_col,excluded_strs,output_dir,name,max_depth=4)
	




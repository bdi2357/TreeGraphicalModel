import pandas as pd
import os,re,sys
import numpy as np
import pandas as pd
import featuretools as ft
import time
import random
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn import tree
import random
import argparse
import glob

import time 
start = time.time()
#from feature_analysis_tickers import predict_col,columns_drop
RFC = RandomForestClassifier()
DTC = DecisionTreeClassifier(max_depth =3)
import os,sys,re

from feature_analysis_tickers import predict_col,columns_drop


def create_tree(data,target_col,excluded_strs,output_dir,name):
	AAA = predict_col(data = data,name = name , target_col = target_col,excluded_strs = ["_f"],models = [DTC],output_dir=output_dir)
	tar_file2= os.path.join(output_dir,"%s_tree_model.dot"%name)
	tree.export_graphviz(decision_tree=DTC, out_file=tar_file2,max_depth=3,feature_names=AAA[0],filled = True,rounded=True)
	os.system("dot -Tpng %s -o %s"%(tar_file2,tar_file2.replace("dot","png")))


if __name__ == "__main__":
	"""
	panel2019 = pd.read_csv("../tmp/test_1000_2019.csv",index_col=[0,1])
	word_df_g1 = {k:v for k,v in panel2019.groupby(level =1)}
	word_df_g0 = {k:v for k,v in panel2019.groupby(level =0)}
	#A1 = pd.read_csv("/Volumes/FINZORFLASH/SHARADAR_RAW/SHARADAR_SF1_086134faf658fcc2cdcb53f4295b5fad.zip",compression="zip")
	cvx= word_df_g0["CVX"].copy()
	"""
	all_f = glob.glob("/Volumes/FINZORFLASH/Merged/*.cvv")
	for x in all_f:
		os.system("mv %s %s"%(x,x.replace("cvv","csv")))
	print(len(all_f))
	all_f = glob.glob("/Volumes/FINZORFLASH/Merged/*.csv")
	print(len(all_f))
	
	df1 = [pd.read_csv(x) for x in all_f ]
	df = pd.concat(df1)
	df = df.sample(frac = 0.03, replace = False)
	df_sf1 = pd.read_csv("/Users/itaybd/Documents/SHARADAR_INDICATORS_c41c38a0aaed171169bb790c5b4b459a.zip", compression="zip")
	dict_ind = df_sf1.set_index("indicator")["title"].to_dict()
	df = df.rename(columns = {c : c.replace(c.split("_")[0],dict_ind[c.split("_")[0]]) for c in df.columns if c.split("_")[0] in dict_ind.keys()})
	df = df.fillna(-999)
	print("before ",df.shape)
	df = df.dropna(1)
	print("after ",df.shape)
	target_col = "Return_f60_LT_-20"
	df[target_col] = df["Return_f60"] < -0.2
	
	output_dir = "../tmp"
	excluded_strs = ["_f"]
	name = "all_20NN"
	print("df_shape : ",df.shape)
	create_tree(data = df,target_col = target_col,excluded_strs= excluded_strs,output_dir = output_dir,name=name)

	"""
	
	AAA = predict_col(data = cvx,name = "CVX", target_col = target_col,excluded_strs = ["_f"],models = [RFC,DTC],output_dir=output_dir)
	tar_file2= "../tmp/tmp_tree.dot"
	tree.export_graphviz(decision_tree=DTC, out_file=tar_file2,max_depth=3,feature_names=AAA[0],filled = True,rounded=True)
	os.system("dot -Tpng %s -o %s"%(tar_file2,tar_file2.replace("dot","png")))
	"""






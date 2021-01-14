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
import joblib
from sklearn import tree
import random
import argparse
import glob
import pickle

import time 
start = time.time()
#from feature_analysis_tickers import predict_col,columns_drop
RFC = RandomForestClassifier()
DTC = DecisionTreeClassifier(max_depth =3)
import os,sys,re

def predict_col(data,name,target_col,model,output_dir):
	data = data.dropna(axis=1,how="all")
	X = data
	print("X shape is:",X.shape)
	y = X.pop(target_col) #.astype('int')
	
	cols_out = X.columns
	clf = model
	print(clf.__doc__.split("\n")[0])
	clf.fit(X,y)
	model_name = clf.__doc__.split("\n")[0].replace("A ","").replace(".","").replace(" ","_")
	model_dest = os.path.join(output_dir,name+"_"+target_col+"_"+model_name+".pkl")
	with open(model_dest,'wb') as fp:
		pickle.dump(clf, fp)
	print("model_dest %s"%model_dest)		
	
	#joblib.dump(clf, os.path.join(output_dir,name+"_"+target_col+"_"+model_name+".pkl")) 
	class_names = []
	for xx in list(y):
		if not xx in class_names:
			class_names.append(xx)
	return list(cols_out),class_names



def create_tree(data,target_col,excluded_strs,output_dir,name,max_depth=3):
	cols_out,cn = predict_col(data = data,name = name , target_col = target_col,model = DTC,output_dir=output_dir)
	with open(os.path.join(output_dir,"features.pkl"),"wb") as fp:
		pickle.dump(cols_out,fp)
	with open(os.path.join(output_dir,"target.pkl"),"wb") as fp:
		pickle.dump(cn,fp)
	tar_file= os.path.join(output_dir,"%s_tree_model.dot"%name)
	tree.export_graphviz(decision_tree=DTC, out_file=tar_file,max_depth=max_depth,feature_names=cols_out,class_names=cn,filled = True,rounded=True)
	os.system("dot -Tpng %s -o %s"%(tar_file,tar_file.replace("dot","png")))








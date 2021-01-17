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
#RFC = RandomForestClassifier()
#DTC = DecisionTreeClassifier(max_depth =3)
import os,sys,re

def predict_col(data,name,target_col,model,output_dir,class_names=[],max_depth=3):
	data = data.dropna(axis=1,how="all")
	X = data
	y = X.pop(target_col) 
	
	cols_out = X.columns
	clf = model(max_depth=max_depth)
	
	clf.fit(X,y)
	model_name = clf.__doc__.split("\n")[0].replace("A ","").replace(".","").replace(" ","_")
	model_dest = os.path.join(output_dir,name+"_"+target_col+"_"+model_name+".pkl")
	with open(model_dest,'wb') as fp:
		pickle.dump(clf, fp)
	print("model_dest %s"%model_dest)		
	
	#joblib.dump(clf, os.path.join(output_dir,name+"_"+target_col+"_"+model_name+".pkl")) 
	if class_names == []:
		for xx in list(y):
			if not xx in class_names:
				class_names.append(xx)

	return list(cols_out),class_names,model_dest,clf



def create_tree(data,target_col,excluded_strs,output_dir,name,max_depth=3,class_names=[]):
	cols_out,cn,model_dest,decision_tree = predict_col(data = data,name = name , target_col = target_col,model = DecisionTreeClassifier,output_dir=output_dir,class_names=class_names,max_depth=max_depth)
	with open(os.path.join(output_dir,"features.pkl"),"wb") as fp:
		pickle.dump(cols_out,fp)
	with open(os.path.join(output_dir,"target.pkl"),"wb") as fp:
		pickle.dump(cn,fp)
	with open(model_dest,"wb") as fp:
		pickle.dump(decision_tree,fp)
	tar_file= os.path.join(output_dir,"%s_tree_model.dot"%name)
	tree.export_graphviz(decision_tree=decision_tree, out_file=tar_file,max_depth=max_depth,feature_names=cols_out,class_names=cn,filled = True,rounded=True)
	os.system("dot -Tpng %s -o %s"%(tar_file,tar_file.replace("dot","png")))








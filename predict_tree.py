
import pandas as pd 
import os,sys,re
import joblib
import pydotplus
import joblib
import time
from sklearn import tree
from feature_analysis_tickers import columns_drop

def decsion_path_visualization(clf,dot_data,samples,output_file):
	stl = time.time()
	graph = pydotplus.graph_from_dot_data(dot_data)
	print("after graph %0.2f"%(time.time()-stl))
	# empty all nodes, i.e.set color to white and number of samples to zero
	for node in graph.get_node_list():
	    if node.get_attributes().get('label') is None:
	        continue
	    if 'samples = ' in node.get_attributes()['label']:
	        labels = node.get_attributes()['label'].split('<br/>')
	        for i, label in enumerate(labels):
	            if label.startswith('samples = '):
	                labels[i] = 'samples = 0'
	        node.set('label', '<br/>'.join(labels))
	        node.set_fillcolor('white')
	print("after nodes %0.2f"%(time.time()-stl))
	decision_paths = clf.decision_path(samples)
	print("after decision_paths %0.2f"%(time.time()-stl))
	print(decision_paths[0])
	for decision_path in decision_paths:
	    for n, node_value in enumerate(decision_path.toarray()[0]):
	        if node_value == 0:
	            continue
	        if len(graph.get_node(str(n))) == 0:
	        	break;
	        print(str(n))
	        print("#"*55)
	        print(graph.get_node(str(n)))
	        node = graph.get_node(str(n))[0]            
	        node.set_fillcolor('green')
	        labels = node.get_attributes()['label'].split('<br/>')
	        for i, label in enumerate(labels):
	            if label.startswith('samples = '):
	                labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

	        node.set('label', '<br/>'.join(labels))
	print("after computing tree %0.2f"%(time.time()-stl))
	graph.write_png(output_file)

if __name__ == "__main__":
	st = time.time()
	tree_path = "/Users/itaybd/tmp/all_1_Return_f60_LT_-5_decision_tree_classifier.pkl"
	t2  = joblib.load(tree_path)
	p1 = pd.read_csv("/Volumes/FINZORFLASH/Merged/BAC.csv")
	df_sf1 = pd.read_csv("/Users/itaybd/Documents/SHARADAR_INDICATORS_c41c38a0aaed171169bb790c5b4b459a.zip", compression="zip")
	dict_ind = df_sf1.set_index("indicator")["title"].to_dict()
	p1 = p1.rename(columns = {c : c.replace(c.split("_")[0],dict_ind[c.split("_")[0]]) for c in p1.columns if c.split("_")[0] in dict_ind.keys()})
	p1 = p1.rename(columns = {c : c.replace("&","and") for c in p1.columns})
	p1 = p1.fillna(-999)
	p1 = p1.dropna(1)
	target_col = "Return_f60_LT_-20"
	p1[target_col] = p1["Return_f60"] < -0.2
	print("before columns_drop %0.2f"%(time.time()-st))
	data = columns_drop(p1,["_f"],target_col,True)
	print("after columns_drop %0.2f"%(time.time()-st))
	data = data.dropna(axis=1,how="all")
	data.shape
	y = data.pop(target_col).astype('int')
	
	samples = data.loc[1291:]
	output_file = "tree_predict_test.png"
	dot_data = tree.export_graphviz(t2, out_file=None,feature_names=data.columns ,filled=True, rounded=True,special_characters=True,max_depth=3)
	print("after export_graphviz %0.2f"%(time.time()-st))
	#graph = pydotplus.graph_from_dot_data(dot_data)
	print(t2(max_depth =3).predict_proba(samples))
	decsion_path_visualization(t2,dot_data,samples,output_file)

	print("total %0.2f"%(time.time()-st))

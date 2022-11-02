
import pandas as pd 
import os,sys,re
import joblib
import pydotplus
import joblib
import time
from sklearn import tree
from feature_analysis_tickers import columns_drop
def breakdown_values(s,class_names):
	v = re.findall('[0-9]+',s)
	sm = [int(x) for x in v]
	sm2 =  sum(sm)
	sm = [ str(x[1])+" : "+ str(round(100*(float(x[0])/sm2),1))+"%" for x in zip(sm,class_names)]
	return ",".join(sm)
def decsion_path_visualization(clf,dot_data,samples,class_names,output_file=""):
	def modify_label(labels,i,label):
		if label.startswith('samples = '):
			labels[i] = ''#'samples = {}'.format(int(label.split('=')[1]) + 1)
		elif label.startswith('class = '):
			labels[i] = labels[i]
		elif label.startswith('value = '):
			M = re.findall('value = \[.*\]',labels[i])[0]
			labels[i] = labels[i].replace(M,breakdown_values(labels[i],class_names))
			#labels[i]#.replace(M,'')
		elif len(re.findall('gini = [0-9]+\.[0-9]+',labels[i]))>0:#label.startswith('gini = '):
			M = re.findall('gini = [0-9]+\.[0-9]+',labels[i])[0]
			labels[i] = labels[i].replace(M,'')
		elif len(re.findall('entropy = [0-9]+\.[0-9]+',labels[i]))>0:
			M = re.findall('entropy = [0-9]+\.[0-9]+',labels[i])[0]
			labels[i] = labels[i].replace(M,'')
		



	stl = time.time()
	graph = pydotplus.graph_from_dot_data(dot_data)
	
	# empty all nodes, i.e.set color to white and number of samples to zero
	for node in graph.get_node_list():
		
		if node.get_attributes().get('label') is None:
		    continue
			
		if 'samples = ' in node.get_attributes()['label']:
			node.set_fillcolor('white')
		labels = node.get_attributes()['label'].split('<br/>')
		for i, label in enumerate(labels):
			modify_label(labels,i,label)
		if len(labels)<10:
			MJ = []
			for ii in range(len(labels)):
				if len(labels[ii])<2:
					MJ.append(ii)
			for index in sorted(MJ, reverse=True):
				del labels[index]

		if len(labels)>2:
			node.set('label', '<br/>'.join(labels))
		else:
			out_label = '\n'.join(labels)
			if out_label[-1:]==">" :
				out_label = out_label[:-1]
			node.set('label', out_label)


	decision_paths = clf.decision_path(samples)
	for decision_path in decision_paths:
	    for n, node_value in enumerate(decision_path.toarray()[0]):

	        if node_value == 0:
	            continue
	       
	        
	        node = graph.get_node(str(n))[0]            
	        node.set_fillcolor('green')
	if output_file != "" :
		graph.write_png(output_file)
	return graph




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
	data = columns_drop(p1,["_f"],target_col,True)
	print("after columns_drop %0.2f"%(time.time()-st))
	data = data.dropna(axis=1,how="all")
	data.shape
	y = data.pop(target_col).astype('int')
	
	samples = data.loc[1291:]
	output_file = "tree_predict_test.png"
	dot_data = tree.export_graphviz(t2, out_file=None,feature_names=data.columns ,filled=True, rounded=True,special_characters=True,max_depth=3)
	print("after export_graphviz %0.2f"%(time.time()-st))

	print(t2(max_depth =3).predict_proba(samples))
	decsion_path_visualization(t2,dot_data,samples,output_file)

	print("total %0.2f"%(time.time()-st))

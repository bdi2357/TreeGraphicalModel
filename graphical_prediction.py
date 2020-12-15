import pydotplus
from sklearn.datasets import load_iris
from sklearn import tree
import re
import pickle

def parse_value(val_s,class_names):
    A = re.findall("\[.*\]",val_s)[0]
    vals = re.findall("[0-9]+",A)
    vals = [int(x) for x in vals ]
    vals_s = sum(vals)
    vals_r = [ round(float(x)/vals_s,2)*100. for x in vals]
    S = ""
    for c in class_names:
        S+='%s %0.0f%% '%(c,vals_r[class_names.index(c)])
    S+='\nclass = %s'%class_names[vals.index(max(vals))]
    return S

def generate_prediction_path(clf,graph,sample,output_file_name):
    decision_paths = clf.decision_path(sample)

    for decision_path in decision_paths:
        for n, node_value in enumerate(decision_path.toarray()[0]):
            if node_value == 0:
                continue
            node = graph.get_node(str(n))[0]            
            node.set_fillcolor('green')
            labels = node.get_attributes()['label'].split('<br/>')
            for i, label in enumerate(labels):
                if label.startswith('samples = '):
                    labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

            node.set('label', '<br/>'.join(labels))
    graph.write_png(output_file_name)

def genetate_graph(clf,feature_names,target_names):
    

    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    # empty all nodes, i.e.set color to white and number of samples to zero
    for node in graph.get_node_list():
        if node.get_attributes().get('label') is None:
            continue
        if 'samples = ' in node.get_attributes()['label']:
            print("#"*50)
            labels = node.get_attributes()['label'].split('<br/>')
            excluded = []
            for i, label in enumerate(labels):

                if label.find('gini')>-1:
                    labels[i] = ' '

                if label.startswith('samples = '):
                    labels[i] = ' '
                if label.find("class")>-1:
                    labels[i] = ' '
                if label.startswith('value'):
                    labels[i] = parse_value(labels[i],list(iris.target_names))

            
            MM = '\n'.join(labels)
            if MM[:1] == "<":
                MM = MM[1:]
            node.set('label', MM)
            
            node.set_fillcolor('white')
    return graph

def create_prediction_path(model_path,feature_names,target_names,sample,output_file_name):
    clf = pickle.load(open(model_path, 'rb'))
    graph = genetate_graph(clf,feature_names,target_names)
    generate_prediction_path(clf,graph,sample,output_file_name)    

if __name__ == "__main__" : 

    
    output_file_name = '../tmp_pred_tree4.png'
    model_path = '../iris_tree_model.sav'
    clf = tree.DecisionTreeClassifier(random_state=42,max_depth=3)
    iris = load_iris()

    clf = clf.fit(iris.data, iris.target)
    pickle.dump(clf, open(model_path, 'wb'))
    feature_names = iris.feature_names
    target_names = iris.target_names
    sample = iris.data[129:130]

    
    create_prediction_path(model_path,feature_names,target_names,sample,output_file_name)
    #generate_prediction_path(clf,graph,sample,output_file_name)
"""
samples = iris.data[129:130]
decision_paths = clf.decision_path(samples)

for decision_path in decision_paths:
    for n, node_value in enumerate(decision_path.toarray()[0]):
        if node_value == 0:
            continue
        node = graph.get_node(str(n))[0]            
        node.set_fillcolor('green')
        labels = node.get_attributes()['label'].split('<br/>')
        for i, label in enumerate(labels):
            if label.startswith('samples = '):
                labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)

        node.set('label', '<br/>'.join(labels))

filename = '../tmp_pred_tree.png'
graph.write_png(filename)
"""

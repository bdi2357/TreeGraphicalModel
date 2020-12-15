import numpy as np
import pandas as pd
import featuretools as ft
import time
import random
import glob
import os
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
import random
start = time.time()
if os.path.basename(os.getcwd()) == "FinzorAnalytics":
  output_dir = "/Volumes/FLASH/SP/Models"
  portfolio_path = "../Output_FinzorAnalytics/berkshire-hathaway-inc/berkshire-hathaway-inc_weights_last_labeledNew.csv"
  LLC = list(pd.read_csv(portfolio_path)["Ticker"])  
  #/Volumes/PORTABLE/SP
  RFC = RandomForestClassifier()
  DTC = DecisionTreeClassifier()
def name_trans(x):
    return x.replace("AND","&").replace("9","(").replace("0",")").replace("_"," ")
def reverse_name_trans(x):
    x.replace("&","AND").replace("(","9").replace(")","0").replace(" ","_")

"""
if os.path.basename(os.getcwd()) == "FinzorAnalytics":
  panel_dir = os.path.join("/Volumes/FLASH/SP")
  Ln = glob.glob("/Volumes/FLASH/SP/TestSnP*")
  Lnx = [x for x in Ln if len(re.findall("[0-9]+-[0-9]+-[0-9]+\.csv",x))>0]
  panel_path = max(Lnx)
  start = time.time()
  #RZO = pd.read_hdf("/Volumes/PORTABLE/SP/TestSnP500O.csv")
  try:
    RZO = pd.read_hdf(panel_path,"ok")
  except:
    RZO = pd.read_hdf(panel_path)
  print("AFTER 1")
  RZMM = RZO
  print("AFTER 2")
  RZO = RZO[:,RZMM.axes[1],:]
  cc = RZMM.axes[2]
"""
def ord_x(s):
  for x in list(s):
    if ord(x) >=128:
      return True
  return False
"""
if os.path.basename(os.getcwd()) == "FinzorAnalytics":
  cc = [x for x in cc if not ord_x(x)]
  RZMM = RZMM[:,:,cc]

  print("AFTER 3")


  tempff = pd.read_csv('/Users/itaybd/Finzor_2_26/dev_code/Engine/DATA/EODETFN/../ShortInt/HistoryT/AAPL_SI.csv')
  colsF = list(tempff.columns)
  colsF = [x for x in colsF if not ord_x(x)]

  cols = ['% Insider Ownership',
   '% Institutional Ownership',
   'Days to Cover',
   'Short: Prior Mo',
   'Total Short Interest',
   'Shares: Outstanding']
  colsF = [x for x in colsF if not x in cols]
"""
#[~D_ret[x].index.duplicated(keep='first')]
#RZO[:,,:]
def st_in(s,L):
  for ll in L:
    if s.lower().find(ll.lower())>-1:
      return True
  return False

def columns_drop(data,excluded_s,target_col=None,f_target = False):
  print("pre remove indexes:",data.shape)
  #data = data[~data.index.duplicated(keep='first')]
  data.index = range(data.shape[0])
  print("post remove indexes:",data.shape)
  def all_types(col):
    Lx = list(data[col])
    return  set([type(x) for x in Lx if not (isinstance(x,float) or isinstance(x,int) )]) 

  excluded_cols = [c for c in data.columns if  st_in(c,excluded_s) ]
  excluded_cols+=[c for c in data.columns if c.find("ReportDate")>-1 and c.lower().find("delta")==-1]
  if f_target:
    excluded_cols = [c for c in excluded_cols if c!=target_col]

  data = data.copy()
  data = data.drop(excluded_cols,axis=1)
  
  if "Ticker" in data.columns:
    data = data.drop(["Ticker"],axis=1)
  if "Symbol" in data.columns:
    data = data.drop(["Symbol"],axis=1)
  if "Industry" in data.columns:
    data = data.drop(["Industry"],axis=1)
  if "Sector" in data.columns:
    data = data.drop(["Sector"],axis=1)
  if 'ShortSqueeze.com Short Interest Data' in data.columns:
    data = data.drop(['ShortSqueeze.com Short Interest Data'],axis=1)
  bad_c = [c for c in data.columns if c.find("ShortSqueeze")>-1]
  if len(bad_c)>-1:
    data = data.drop(bad_c,axis=1)
  
  GG = ([ (c,all_types(c)) for c in data.columns if len(all_types(c))>1 or str in all_types(c) ])
  #exit(0)
  data = data.drop([xx[0] for xx in GG],axis=1)
  data = data.drop([c for c in data.columns if c.lower().find("unnamed")>-1],axis=1)
  print("Pre dropna shape is:",data.shape)
  #data = data.dropna(axis=1,how="all")
  """
  Lx = list(data.columns)
  for c in Lx:
    print(c,Lx.index(c))
  """
  return data


def predict_col(data,name,target_col,excluded_strs,models,output_dir):
  results ={}
  
  print("excluded_strs are:",excluded_strs)
  #data.to_csv("check_nans_pre_excluded.csv")
  print("pre excluded shape is :",data.shape)
  data = columns_drop(data,excluded_strs,target_col,True)
  print("post excluded shape is :",data.shape)
  #data.to_csv("check_nans.csv")
  data = data.dropna(axis=1,how="all")
  
  
  X = data
  X.index = pd.to_datetime(X.index)
  print("X shape is:",X.shape)
  y = X.pop(target_col).astype('int')
  X = X.sort_index()
  cols_out = X.columns
  print("X shape after is:",X.shape)
  
  X_train, X_test, y_train, y_test = train_test_split(X, y,shuffle=False)
  
  medianpredict1 = [np.median(y_train) for _ in y_test]

  print("train shape is ",X_train.shape)
  
  
  print(">>>>in predict_col<<<")
  LL = {}
  predictors = {}
  for clf in models:
    print(clf.__doc__.split("\n")[0])
    model_name = clf.__doc__.split("\n")[0].replace("A ","").replace(".","").replace(" ","_")
    
    LL[model_name] = []
    clf.fit(X_train,y_train)
    preds = clf.predict(X_test)
    
    pred2_prob = clf.predict_proba(X_test)
    scores = mean_absolute_error(preds, y_test)
    #print('Mean Abs Error: {:.2f}'.format(scores))
    LL[model_name]= sorted(zip(X.columns,clf.feature_importances_),key = lambda x:x[1],reverse =True)
    #LL[model_name].sort(key = lambda x:x[1],reverse =True)
    """
    for jj in range(10):
      print(LL[model_name][jj])
    """
    print("last proba :",pred2_prob[-1])
    
    results[model_name] = [scores,pred2_prob[-1]]
    joblib.dump(clf, os.path.join(output_dir,name+"_"+target_col+"_"+model_name+".pkl")) 
    LL[model_name] = [x for x in LL[model_name] if x[1]>0.01]
    predictors[model_name] = clf
  print("total time %s"%str(round(time.time()-start,2)))
  #LL = [x for x in LL if x[1]>0.01]
  print("END predict_col")
  print("LLLNNN",len([results,round(mean_absolute_error(medianpredict1, y_test),3),LL,predictors]))
  return cols_out,results,round(mean_absolute_error(medianpredict1, y_test),3),LL,predictors

  def feature_models_ticker(ticker,pnl,colsF,target_col,excluded_cols,output_dir=output_dir):
    #predict_col(data,target_col,excluded_strs,models)
    print("-"*55+"\n%s %s\n"%(ticker,target_col)+"-"*55)
    aapl = pnl.ix[ticker]["2017-01-01":]
    
    for ii in range(8):
      cols = list(aapl.columns)[ii*10:ii*10+10]
      print(aapl[:3][cols])

    print("BUG test 1")
    res1,med1,features,predictors = predict_col(data = aapl,name = ticker, target_col = target_col,excluded_strs = excluded_cols,models = [RFC,DTC],output_dir=output_dir)
    aapl2 = pnl.ix[ticker]

    aapl2 = aapl2["2017-01-01":]
    print("BUG test 2")
    res2,med2,features2,predictors = predict_col(data = aapl2,name = ticker, target_col = target_col,excluded_strs = excluded_cols+colsF,models = [RFC,DTC],output_dir=output_dir)
    """
    r1 = {x :res1[x][0] for x in res1.keys()}
    r2 = {x :res2[x][0] for x in res2.keys()}
    print(r1,"\n",r2)
    print("medians %0.3f,%0.3f"%(med1,med2))
    
    if res2[res1.keys()[0]][1][0] >=0.1:
      print("%s in danger!!!"%ticker)
    """
    return features2


  def features_models_extractions(pnl = RZO,colsF = colsF,LLC =LLC,target_col='LabeledFwdDD30'):
      counter_m={}
      LLC = [x for x in LLC if x in pnl.axes[0]]
      for zxz in LLC[:3]:
        #predict_col(data,target_col,excluded_strs,models)
        print("-"*55+"\n%s\n"%zxz+"-"*55)
        aapl = RZO.ix[zxz]

        res1 = predict_col(data = aapl,name = zxz, target_col = target_col,excluded_strs = excluded_cols,models = [RFC,DTC],output_dir=output_dir)
        aapl2 = RZMM.ix[zxz]
        
        res2 = predict_col(data = aapl2,name = zxz, target_col = target_col,excluded_strs = excluded_cols+colsF,models = [RFC,DTC],output_dir=output_dir)
        r1 = {x :res1[x][0] for x in res1.keys()}
        r2 = {x :res2[x][0] for x in res2.keys()}
        print(r1,"\n",r2)
        for kk in res1.keys():
          if kk in counter_m.keys():
            if r2[kk] < r1[kk]:
              counter_m[kk] +=1.
          else:
            counter_m[kk] = 0.
            if r2[kk] < r1[kk]:
              counter_m[kk] +=1.
          if res2[kk][1][0] >=0.1:
            print("%s in danger!!!"%zxz)



if __name__ == "__main__":

  excluded_cols =['FwdDD30', 'FwdReturn10','Ticker','SnP_Industry','SnP_Sector','unnamed','LabeledFwdReturn10','Record Date','Exchange','New_index',"Market Cap"]
  #col_st += [c for c in aapl.columns if c.find("ReportDate")>-1 and c.lower().find("delta")==-1]
  colsF = [x for x in colsF if not x in excluded_cols]
  #LLC = random.sample(list(RZO.axes[0]),400)
  portfolio_path = "../Output_FinzorAnalytics/berkshire-hathaway-inc/berkshire-hathaway-inc_weights_last_labeledNew.csv"
  LLC = list(pd.read_csv(portfolio_path)["Ticker"])  
  start = time.time()
  Sl = list(set(RZO.axes[0]).intersection(set(RZMM.axes[0])))
  
  
  output_dir = "/Volumes/FLASH/SP/Models"
  #LLC = random.sample(list(DHDF.keys()),5)

  
  """
  for kk in counter_m.keys():
    print("for %s the results are: %0.3f"%(kk,counter_m[kk]/float(len(LLC))))
  """
  target_col='LabeledFwdDD30'
  target_col2 = 'LabeledFwdReturn10'
  excluded_cols2 = excluded_cols + ['LabeledFwdDD30']
  excluded_cols2.remove('LabeledFwdReturn10')
  spy_tickers = list(RZO.axes[0])
  test_list = random.sample(spy_tickers,10)
  print("test_list:",test_list)
  for ticker in test_list:
    print("TICKER %s"%ticker)
    feature_models_ticker(ticker,RZMM,colsF,target_col,excluded_cols)
    feature_models_ticker(ticker,RZMM,colsF,target_col2,excluded_cols2)


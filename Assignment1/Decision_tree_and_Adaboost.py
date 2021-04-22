# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:27:42 2020

@author: Nabil
"""

import numpy as np
import pandas as pd
import random

def entropy(target_col):
    s = pd.Series(target_col)
    p = s.value_counts(normalize=True)
    #print(p)
    sum = 0
    for i in range(len(p)):
        sum = sum + p[i]*np.log2(p[i])
    sum = sum*(-1)
    return sum


def info_gain(target_col,ref_col,dataset):
    tot_entropy = entropy(dataset[target_col])
    vals = dataset[ref_col].unique()
    """vals = set()
    for data in dataset[ref_col]:
        vals.add(data)"""
    cnt = len(vals)
    sum = tot_entropy
    for i in range(cnt):
        r = dataset.where(dataset[ref_col] == vals[i]).dropna()
        sum = sum - ((len(r)/len(dataset))*entropy(r[target_col]))
    return sum

def handling_missed_value_string(col,dataset):
    #c = 0
    missed_index = []
    for i in range(len(dataset)):
        if '?' in dataset[col][i]:
            #c = c + 1
            missed_index.append(i)
    s = pd.Series(dataset[col])
    p = s.value_counts()
    occ = np.argmax(p)
    for i in range(len(missed_index)):
        dataset[col][missed_index[i]] = occ
        
def handling_missed_value_num(col,dataset):
    missed_index = []
    for i in range(len(dataset)):
        if dataset[col][i] == ' ':
            #c = c + 1
            missed_index.append(i)
    sum = 0
    for i in range(len(df)):
        if dataset[col][i] != ' ':
            sum = sum + float(dataset[col][i])
    mean = sum/len(dataset)
    for i in range(len(missed_index)):
        dataset[col][missed_index[i]] = mean
        
def binarize_with_mean(col,d):
    sum = 0
    for i in range(len(d)):
        sum = sum + float(d[col][i])
    mean = sum/(len(d))
    for i in range(len(d)):
        if float(d[col][i]) > mean:
            d[col][i] = '1.0'
        else:
            d[col][i] = '0.0'
    #print(mean)
    
"""
preprocessing for telco_data_set

"""
df = pd.read_csv("F:\\Extra\\telco_dataset.csv")
file = open("F:\\Extra\\telco_dataset.csv")
lines = file.readlines()

dataset = []
dt = []

count = 0

handling_missed_value_num("TotalCharges",df)
for line in lines:
    if count == 0:
        var = line.split(',')
        ft = var[19]
    else:
        var = line.split(',')
        if '.' in var[19]:
            dataset.append(float(var[19]))
            dt.append(float(var[19]))
        else:
            var[19] = var[19]+'.0'
            dataset.append(float(var[19]))
            dt.append(float(var[19]))
            
    count = count + 1

dataset.sort()

mt = []
for i in range(len(dataset)-1):
    mt.append(float((dataset[i]+dataset[i+1])/2))
igl = []
for k in range(len(mt)):
    pp = []
    for i in range(len(dt)):
        if dt[i] >= mt[k]:
            pp.append(1.0)
        else:
            pp.append(0.0)
    for i in range(len(df['TotalCharges'])):
        df['TotalCharges'][i] = float(pp[i])
    ig = info_gain('Churn','TotalCharges',df)
    #print(k,ig)
    igl.append(ig)
    
    m = igl.index(max(igl))
    
    for i in range(len(dataset)):
        if float(df['TotalCharges'][i]) > dataset[m]:
            df['TotalCharges'][i] = '1.0'
        else:
            df['TotalCharges'][i] = '0.0'
            
lines1 = file.readlines()

dataset1 = []
dt1 = []

count = 0

handling_missed_value_num("MonthlyCharges",df)
for line in lines1:
    if count == 0:
        var = line.split(',')
        ft = var[18]
    else:
        var = line.split(',')
        if '.' in var[18]:
            dataset1.append(float(var[18]))
            dt1.append(float(var[18]))
        else:
            var[18] = var[18]+'.0'
            dataset1.append(float(var[18]))
            dt1.append(float(var[18]))
            
    count = count + 1

dataset1.sort()

mt1 = []
for i in range(len(dataset1)-1):
    mt.append(float((dataset1[i]+dataset1[i+1])/2))
igl1 = []
for k in range(len(mt1)):
    pp = []
    for i in range(len(dt1)):
        if dt1[i] >= mt1[k]:
            pp.append(1.0)
        else:
            pp.append(0.0)
    for i in range(len(df['MonthlyCharges'])):
        df['MonthlyCharges'][i] = float(pp[i])
    ig = info_gain('Churn','MonthlyCharges',df)
    #print(k,ig)
    igl1.append(ig)
    
    m1 = igl1.index(max(igl1))
    
    for i in range(len(dataset)):
        if float(df['MonthlyCharges'][i]) > dataset[m1]:
            df['MonthlyCharges'][i] = '1.0'
        else:
            df['MonthlyCharges'][i] = '0.0'

binarize_with_mean('tenure',df)
df.to_csv("F:\\Extra\\telco_dataset_final.csv",index=False)
"""
prepocessing for telco ends

preprocessing for adult
"""

file = open("F:\\Extra\\adult.data")
lines = file.readlines()

dataset = []
dt = []

count = 0

for line in lines:
    var = line.split(',')
    #var[14] = var[14][:-2]
    dataset.append(var)


dataset.pop()

file = open("F:\\Extra\\adult.names")
lines = file.readlines()


dt = []

count = 0

for line in lines:
    var = line.split(':')
    dt.append(var[0])

dff = pd.DataFrame(dataset,columns=dt)

for i in range(len(dff)):
    if '<=50K' in dff['salary'][i]:
        dff['salary'][i] = '<=50K'
    else:
        dff['salary'][i] = '>50K'

handling_missed_value_string('workclass',dff)
handling_missed_value_string('occupation',dff)
handling_missed_value_string('native-country',dff)

binarize_with_mean('age',dff)
binarize_with_mean('fnlwgt',dff)
binarize_with_mean('education-num',dff)
binarize_with_mean('capital-gain',dff)
binarize_with_mean('capital-loss',dff)
binarize_with_mean('hours-per-week',dff)
       
dff.to_csv("F:\\Extra\\adult_final.csv",index=False)


"""
pre processing for credit card
"""
dfc = pd.read_csv("F:\\Extra\\creditcard.csv")
r = dfc.where(dfc['Class'] == '0.0').dropna()
r1 = r.sample(n = 20000)
rr = dfc.where(dfc['Class'] == '1.0').dropna()
dtst = pd.concat([r1,rr])
dt_sorted = dtst.sort_index()

col = []
for data in dt_sorted.columns:
    col.append(data)
del dt_sorted[col[0]]
col.pop(0)
for i in range(len(col)-1):
    binarize_with_mean(data,dt_sorted)

dt_sorted.to_csv("F:\\Extra\\credit_card_final.csv",index=False)

"""
preprocessing ends
"""
df = pd.read_csv("F:\\Extra\\adult_final.csv")

def gini_index(target_col,ref_col,dataset):
    vals = dataset[ref_col].unique()
    val_cnt = []
    val_prob = []
    for i in range(len(vals)):
        r = dataset.where(dataset[ref_col] == vals[i]).dropna()
        val_cnt.append(len(r))
        s = pd.Series(r[target_col])
        p = s.value_counts(normalize=True)
        sum = 1
        for j in range(len(p)):
            sum = sum - (p[j]*p[j])
        val_prob.append(sum)
    gini = 0
    for i in range(len(val_prob)):
        gini = gini + float((val_cnt[i]/len(dataset))*val_prob[i])
    return gini
        

def Decision_tree_algo(dataset,odataset,features,target_att_name,parent_node=None):
    c = dataset[target_att_name].unique()
    if len(c) == 1:
        return c[0]
    elif len(dataset) == 0:
        c1 = odataset[target_att_name].unique()
        cnt = []
        #c = df[target_att_name].unique()
        for i in range(len(c1)):
            cm = 0
            for data in odataset[target_att_name]:
                if data == c1[i]:
                    cm = cm + 1
            cnt.append(cm)
        nn = cnt.index(max(cnt))
        return c1[nn]
    elif len(features) == 0:
        return parent_node
    else:
        #ee = []
        #for data in dataset.columns:
        #    ee.append(data)
        l = []
        #l.append(0)
        ftr=dataset[target_att_name].unique()
        for i in range(len(ftr)):
            c = 0
            for data in dataset[target_att_name]:
                if data == ftr[i]:
                    c = c + 1
            l.append(c)
        p_node = l.index(max(l))
        parent_node = ftr[p_node]
        #print(parent_node)
        
        IG_for_ft = []
        #row,col = dataset.shape
        #fg = pd.DataFrame(features)
        for i in range(len(features)):
            IG_for_ft.append(info_gain(target_att_name,features[i],dataset))
        #IG_for_ft = [info_gain(target_att_name,features[i],dataset) for i in range(len(features))] 
        idx = IG_for_ft.index(max(IG_for_ft)) 
        best_feature = features[idx]
        #print(best_feature)
        #IG_for_ft.pop(idx)
        
        decision_tree = {best_feature:{}}
        #depth = depth + 1
        
        #print(depth)
        #rint(best_feature)
        best_feature_val = dataset[features[idx]].unique()
        features = [i for i in features if i != best_feature] 
        
        for value in best_feature_val:
            #rint(value)
            sub_data = dataset.where(dataset[best_feature] == value).dropna()
            #print(depth)
            subtree = Decision_tree_algo(sub_data,dataset,features,target_att_name,parent_node)
            decision_tree[best_feature][value] = subtree
        #print(depth)
        return (decision_tree)
        
def predict(query,tree,default = 1):
    tk = []
    qk = []
    for key in tree.keys():
        tk.append(key)
    for key in query.keys():
        qk.append(key)
    #print(len(tk))
    #print(len(qk))
    for key in qk:
        for i in range(len(tk)):        
            if key == tk[i]:
                try:
                    result = tree[tk[i]][query[key]]
                except:
                    return default
                result = tree[tk[i]][query[key]]                          
                if type(result) == dict:                 
                    return predict(query,result) 
                else:                 
                    return result

de2 = []
dff = df
for data in df.columns:
   de2.append(data) 
#del dff[de2[0]]
t = int(len(df)*0.8)
print(t)
training_data = df.iloc[:t].reset_index(drop=True)
print(len(training_data))
testing_data = df.iloc[t:].reset_index(drop=True)
print(len(testing_data))
ftr = []
for data in testing_data.columns:
    ftr.append(data)
ftr.pop(0)
#print(ftr)
s = ftr.pop()
dt = Decision_tree_algo(training_data,training_data,ftr,s)
print(dt)


def calculate_accuracy(dataset,tree,target_col):
    #tree = Decision_tree_algo(dataset,dataset,features,tc)
    queries = dataset.iloc[:,:-1].to_dict(orient = "records")
    predictions = []
    result = []
    for i in range(len(queries)):
        predictions.append(predict(queries[i],tree))
    #print(len(predictions))
    for data in dataset[target_col]:
        result.append(data)
    #print(len(result))
    cnt = 0
    for i in range(len(dataset)):
        if predictions[i] == result[i]:
            cnt = cnt + 1
    acc = float(cnt/len(dataset))*100
    #for i in range(le(dataset)):
    return acc

training_accuracy = calculate_accuracy(training_data,dt,s)   
print('Training Accuracy ', training_accuracy,'%')
testing_accuracy = calculate_accuracy(testing_data,dt,s)   
print('Testing Accuracy ', testing_accuracy,'%')  

def confusuion_matrix(dataset,tree,target_col,yes,no):
    queries = dataset.iloc[:,:-1].to_dict(orient = "records")
    predictions = []
    result = []
    for i in range(len(queries)):
        predictions.append(predict(queries[i],tree))
    #print(len(predictions))
    for data in dataset[s]:
        result.append(data)
    #print(len(result))
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(dataset)):
        if predictions[i] == result[i]:
            if result[i] == yes:
                TP = TP + 1
            else:
                TN = TN + 1
        else:
            if result[i] == yes:
                FN = FN + 1
            else:
                FP = FP + 1
    conf_mat = []
    py = []
    pn = []
    py.append(TP)
    py.append(FP)
    pn.append(FN)
    pn.append(TN)
    conf_mat.append(py)
    conf_mat.append(pn)
    return conf_mat

def performance_metrics(dataset,tree,target_col,yes,no):
    cm_train = confusuion_matrix(dataset,tree,target_col,yes,no)
    print(cm_train)
    TP = cm_train[0][0]
    FP = cm_train[0][1]
    FN = cm_train[1][0]
    TN = cm_train[1][1]
    TPR = float(TP/(TP+FN))*100
    print('True Positive Value ', TPR,'%')
    TNR = float(TN/(TN+FP))*100
    print('True Negative Value ', TNR,'%')
    PPV = float(TP/(TP+FP))*100
    print('Positive Predictive Value',PPV,'%')
    FDR = float(FP/(TP+FP))*100
    print('False Discovery Rate',FDR,'%')
    prec = float(TP/(TP+FP))
    rec =  float(TP/(TP+FN))
    f1_score = 2*((prec*rec)/(prec+rec))
    print('F1 Score',f1_score*100,'%')


performance_metrics(training_data,dt,s,'>50K','<=50')
performance_metrics(testing_data,dt,s,'>50K','<=50')


def decision_stump_generation(dataset,target_col,best_feature):
    decision_stump = {best_feature:{}}
    f = dataset[best_feature].unique()
    tc = dataset[target_col].unique()
    for i in range(len(f)):
        cnt = []
        for j in range(len(tc)):
            c = 0
            rg = dataset.where(dataset[best_feature] == f[i]).dropna()
            for data in rg[target_col]:
                if data == tc[j]:
                    c = c + 1
            cnt.append(c)
        g = cnt.index(max(cnt))
        decision_stump[best_feature][f[i]] = tc[g]
    return (decision_stump)

col = []
for data in df.columns:
    col.append(data)
#print(col)
#for i in range(1,len(col)-1):
#    print(decision_stump_generation(df,col[len(col)-1],col[i]))
    
def resampling(dataset,wt):
    dfrand = pd.DataFrame(columns=dataset.columns)
    nw = []
    for i in range(len(training_data)):
        nw.append(wt[i])
    for i in range(1,len(wt)):
        nw[i] = nw[i] + nw[i-1]
    #print(nw)
    for j in range(len(dataset)):
        #value = np.random.uniform(0.0,1.0)
        v = np.random.uniform(0.0,1.0)
        idx = 0
        for i in range(len(nw)):
            if nw[i] > v:
                idx = i
                break
        a_row = training_data.iloc[idx]
        row_df = pd.DataFrame([a_row])
        dfrand = pd.concat([row_df, dfrand])
    return dfrand
"""
def resampling_for_larger_dataset(dataset,wt):
    idx = []
    for data in dataset.index.values:
        idx.append(data)
    selects = random.choices(idx,weights=wt,k=len(dataset))
    data = dataset.loc[(selects)]
    return data
"""
def adaboost(dataset,K,features,target_col,w):
    queries = dataset.iloc[:,:-1].to_dict(orient = "records")
    stumps = []
    Z = []
    for k in range(K):
        dtst = resampling(dataset,w)
       #dtst = resampling_for_larger_dataset(dataset,w)
        #print(dtst)
        ig = []
        for data in features:
            ig.append(info_gain(target_col,data,dtst))
        a = ig.index(max(ig))
        bst_ftr = features[a]
        d = decision_stump_generation(dtst,target_col,bst_ftr)
        #print(d)
        stumps.append(d)
        error = 0.0
        predictions = []
        result = []
        #print(w)
        for j in range(len(dataset)):
            y = predict(queries[j],stumps[k])
            predictions.append(y)
            result.append(dataset[target_col][j])
            #print(predictions[j],result[j])
        for j in range(len(dataset)):
            if predictions[j] != result[j]:
                error = error + w[j]
        #print(error)
        if error > 0.5:
            #print('bok')
            continue
        for j in range(len(dataset)):
            if predictions[j] == result[j]:
                w[j] = w[j]*(error/(1-error))
        #norm = np.linalg.norm(weight)
        sum = 0
        for i in range(len(w)):
            sum = sum + w[i]
        for i in range(len(w)):
            w[i] = float(w[i]/sum)
        r = 0.5*np.log((1-error)/error)
        #print(r)
        Z.append(r)       
    return stumps,Z

weight = []
for i in range(len(training_data)):
    weight.append(float(1/len(training_data)))
    
def adaboost_acc(dataset,stumps,Z,target_col):
    queries = dataset.iloc[:,:-1].to_dict(orient = "records")
    rr = []
    for i in range(len(dataset)):
        res = []
        for j in range(len(Z)):
            y = predict(queries[i],stumps[j],1)
            res.append(y)
        tt = np.unique(res)
        cnt = []
        for i in range(len(tt)):
            sum = 0
            for j in range(len(Z)):
                if tt[i] == res[j]:
                    sum = sum + Z[j]
            cnt.append(sum)
        rr.append(tt[cnt.index(max(cnt))])
    acc = 0
    for i in range(len(dataset)):
        if rr[i] == dataset[target_col][i]:
            acc = acc + 1
    accu = float(acc/(len(dataset)))*100
    return accu


for i in range(1,5):
    qq = adaboost(training_data,5*i,ftr,s,weight)
    st = qq[0]
    zz = qq[1]
    print(st)
    print(zz)
    acr = adaboost_acc(training_data,st,zz,s)  
    print(acr)
    acrr = adaboost_acc(testing_data,st,zz,s)  
    print(acrr)
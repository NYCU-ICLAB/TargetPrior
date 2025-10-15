# conda activate ml
import pandas as pd
import numpy as np
import scipy.stats as st 
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statannotations.Annotator import Annotator
from itertools import combinations
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import *
from statistics import *
from sklearn.model_selection import train_test_split 
# color map
# https://seaborn.pydata.org/tutorial/color_palettes.html

# read data
df_ip = pd.read_excel('TARGET-AML_NormalandAML.xlsx', header=0)

# case 1 normal_aml box =======================================
df_tmp = df_ip

df_miRNA = df_tmp.iloc[:, 7:]
target = df_miRNA.astype(float)

# read features --------
data = df_miRNA
label = df_tmp.iloc[:, -1]
input = target
# set label --------
pos_mask = label >= 1
pos = data[pos_mask]
neg_mask = label == 0
neg= data[neg_mask]

# case normal_aml auc =======================================
df_ip_2 = df_ip
df_ip_2['Label'] = 0
df_ip_2.loc[(df_ip_2['label']==0), 'label'] = 0
df_ip_2.loc[(df_ip_2['label'] >=1), 'label'] = 1

df_tmp = df_ip_2
df_miRNA = df_tmp.iloc[:, 7:]
df_clinical = df_tmp.iloc[:, :7]
target = df_miRNA.astype(float)

# read features --------
input_X = data.iloc[:,:-2]
input_Y = df_tmp.iloc[:, -1]


kf = KFold(n_splits=5)

# Stratified to get KFold dataset --------
skf = StratifiedKFold(n_splits=5)
cv_auc_final = []
for ii in range(len(input_X.T)):
    sig_input_X = input_X.iloc[:,ii]    
    cv_auc_tmp = []

    for i, (train, test) in enumerate(skf.split(sig_input_X, input_Y)):
        
        print(f"Fold {i}:")
        train_X = sig_input_X.iloc[train].values.astype(np.float64)
        train_Y = input_Y.iloc[train].values.astype(np.int_)
        test_X = sig_input_X.iloc[test].values.astype(np.float64)
        test_Y = input_Y.iloc[test].values.astype(np.int_)
        train_X = train_X.reshape(-1, 1)
        test_X = test_X.reshape(-1, 1)

        estimator = SVC(class_weight = "balanced",)#max_iter=10000
        estimator.fit(train_X, train_Y)
        pred = estimator.predict(test_X)
        decision = estimator.decision_function(test_X)
        auc = roc_auc_score(test_Y, decision)
        print("%-20s:  %5.5f" % ("Auc", auc))
        cv_auc_tmp.append(auc)
        fpr, tpr, _ = roc_curve(test_Y, decision)
        plt.plot(fpr,tpr,label="AUC="+str(round(auc,2)))
        
    cv_auc = np.append(np.float64(cv_auc_tmp),np.float64(mean(cv_auc_tmp)))
    cv_auc_final = np.append(cv_auc_final,cv_auc,axis=0)    

    plt.title("ROC curve")
    plt.xlabel("False positive rate ")
    plt.ylabel("True positive rate ")
    plt.legend(loc=4)
    plt.savefig(target.columns[ii]+"ROC.jpg")
    plt.close()    

cv_auc_op = cv_auc_final.reshape(-1, 6)


# Stratified to get ind dataset --------    
ind_auc = []
for ii in range(len(input_X.T)):
    sig_input_X = input_X.iloc[:,ii]  
    train_X,test_X,train_Y,test_Y = train_test_split(sig_input_X, input_Y,test_size=0.3,stratify=y)
    train_X = train_X.values.astype(np.float64)
    train_Y = train_Y.values.astype(np.int_)
    test_X = test_X.values.astype(np.float64)
    test_Y = test_Y.values.astype(np.int_)
    train_X = train_X.reshape(-1, 1)
    test_X = test_X.reshape(-1, 1)
    
    estimator = SVC(class_weight = "balanced",)#max_iter=10000
    estimator.fit(train_X, train_Y)
    pred = estimator.predict(test_X)
    decision = estimator.decision_function(test_X)
    auc = roc_auc_score(test_Y, decision)
    print("%-20s:  %5.5f" % ("Auc", auc))
    ind_auc.append(auc)
    fpr, tpr, _ = roc_curve(test_Y, decision)
    plt.plot(fpr,tpr,label="AUC="+str(round(auc,2)))
    plt.title("ROC curve")
    plt.xlabel("False positive rate ")
    plt.ylabel("True positive rate ")
    plt.legend(loc=4)
    plt.savefig(target.columns[ii]+"ind_ROC.jpg")
    plt.close()   
    
# write the auc result ---------------------
results_final = pd.DataFrame({
    'Feature': input_X.columns.values,
    'CV1_auc': cv_auc_op[:,0],
    'CV2_auc': cv_auc_op[:,1],
    'CV3_auc': cv_auc_op[:,2],
    'CV4_auc': cv_auc_op[:,3],
    'CV5_auc': cv_auc_op[:,4],
    'CVmean_auc': cv_auc_op[:,5],
    'ind_auc': ind_auc,
})

results_final.to_excel(index=False,header=True)

# Stratified to get ind dataset (30run) --------    
input_X = data.iloc[:,:-1]
input_Y = df_tmp.iloc[:, -1]

from sklearn.model_selection import train_test_split  

ind_auc = []
ind_auc_final = []
for rr in range(30):
    for ii in range(len(input_X.T)):
        sig_input_X = input_X.iloc[:,ii]  
        train_X,test_X,train_Y,test_Y = train_test_split(sig_input_X, input_Y,test_size=0.3,stratify=input_Y)

        train_X = train_X.values.astype(np.float64)
        train_Y = train_Y.values.astype(np.int_)
        test_X = test_X.values.astype(np.float64)
        test_Y = test_Y.values.astype(np.int_)
        train_X = train_X.reshape(-1, 1)
        test_X = test_X.reshape(-1, 1)
        
        estimator = SVC(class_weight = "balanced",)#max_iter=10000
        estimator.fit(train_X, train_Y)
        pred = estimator.predict(test_X)
        decision = estimator.decision_function(test_X)
        auc = roc_auc_score(test_Y, decision)
        print("%-20s:  %5.5f" % ("Auc", auc))
        ind_auc.append(auc)
        fpr, tpr, _ = roc_curve(test_Y, decision)
ind_auc_tmp = np.asarray(ind_auc)
ind_auc_op = ind_auc_tmp.reshape(-1, 30)
ind_auc_op.shape 
ind_auc_op[0,:].shape 
ind_auc_op[:,0].shape 

# Calculate Confidence Intervals---------------------
ci1 = []
ci2 = []
mean_op = []
ciop = []
for ii in range(len(input_X.T)):
    ci_ip = ind_auc_op[ii,:]
    ci_1,ci_2 = t.interval(confidence=0.95, df=len(ci_ip)-1,loc=np.mean(ci_ip),scale=st.sem(ci_ip)) 
    mean_op.append(np.mean(ci_ip))
    ci1.append(ci_1)
    ci2.append(ci_2)
    ciop.append(np.mean(ci_ip)-ci_1)



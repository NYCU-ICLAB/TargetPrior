#!/usr/bin/env python

import argparse
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *
from sklearn.metrics import *
from openpyxl import *
# conda upgrade -c conda-forge scikit-learn
# conda install -c anaconda xlrdy
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression,  Lasso
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
# https://xgboost.readthedocs.io/en/stable/parameter.html
# conda install -c conda-forge scikit-survival
from sklearn.linear_model import  Lasso
from sklearn.metrics import matthews_corrcoef

def read_data(data):
    df = pd.read_csv(data, header=0)
    feature = np.array(df.iloc[:, 0:-1])
    label = np.array(df.iloc[:, -1])
    
    return feature, label, df


def binary_test(estimator, train_X, train_Y, test_X, test_Y,method_type):
    print(method_type)
    estimator.fit(train_X, train_Y)
    pred = estimator.predict(test_X)
    
    tr_pred = estimator.predict(train_X)

    if method_type == "GaussianNB":
        decision = estimator.predict_proba(test_X)[:,1]
        tr_decision = estimator.predict_proba(train_X)[:,1]
    elif  method_type == "MultinomialNB":
        decision = estimator.predict_proba(test_X)[:,1]
        tr_decision = estimator.predict_proba(train_X)[:,1]
    elif  method_type == "KNeighborsClassifier":
        decision = estimator.predict_proba(test_X)[:,1]
        tr_decision = estimator.predict_proba(train_X)[:,1]
    elif  method_type == "MLPClassifier":
        decision = estimator.predict_proba(test_X)[:,1]
        tr_decision = estimator.predict_proba(train_X)[:,1]
    elif  method_type == "DecisionTreeClassifier":
        decision = estimator.predict_proba(test_X)[:,1]
        tr_decision = estimator.predict_proba(train_X)[:,1] 
    elif  method_type == "RandomForestClassifier":
        decision = estimator.predict_proba(test_X)[:,1]
        tr_decision = estimator.predict_proba(train_X)[:,1]
    elif  method_type == "XGBClassifier":
        decision =  (pred >= 0.5)*1
        tr_decision = (tr_pred >= 0.5)*1
    else:

        decision = estimator.decision_function(test_X)
        tr_decision = estimator.decision_function(train_X)

    auc = roc_auc_score(test_Y, decision)
    print("%-20s:  %5.5f" % ("Auc", auc))

    print("%-20s: " % ("Confusion matrix"))
    print(confusion_matrix(test_Y, pred))

    fpr, tpr, _ = roc_curve(test_Y, decision)
    plt.plot(fpr,tpr,label="AUC="+str(round(auc,2)))
    plt.title("ROC curve")
    plt.xlabel("False positive rate ")
    plt.ylabel("True positive rate ")
    plt.legend(loc=4)
    plt.savefig("./" + method_type + "_test_AUC.jpg")
    plt.close()

    df_roc = pd.DataFrame()
    df_roc['fpr'] = fpr
    df_roc['tpr'] = tpr
    writer = pd.ExcelWriter("./" + method_type + "_roc.xlsx")
    df_roc.to_excel(writer,method_type,index=False)
    # writer.save()
    writer.close()


    scores = cross_validate(estimator, train_X, train_Y, cv=5,scoring=('accuracy','recall','precision','f1','roc_auc'),\
        return_train_score=True)
    
    cv_accuracy = np.mean(scores['test_accuracy'])
    cv_recall = np.mean(scores['test_recall'])
    cv_precision = np.mean(scores['test_precision'])
    cv_f1 = np.mean(scores['test_f1'])
    cv_auc = np.mean(scores['test_roc_auc'])    


    # calculate performance
    te_CM = confusion_matrix(test_Y, pred)
    te_TN = te_CM[0][0]
    te_FN = te_CM[1][0] 
    te_TP = te_CM[1][1]
    te_FP = te_CM[0][1]
    te_Population = te_TN+te_FN+te_TP+te_FP

    #tr_pred = estimator.predict(train_X)
    #tr_decision = estimator.decision_function(train_X)

    tr_CM = confusion_matrix(train_Y, tr_pred)
    tr_TN = tr_CM[0][0]
    tr_FN = tr_CM[1][0] 
    tr_TP = tr_CM[1][1]
    tr_FP = tr_CM[0][1]
    tr_Population = tr_TN+tr_FN+tr_TP+tr_FP

    perform = {"title":["Prevalence", "Accuracy", "Sensitivity", "Specificity", "Precision","F1 Score","MCC","AUC"],\
        "Training": [round( (tr_TP+tr_FN) / tr_Population*100,2), round( (tr_TP+tr_TN) / tr_Population*100,2), round( tr_TP / (tr_TP+tr_FN)*100,2 ),round( tr_TN / (tr_TN+tr_FP)*100,2 ),\
            round( tr_TP / (tr_TP+tr_FP)*100,2 ), round(f1_score(train_Y, tr_pred),3),round(matthews_corrcoef(train_Y, tr_pred),3), round(roc_auc_score(train_Y, tr_decision),3) ],\
        "CV": [round( (tr_TP+tr_FN) / tr_Population*100,2), round( cv_accuracy*100,2 ), round( cv_recall*100,2 ),'NA',\
            round( cv_precision*100,2 ), round(cv_f1,3),"NA", round(cv_auc,3) ],\
        "Test": [round( (te_TP+te_FN) / te_Population*100,2), round( (te_TP+te_TN) / te_Population*100,2), round( te_TP / (te_TP+te_FN)*100,2 ),round( te_TN / (te_TN+te_FP)*100,2 ),\
            round( te_TP / (te_TP+te_FP)*100,2 ), round(f1_score(test_Y, pred),3),round(matthews_corrcoef(test_Y, pred),3), round(roc_auc_score(test_Y, decision),3) ]}
       
    # write performance output
    df = pd.DataFrame(perform)
    df.to_excel(index=False)

    # training & test ROC
    tr_fpr, tr_tpr, _ = roc_curve(train_Y, tr_decision)
    plt.plot(tr_fpr,tr_tpr,label="Training AUC ="+str(round(roc_auc_score(train_Y, tr_decision),3)))
    plt.plot(fpr,tpr,label="Test AUC ="+str(round(auc,3)))
    
    plt.title("ROC curve")
    plt.xlabel("False positive rate ")
    plt.ylabel("True positive rate ")
    plt.legend(loc=4)
    plt.savefig("./" + method_type + "_training-test_AUC.jpg")
    plt.close()
    return fpr, tpr, auc, decision

import glob

if __name__ == '__main__':
    
    data_tr = glob.glob(r"train*")[0]
    data_te = glob.glob(r"ind_*")[0]
    feature_n = 18

    train_X, train_Y, train_df = read_data(data_tr)
    test_X, test_Y, test_df = read_data(data_te)


    out_method_df = pd.read_csv('./roc_ouput.csv', header=None)
    out_method_df.columns = ['our_fpr', 'our_tpr']
    out_method_aucip = pd.read_csv('./auc_ouput.csv', header=None)
    out_method_aucip.columns = ['our_testy', 'our_dv']
    our_auc = roc_auc_score(out_method_aucip['our_testy'], out_method_aucip['our_dv'])

    # performance file
    df_dv = pd.DataFrame()
    df_dv['test_Y'] = test_Y
    df_auc = pd.DataFrame()

    df = pd.DataFrame()
    df.to_excel(index=False)


    # total =======================================
    estimator = RandomForestClassifier(n_estimators=1000)
    method_type = "RandomForestClassifier"
    RandomForestClassifier_fpr, RandomForestClassifier_tpr, RandomForestClassifier_auc, RandomForestClassifier_dv = binary_test(estimator, train_X, train_Y, test_X, test_Y,method_type)
    df_dv['T_RF'] = RandomForestClassifier_dv
    df_auc['T_RF'] = RandomForestClassifier_auc

    estimator = AdaBoostClassifier(n_estimators=1000)
    method_type = "AdaBoostClassifier"
    AdaBoostClassifier_fpr, AdaBoostClassifier_tpr, AdaBoostClassifier_auc, AdaBoostClassifier_dv = binary_test(estimator, train_X, train_Y, test_X, test_Y,method_type)
    df_dv['T_AdaBoost'] = AdaBoostClassifier_dv
    df_auc['T_AdaBoost'] = AdaBoostClassifier_auc

    estimator = GradientBoostingClassifier(n_estimators=1000)
    method_type = "GradientBoostingClassifier"
    GradientBoostingClassifier_fpr, GradientBoostingClassifier_tpr, GradientBoostingClassifier_auc, GradientBoostingClassifier_dv = binary_test(estimator, train_X, train_Y, test_X, test_Y,method_type)
    df_dv['T_GradientBoosting'] = GradientBoostingClassifier_dv
    df_auc['T_GradientBoosting'] = GradientBoostingClassifier_auc

    estimator = xgb.XGBClassifier(objective="binary:logistic",n_estimators=1000)
    method_type = "XGBClassifier"
    XGBClassifier_fpr, XGBClassifier_tpr, XGBClassifier_auc, XGBClassifier_dv = binary_test(estimator, train_X, train_Y, test_X, test_Y,method_type)
    df_dv['T_XGB'] = XGBClassifier_dv
    df_auc['T_XGB'] = XGBClassifier_auc

    plt.plot(out_method_df['our_fpr'],out_method_df['our_tpr'],label= "EL-CAML AUC="+str(round(our_auc,3)))
    plt.plot(RandomForestClassifier_fpr,RandomForestClassifier_tpr,label= "RandomForest AUC="+str(round(RandomForestClassifier_auc,3)))
    plt.plot(AdaBoostClassifier_fpr,AdaBoostClassifier_tpr,label= "AdaBoost AUC="+str(round(AdaBoostClassifier_auc,3)))
    plt.plot(GradientBoostingClassifier_fpr,GradientBoostingClassifier_tpr,label= "GradientBoosting AUC="+str(round(GradientBoostingClassifier_auc,3)))
    plt.plot(XGBClassifier_fpr,XGBClassifier_tpr,label= "XGBoost AUC="+str(round(XGBClassifier_auc,3)))

    plt.title("ROC curve")
    plt.xlabel("False positive rate ")
    plt.ylabel("True positive rate ")
    plt.savefig("./total_test_AUC_raw.jpg")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='lower left')
    plt.tight_layout()
    plt.savefig("./total_test_AUC.jpg")
    plt.close()




    # feature selection - Lasso ===================================================
    model_lasso = Lasso(alpha=0.01)
    model_lasso.fit(train_X, train_Y)

    weigh_lasso = abs(model_lasso.coef_)

    selectf_lasso = np.argpartition(weigh_lasso, feature_n*-1)[feature_n*-1:] # index

    train_X_lasso = np.array(train_df.iloc[:,selectf_lasso])
    train_Y_lasso = train_Y
    test_X_lasso = np.array(test_df.iloc[:,selectf_lasso])
    test_Y_lasso = test_Y

    estimator = GaussianNB()
    method_type = "GaussianNB"
    GaussianNB_fpr, GaussianNB_tpr, GaussianNB_auc, GaussianNB_dv = binary_test(estimator, train_X_lasso, train_Y_lasso, test_X_lasso, test_Y_lasso,method_type)
    df_dv['p_GaussianNB'] = GaussianNB_dv
    df_auc['p_GaussianNB'] = GaussianNB_auc

    estimator = LogisticRegression()
    method_type = "LogisticRegression"
    LogisticRegression_fpr, LogisticRegression_tpr, LogisticRegression_auc, LogisticRegression_dv = binary_test(estimator, train_X_lasso, train_Y_lasso, test_X_lasso, test_Y_lasso,method_type)
    df_dv['p_LR'] = LogisticRegression_dv
    df_auc['p_LR'] = LogisticRegression_auc

    estimator = KNeighborsClassifier()
    method_type = "KNeighborsClassifier"
    KNeighborsClassifier_fpr, KNeighborsClassifier_tpr, KNeighborsClassifier_auc, KNeighborsClassifier_dv = binary_test(estimator, train_X_lasso, train_Y_lasso, test_X_lasso, test_Y_lasso,method_type)
    df_dv['p_KNN'] = KNeighborsClassifier_dv
    df_auc['p_KNN'] = KNeighborsClassifier_auc

    estimator = DecisionTreeClassifier()
    method_type = "DecisionTreeClassifier"
    DecisionTreeClassifier_fpr, DecisionTreeClassifier_tpr, DecisionTreeClassifier_auc, DecisionTreeClassifier_dv = binary_test(estimator, train_X_lasso, train_Y_lasso, test_X_lasso, test_Y_lasso,method_type)
    df_dv['p_DT'] = DecisionTreeClassifier_dv
    df_auc['p_DT'] = DecisionTreeClassifier_auc


    plt.plot(out_method_df['our_fpr'],out_method_df['our_tpr'],label= "EL-CAML AUC="+str(round(our_auc,3)))
    plt.plot(GaussianNB_fpr,GaussianNB_tpr,label= "GaussianNB AUC="+str(round(GaussianNB_auc,3)))
    plt.plot(LogisticRegression_fpr,LogisticRegression_tpr,label= "LogisticRegression AUC="+str(round(LogisticRegression_auc,3)))
    plt.plot(KNeighborsClassifier_fpr,KNeighborsClassifier_tpr,label= "KNeighbors AUC="+str(round(KNeighborsClassifier_auc,3)))
    plt.plot(DecisionTreeClassifier_fpr,DecisionTreeClassifier_tpr,label= "DecisionTree AUC="+str(round(DecisionTreeClassifier_auc,3)))


    plt.title("ROC curve")
    plt.xlabel("False positive rate ")
    plt.ylabel("True positive rate ")
    plt.savefig("./lassoselect_test_AUC_raw.jpg")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='lower left')
    plt.tight_layout()
    plt.savefig("./lassoselect_test_AUC.jpg")
    plt.close()



    # feature selection - pvalue ===================================================
    df = pd.DataFrame()
    writer = pd.ExcelWriter("./total_performance.xlsx")
    df.to_excel(writer,'blank',index=False)
    writer.close()

    select_class = SelectKBest(f_classif, k=feature_n)
    train_X = select_class.fit_transform(train_X, train_Y)

    estimator = GaussianNB()
    method_type = "GaussianNB"
    GaussianNB_fpr, GaussianNB_tpr, GaussianNB_auc, GaussianNB_dv = binary_test(estimator, train_X, train_Y, test_X, test_Y,method_type)
    df_dv['p_GaussianNB'] = GaussianNB_dv
    df_auc['p_GaussianNB'] = GaussianNB_auc

    estimator = LogisticRegression()
    method_type = "LogisticRegression"
    LogisticRegression_fpr, LogisticRegression_tpr, LogisticRegression_auc, LogisticRegression_dv = binary_test(estimator, train_X, train_Y, test_X, test_Y,method_type)
    df_dv['p_LR'] = LogisticRegression_dv
    df_auc['p_LR'] = LogisticRegression_auc

    estimator = KNeighborsClassifier()
    method_type = "KNeighborsClassifier"
    KNeighborsClassifier_fpr, KNeighborsClassifier_tpr, KNeighborsClassifier_auc, KNeighborsClassifier_dv = binary_test(estimator, train_X, train_Y, test_X, test_Y,method_type)
    df_dv['p_KNN'] = KNeighborsClassifier_dv
    df_auc['p_KNN'] = KNeighborsClassifier_auc


    estimator = DecisionTreeClassifier()
    method_type = "DecisionTreeClassifier"
    DecisionTreeClassifier_fpr, DecisionTreeClassifier_tpr, DecisionTreeClassifier_auc, DecisionTreeClassifier_dv = binary_test(estimator, train_X, train_Y, test_X, test_Y,method_type)
    df_dv['p_DT'] = DecisionTreeClassifier_dv
    df_auc['p_DT'] = DecisionTreeClassifier_auc


    plt.plot(out_method_df['our_fpr'],out_method_df['our_tpr'],label= "EL-CAML AUC="+str(round(our_auc,3)))
    plt.plot(GaussianNB_fpr,GaussianNB_tpr,label= "GaussianNB AUC="+str(round(GaussianNB_auc,3)))
    plt.plot(LogisticRegression_fpr,LogisticRegression_tpr,label= "LogisticRegression AUC="+str(round(LogisticRegression_auc,3)))
    plt.plot(KNeighborsClassifier_fpr,KNeighborsClassifier_tpr,label= "KNeighbors AUC="+str(round(KNeighborsClassifier_auc,3)))
    plt.plot(DecisionTreeClassifier_fpr,DecisionTreeClassifier_tpr,label= "DecisionTree AUC="+str(round(DecisionTreeClassifier_auc,3)))


    plt.title("ROC curve")
    plt.xlabel("False positive rate ")
    plt.ylabel("True positive rate ")
    plt.savefig("./pvalueselect_test_AUC_raw.jpg")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='lower left')
    plt.tight_layout()
    plt.savefig("./pvalueselect_test_AUC.jpg")
    plt.close()

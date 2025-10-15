
import argparse
from sksurv.nonparametric import kaplan_meier_estimator, nelson_aalen_estimator #coxPH
from lifelines.statistics import logrank_test #coxPH
from lifelines import KaplanMeierFitter #coxPH
from lifelines import NelsonAalenFitter
from sklearn.svm import SVC,SVR #SVC&SVR
from sklearn.metrics import PrecisionRecallDisplay #Precision-Recall curve
from sksurv.linear_model import CoxPHSurvivalAnalysis

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# data = 'train_pre_cox.csv'
data = 'ind_pre_total_cox.csv'
# data = 'ind_pre_large_cox.csv'
# data = 'ind_pre_small_cox.csv'

ext = data.split(".")[-1]
name = data.split(".")[0]
df = pd.read_csv(data, header=0)

dv_label = np.array(df["Decision Value"])
label = np.array(df["Status"])
time_label = np.array(df["Survival"])

y = np.column_stack((label, time_label)).ravel()
y = y.view([('Status', np.int_), ('Survival', np.float64)])


label = y.astype([('Status', np.bool_),
                ('Survival', np.float64)])

pred = dv_label
test_Y = label

## 2 lines - KM ==============================
# (pred > 0) model predict positive -------------
mask_high = pred > 0
time_high, survival_prob_high = kaplan_meier_estimator(test_Y["Status"][mask_high], time_label[mask_high])
plt.step(time_high, survival_prob_high, where="post",label='Relapse')

mask_low = pred < 0
time_low, survival_prob_low = kaplan_meier_estimator(test_Y["Status"][mask_low], time_label[mask_low])
plt.step(time_low, survival_prob_low, where="post",label='Non-relapse')
result_test = logrank_test(time_label[mask_high],time_label[mask_low],event_observed_A=test_Y["Status"][mask_high],event_observed_B=test_Y["Status"][mask_low],)
plt.title("p-value = %1.3e" % result_test.p_value)
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time (Months) ")
plt.legend(frameon=False)
plt.savefig(name+"KaplanMeier-plot.jpg")
plt.close()



## total 2 lines - NAF ==============================
# method 1 -------------
naf_time_high, naf_cum_hazard_high = nelson_aalen_estimator(test_Y["Status"][mask_high], time_label[mask_high])
plt.step(naf_time_high, naf_cum_hazard_high, where="post",label='Relapse')

naf_time_low, naf_cum_hazard_low = nelson_aalen_estimator(test_Y["Status"][mask_low], time_label[mask_low])
plt.step(naf_time_low, naf_cum_hazard_low, where="post",label='Non-relapse')
plt.ylabel("Cummulative hazard rate H(t|x)")
plt.xlabel("time (Months) ")
plt.legend(frameon=False)
plt.savefig(name+"cumulative_hazard-plot.jpg")
plt.close()


## Q1-Q4: 4 lines of KM ==============================
pred_max = max(pred)-0
pred_pos_1 = pred_max*0.75
pred_pos_2 = pred_max*0.5
pred_pos_3 = pred_max*0.25

pred_min = 0-min(pred) 
pred_neg_1 = pred_min*-0.25
pred_neg_2 = pred_min*-0.5
pred_neg_3 = pred_min*-0.75


mask_1 = pred > pred_pos_2
time_pos_1, survival_prob_pos_1 = kaplan_meier_estimator(test_Y["Status"][mask_1], time_label[mask_1])
plt.step(time_pos_1, survival_prob_pos_1, where="post",label='DV 0.75-1.00')


mask_2 = np.logical_and(pred >0,pred <= pred_pos_2)
time_pos_2, survival_prob_pos_2 = kaplan_meier_estimator(test_Y["Status"][mask_2],time_label[mask_2])
plt.step(time_pos_2, survival_prob_pos_2, where="post",label='DV 0.5-0.75')

mask_3 = np.logical_and(pred >= pred_neg_2, pred < 0)
time_neg_1, survival_prob_neg_1 = kaplan_meier_estimator(test_Y["Status"][mask_3], time_label[mask_3])
plt.step(time_neg_1, survival_prob_neg_1, where="post",label='DV 0.25-0.5')

mask_4 = pred < pred_neg_2
time_neg_2, survival_prob_neg_2 = kaplan_meier_estimator(test_Y["Status"][mask_4], time_label[mask_4])
plt.step(time_neg_2, survival_prob_neg_2, where="post",label='DV 0-0.25')

plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$ ")
plt.legend(frameon=False)
plt.savefig(name+"KaplanMeier-plotQuartiles.jpg")
plt.close()


## plot 4 lines ==============================
naf_time_pos_1, naf_survival_prob_pos_1 = nelson_aalen_estimator(test_Y["Status"][mask_1], time_label[mask_1])
plt.step(naf_time_pos_1, naf_survival_prob_pos_1, where="post",label='DV 0.75-1.00')

naf_time_pos_2, naf_survival_prob_pos_2 = nelson_aalen_estimator(test_Y["Status"][mask_2],time_label[mask_2])
plt.step(naf_time_pos_2, naf_survival_prob_pos_2, where="post",label='DV 0.5-0.75')

naf_time_neg_1, naf_survival_prob_neg_1 = nelson_aalen_estimator(test_Y["Status"][mask_3], time_label[mask_3])
plt.step(naf_time_neg_1, naf_survival_prob_neg_1, where="post",label='DV 0.25-0.5')

naf_time_neg_2, naf_survival_prob_neg_2 = nelson_aalen_estimator(test_Y["Status"][mask_4], time_label[mask_4])
plt.step(naf_time_neg_2, naf_survival_prob_neg_2, where="post",label='DV 0-0.25')

plt.ylabel("Cummulative hazard rate H(t|x)")
plt.xlabel("time $t$ ")
plt.legend(frameon=False)
plt.savefig(name+"cumulative_hazard-plotQuartiles.jpg")
plt.close()

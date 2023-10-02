# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 08:19:09 2023

@author: Santiago y Laura
"""

import pandas as pd
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import MaximumLikelihoodEstimator
import numpy as np
import random

#%%

#Lee el archivo csv con las observaciones
df = pd.read_csv('D:/Users/Santiago/OneDrive - Universidad de los Andes/ACTD/Proyecto/desercion.csv',
                 delimiter = ';')

entrenamiento = df[df.columns][:int(0.8*int(len(df)))]
#print(entrenamiento)

prueba = df[df.columns][int(0.8*int(len(df))):]
df1 = entrenamiento
df2 = prueba

#%%

modelo = BayesianNetwork([("Admission grade", "Target"),
                          ("Debtor", "Target"), ("Tuition fees up to date", "Target"),
                          ("Scholarship holder", "Target"), ("Age at enrollment Group", "Target"),
                          ("Approved ratio sem 1", "Target"), ("Approved ratio sem 2", "Target"),
                          ("Curricular units 1st sem (grade)", "Target"), ("Curricular units 2nd sem (grade)", "Target")])

# modelo = BayesianNetwork([("Admission grade", "Target"), ("Displaced", "Target")])

emv = MaximumLikelihoodEstimator (model = modelo , data = df1)

# %%
#CPDs estimadas por máxima verosimilitud
cpdem_AG = emv.estimate_cpd(node ="Admission grade")
cpdem_Debt = emv.estimate_cpd(node ="Debtor")
cpdem_Tuit = emv.estimate_cpd(node ="Tuition fees up to date")
cpdem_Scho = emv.estimate_cpd(node ="Scholarship holder")
cpdem_Age = emv.estimate_cpd(node ="Age at enrollment Group")
cpdem_Appr1 = emv.estimate_cpd(node ="Approved ratio sem 1")
cpdem_Appr2 = emv.estimate_cpd(node ="Approved ratio sem 2")
cpdem_Grade1 = emv.estimate_cpd(node ="Curricular units 1st sem (grade)")
cpdem_Grade2 = emv.estimate_cpd(node ="Curricular units 2nd sem (grade)")
cpdem_Targ = emv.estimate_cpd(node ="Target")

modelo.add_cpds(cpdem_AG, cpdem_Debt, 
                cpdem_Tuit, cpdem_Scho, cpdem_Age, 
                cpdem_Appr1, cpdem_Appr2,
                cpdem_Grade1, cpdem_Grade2, cpdem_Targ)

# modelo.add_cpds(cpdem_AG, cpdem_Disp, cpdem_Targ)

#%%

from pgmpy.inference import VariableElimination
infer = VariableElimination(modelo)
pred = []
Targ_pred = []
j = 0

for i in df2.index:
    Targ_pred.append(infer.query(["Target"], evidence={"Admission grade": df2["Admission grade"][i],
                                                "Debtor": df2["Debtor"][i], "Tuition fees up to date": df2["Tuition fees up to date"][i],
                                                "Scholarship holder": df2["Scholarship holder"][i], "Age at enrollment Group": df2["Age at enrollment Group"][i],
                                                "Approved ratio sem 1": df2["Approved ratio sem 1"][i], "Approved ratio sem 2": df2["Approved ratio sem 2"][i],
                                                "Curricular units 1st sem (grade)": df2["Curricular units 1st sem (grade)"][i], "Curricular units 2nd sem (grade)": df2["Curricular units 2nd sem (grade)"][i]}))
    # El 1 significa que no deserta, 0 es que deserta
    if Targ_pred[j].values[1] > 0.5:
        pred.append(1)
    else:
        pred.append(0)
    j = j+1

# from pgmpy.inference import VariableElimination
# infer = VariableElimination(modelo)
# pred = []
# Targ_pred = []
# j = 0

# for i in df2.index:
#     Targ_pred.append(infer.query(["Target"], evidence={"Admission grade": df2["Admission grade"][i], "Displaced": df2["Displaced"][i]}))
#     # El 1 significa que no deserta, 0 es que deserta
#     if Targ_pred[j].values[1] > 0.5:
#         pred.append(1)
#     else:
#         pred.append(0)
#     j = j+1

#%%
from sklearn import metrics

#Retorneme la matriz de confusión
confusion_matrix = metrics.confusion_matrix(df2['Target'], pred)
tn, fp, fn, tp = metrics.confusion_matrix(df2['Target'], pred).ravel()
print(tn, fp, fn, tp)

accuracy = metrics.accuracy_score(df2['Target'], pred)
print(accuracy)

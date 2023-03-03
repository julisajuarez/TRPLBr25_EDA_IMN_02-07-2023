#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fmin_tnc, differential_evolution
from scipy.special import gamma
from scipy.signal import fftconvolve

from functions import *
from scipy import signal
# from Fit_functions_with_irf import *
from scipy.optimize import Bounds


# In[2]:


"""Recycle params for plotting"""
plt.rc('xtick', labelsize = 15)
plt.rc('xtick.major', pad = 3)
plt.rc('ytick', labelsize = 15)
plt.rc('lines', lw = 0.5, markersize = 20)
plt.rc('legend', fontsize = 10)


# *DATA SET 01/31/2023*: **Time-Resolved Photoluminesence**

# In[143]:


JTRPL= pd.read_csv('~/Desktop/TRPL:Lifetime Data/01:31:2023/Julisa_series.csv', names=['Time', 'Count1','Count2','Count3','Count4', 'Count5', 'Count6'],delimiter=',',index_col=False)
MTRPL= pd.read_csv('~/Desktop/TRPL:Lifetime Data/01:31:2023/Margherita_series.csv', names=['Time', 'Count1','Count2','Count3','Count4', 'Count5', 'Count6'],delimiter=',',index_col=False)

JTRPL1 = JTRPL[0:31250]
MTRPL1 = MTRPL[0:31249]

#for loop data through a graph
colors = ['coral','cyan','lime','violet', 'purple', 'orange']
C = ['Ref','Ref','Imn1%','Imn1%','Imn 0.5%', 'Imn 2%']
for i in range(1,7):
    plt.plot(JTRPL1['Time'].values, JTRPL1['Count'+str(i)].values/np.max(JTRPL1['Count'+str(i)].values), label = C[i-1], c= colors[i-1])
    plt.yscale('log')
plt.legend()


# In[142]:


colors = ['coral','cyan','lime','violet', 'orange']
C = ['Ref','Imn 1%','Imn 1%','Imn 0.5%','Imn 2%']
for i in range(1,6):
    plt.plot(MTRPL1['Time'].values, MTRPL1['Count'+str(i)].values/np.max(JTRPL1['Count'+str(i)].values), label = C[i-1], c= colors[i-1])
    plt.yscale('log')
plt.legend()


# In[5]:


#if i shoose to graph individually
#plt.plot(JTRPL1['Time'].values, JTRPL1['Count1'].values/np.max(JTRPL1['Count1'].values), label = 'A1', c= 'red')
#plt.yscale('log')


# *DATA SET 01/31/2023*: **Lifetime**

# In[ ]:





# In[ ]:





# *DATA SET 02/07/2023*: **TRPL**

# In[133]:


Timex=pd.Series(np.arange(0,4000,0.256)) #add the time scale by looking at the resolution and frequency of your measurement- for each pulse!! 1/250kHz
J1refTRPL= pd.read_csv('~/Desktop/TRPL:Lifetime Data/02_07_2023_eda_imn_surface/J1_ref_eda1mM_Imn1mM_eda20mM_250KHz_256ps_1pcint.txt',names=['Count1','Count2','Count3','Count4'], skiprows = 10 , delimiter='	',index_col=False)
#convert pandas --> np arrays!!! easier to work with
colors = ['coral','cyan','lime','violet']
C = ['REF','EDA1mM','IMN1mM','EDA20mM']
for i in range(1,5):
    plt.plot(Timex, J1refTRPL['Count'+str(i)].values/np.max(J1refTRPL['Count'+str(i)].values), label = C[i-1], c= colors[i-1])
    plt.yscale('log')
plt.legend()


# In[132]:


J1refTRPL1 = pd.concat([Timex,J1refTRPL], axis =1)
Time = J1refTRPL1.iloc[:,0].astype(np.int64) #calls and converts Timex from float to int
fixedJ1refTRPL1 = J1refTRPL1.drop(J1refTRPL1.index[range(J1refTRPL1['Count1'].idxmax())])
fixedJ1refTRPL1.columns = ['Time', 'Count1','Count2','Count3','Count4']

colors = ['coral','cyan','lime','violet']
C = ['REF','EDA1mM','IMN1mM','EDA20mM']
for i in range(1,5):
    plt.plot(fixedJ1refTRPL1['Time'], fixedJ1refTRPL1['Count'+str(i)].values, label = C[i-1], c= colors[i-1])
    plt.yscale('log')
plt.legend()


# In[8]:


print(J1refTRPL1['Count1'].idxmax())   #index where max happens


# In[9]:


print(fixedJ1refTRPL1)
print(np.shape(fixedJ1refTRPL1.values[:,1:]))


# In[10]:


fixedJ1refTRPL1['Count'+str(i)].values/np.max(fixedJ1refTRPL1['Count'+str(i)].values)


# In[61]:


J1refTRPL1 = pd.concat([Timex,J1refTRPL], axis =1)
Time = J1refTRPL1.iloc[:,0].astype(np.int64) #calls and converts Timex from float to int
fixedJ1refTRPL1 = J1refTRPL1.drop(J1refTRPL1.index[range(J1refTRPL1['Count1'].idxmax())])
fixedJ1refTRPL1.columns = ['Time', 'Count1','Count2','Count3','Count4']

J1tau1 = []
J1tau2 = []
J1tauavg = []
colors = ['coral','cyan','lime','violet']
colors2 = ['r','b','g','m']
C = ['REF','EDA1mM','IMN1mM','EDA20mM']
for i in range(1,5):
    plt.plot(fixedJ1refTRPL1['Time'].values , fixedJ1refTRPL1['Count'+str(i)].values/np.max(fixedJ1refTRPL1['Count'+str(i)].values), label = C[i-1], c= colors[i-1])
for i in range(1,5):
    biexp_Count= double_exp_fit(fixedJ1refTRPL1['Count'+str(i)].values/np.max(fixedJ1refTRPL1['Count'+str(i)].values),fixedJ1refTRPL1['Time'].values , tau1_bounds=(0,1000), a1_bounds=(0,1000), tau2_bounds=(0,10000), a2_bounds=(0,1000))
    plt.plot(fixedJ1refTRPL1['Time'].values, biexp_Count[5], colors2[i-1])
    plt.legend()
    J1tau1.append(biexp_Count[0]) #places values into a list 
    J1tau2.append(biexp_Count[2])
    J1tauavg.append(biexp_Count[4])
plt.yscale('log')
plt.xlabel('Time (ns)', fontsize =  20)
plt.ylabel('Counts', fontsize =  20)
plt.show()
print(J1tauavg)


# In[28]:





# In[ ]:


#running this for all data, plot this for every decay, plot of the average lifetime for each decay
#same thing for the stretched exponential 


# In[62]:


J2refTRPL= pd.read_csv('~/Desktop/TRPL:Lifetime Data/02_07_2023_eda_imn_surface/J2_ref_imn20mM_Imn1mM_eda1mM_250KHz_256ps_1pcint.txt', names=['Count1','Count2','Count3','Count4'], skiprows = 10 ,delimiter='	',index_col=False)
J2refTRPL1 = pd.concat([Timex,J2refTRPL], axis =1)
Time1 = J2refTRPL1.iloc[:,0].astype(np.int64) 
fixedJ2refTRPL1 = J2refTRPL1.drop(J2refTRPL1.index[range(J2refTRPL1['Count1'].idxmax())])
fixedJ2refTRPL1.columns = ['Time1', 'Count1','Count2','Count3','Count4']   

J2tau1 = []
J2tau2 = []
J2tauavg = []
colors = ['coral','cyan','lime','violet']
colors2 = ['r','b','g','m']
C = ['REF','IMN20mM','IMN1mM','EDA1mM']
for i in range(1,5):
    plt.plot(fixedJ2refTRPL1['Time1'].values , fixedJ2refTRPL1['Count'+str(i)].values/np.max(fixedJ2refTRPL1['Count'+str(i)].values), label = C[i-1], c= colors[i-1])
for i in range(1,5):
    biexp_Count= double_exp_fit(fixedJ2refTRPL1['Count'+str(i)].values/np.max(fixedJ2refTRPL1['Count'+str(i)].values),fixedJ2refTRPL1['Time1'].values , tau1_bounds=(0,1000), a1_bounds=(0,1000), tau2_bounds=(0,10000), a2_bounds=(0,1000))
    plt.plot(fixedJ2refTRPL1['Time1'].values, biexp_Count[5], colors2[i-1])
    plt.legend()
    J2tau1.append(biexp_Count[0])
    J2tau2.append(biexp_Count[2])
    J2tauavg.append(biexp_Count[4])
plt.yscale('log')
plt.xlabel('Time (ns)',fontsize =  20)
plt.ylabel('Counts',fontsize =  20)
plt.show()
print(J2tauavg)


# In[63]:


M1refTRPL= pd.read_csv('~/Desktop/TRPL:Lifetime Data/02_07_2023_eda_imn_surface/M1_ref_eda1mM_Imn1mM_eda20mM_250KHz_256ps_1pcint.txt', names=['Count1','Count2','Count3','Count4'], skiprows = 10 ,delimiter='	',index_col=False)
M1refTRPL1 = pd.concat([Timex,M1refTRPL], axis =1)
Time2 = M1refTRPL1.iloc[:,0].astype(np.int64) 
fixedM1refTRPL1 = M1refTRPL1.drop(M1refTRPL1.index[range(M1refTRPL1['Count1'].idxmax())])
fixedM1refTRPL1.columns = ['Time2', 'Count1','Count2','Count3','Count4'] 

M1tau1 = []
M1tau2 = []
M1tauavg = []
colors = ['coral','cyan','lime','violet']
colors2 = ['r','b','g','m']
C = ['REF','EDA1mM','IMN1mM','EDA20mM']
for i in range(1,5):
    plt.plot(fixedM1refTRPL1['Time2'].values , fixedM1refTRPL1['Count'+str(i)].values/np.max(fixedM1refTRPL1['Count'+str(i)].values), label = C[i-1], c= colors[i-1])
for i in range(1,5):
    biexp_Count= double_exp_fit(fixedM1refTRPL1['Count'+str(i)].values/np.max(fixedM1refTRPL1['Count'+str(i)].values),fixedM1refTRPL1['Time2'].values , tau1_bounds=(0,1000), a1_bounds=(0,1000), tau2_bounds=(0,10000), a2_bounds=(0,1000))
    plt.plot(fixedM1refTRPL1['Time2'].values, biexp_Count[5], colors2[i-1])
    plt.legend()
    M1tau1.append(biexp_Count[0])
    M1tau2.append(biexp_Count[2])
    M1tauavg.append(biexp_Count[4])
plt.yscale('log')
plt.xlabel('Time (ns)', fontsize =  20)
plt.ylabel('Counts', fontsize =  20)
plt.show()
print(M1tauavg)


# In[64]:


M2refTRPL= pd.read_csv('~/Desktop/TRPL:Lifetime Data/02_07_2023_eda_imn_surface/M2_ref_eda1mM_Imn1mM_imn20mM_250KHz_256ps_1pcint.txt', names=['Count1','Count2','Count3','Count4'], skiprows = 10 ,delimiter='	',index_col=False)
M2refTRPL1 = pd.concat([Timex,M2refTRPL], axis =1)
Time3 = M2refTRPL1.iloc[:,0].astype(np.int64) 
fixedM2refTRPL1 = M2refTRPL1.drop(M2refTRPL1.index[range(M2refTRPL1['Count1'].idxmax())])
fixedM2refTRPL1.columns = ['Time3', 'Count1','Count2','Count3','Count4']  

M2tau1 = []
M2tau2 = []
M2tauavg = []
colors = ['coral','cyan','lime','violet']
colors2 = ['r','b','g','m']
C = ['REF','EDA1mM','IMN1mM','IMN20mM']
for i in range(1,5):
    plt.plot(fixedM2refTRPL1['Time3'].values , fixedM2refTRPL1['Count'+str(i)].values/np.max(fixedM2refTRPL1['Count'+str(i)].values), label = C[i-1], c= colors[i-1])
for i in range(1,5):
    biexp_Count= double_exp_fit(fixedM2refTRPL1['Count'+str(i)].values/np.max(fixedM2refTRPL1['Count'+str(i)].values),fixedM2refTRPL1['Time3'].values , tau1_bounds=(0,1000), a1_bounds=(0,1000), tau2_bounds=(0,10000), a2_bounds=(0,1000))
    plt.plot(fixedM2refTRPL1['Time3'].values, biexp_Count[5], colors2[i-1])
    plt.legend()
    M2tau1.append(biexp_Count[0])
    M2tau2.append(biexp_Count[2])
    M2tauavg.append(biexp_Count[4])
plt.yscale('log')
plt.xlabel('Time (ns)',fontsize =  20)
plt.ylabel('Counts',fontsize =  20)
plt.show()
print(M2tauavg)


# In[130]:


tauavg_df = pd.DataFrame({'J1Tauavg': J1tauavg, 
                       'J2Tauavg': J2tauavg,
                       'M1Tauavg': M1tauavg,
                       'M2Tauavg': M2tauavg}, 
                  index=list('ABCD'))

print(tauavg_df)
#df_lists = tau1_df[['J1Tau1','J2Tau1', 'M1Tau1', 'M2Tau1']].unstack().apply(pd.Series)
#df_lists.plot.bar(rot=0, cmap=plt.cm.jet, fontsize=8, width=0.7, figsize=(8,4))


# In[131]:


boxplot_df= pd.read_csv('~/Desktop/TRPL:Lifetime Data/02_07_2023_eda_imn_surface/02-07-2023Br25_EDA_IMN_boxplot.csv',index_col=False)
print(boxplot_df)


# In[127]:


fig, ax = plt.subplots()
ax.boxplot(df.values, patch_artist=True, whis=100, boxprops=dict(facecolor='lightgreen'))
plt.plot([1,2,3,4,5],df.values, ".", color = "m", ms = "10", mec = "k", mew = "0.75")

ax.set_xticklabels(["Control", "EDA 1mM", "IMN 1mM", "EDA 20mM"], fontsize = 10)
plt.title('Br25 TRPL Tau_Average')
plt.ylabel("Time (ns)")
plt.show()


# In[ ]:





# In[ ]:





# In[92]:





# In[ ]:





# 
# **DATA SET 02/07/2023**: *Stretched Exponential*
# 

# In[ ]:


stretchexp_Count = stretch_exp_fit(fixedJ1refTRPL1['Count1'].values/np.max(fixedJ1refTRPL1['Count1'].values), fixedJ1refTRPL1['Time'].values, Tc = (0,1e4*1e-9), Beta = (0,1), A = (0,1.5))
plt.plot(stretchexp_Count)


# In[22]:


J1refTRPL1 = pd.concat([Timex,J1refTRPL], axis =1)
Time = J1refTRPL1.iloc[:,0].astype(np.int64) #calls and converts Timex from float to int
fixedJ1refTRPL1 = J1refTRPL1.drop(J1refTRPL1.index[range(J1refTRPL1['Count1'].idxmax())])
fixedJ1refTRPL1.columns = ['Time', 'Count1','Count2','Count3','Count4']

colors = ['coral','cyan','lime','violet']
colors2 = ['r','b','g','m']
C = ['REF','EDA1mM','IMN1mM','EDA20mM']
for i in range(1,5):
    plt.plot(fixedJ1refTRPL1['Time'].values , fixedJ1refTRPL1['Count'+str(i)].values/np.max(fixedJ1refTRPL1['Count'+str(i)].values), label = C[i-1], c= colors[i-1])
for i in range(1,5):
    stretchexp_Count = stretch_exp_fit(fixedJ1refTRPL1['Count'+str(i)].values/np.max(fixedJ1refTRPL1['Count'+str(i)].values), fixedJ1refTRPL1['Time'].values, Tc = (0,1e4*1e-9), Beta = (0,1), A = (0,1.5))  
    plt.plot(fixedJ1refTRPL1['Time'].values, stretchexp_Count[4], colors2[i-1])
    plt.legend()

plt.yscale('log')
plt.xlabel('Time (ns)',fontsize =  20)
plt.ylabel('Counts',fontsize =  20)
plt.show()


# In[ ]:


biexp_Count= double_exp_fit(fixedM2refTRPL1['Count'+str(i)].values/np.max(fixedM2refTRPL1['Count'+str(i)].values),fixedM2refTRPL1['Time3'].values , tau1_bounds=(0,1000), a1_bounds=(0,1000), tau2_bounds=(0,10000), a2_bounds=(0,1000))


# In[ ]:





# In[ ]:


#now get rid of first line (start at max) and fit with exponateial (Aaorons code probably)


# In[ ]:





# In[ ]:





# In[ ]:





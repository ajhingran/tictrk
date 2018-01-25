
# coding: utf-8

# In[1]:

"""
Created on Tue Jan  9 00:18:38 2018

@author: ajhingran
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
#matplotlib inline
plt.style.use('ggplot')


# In[2]:


columns = ['attitude_roll', 'attitude_pitch', 'attitude_yaw',                    'rotation_rate_x','rotation_rate_y','rotation_rate_z',                    'gravity_x','gravity_y','gravity_z',                    'user_acc_x','user_acc_y','user_acc_z']
activities = ['walking', 'running', 'climbing', 'nose']


# In[3]:


print columns
print ['timestamp'] + columns


# In[4]:


def read_data(file_path):
#    column_names = ['timestamp', 'attitude_roll', 'attitude_pitch', 'attitude_yaw',\
#                   'rotation_rate_x','rotation_rate_y','rotation_rate_z',\
#                    'gravity_x','gravity_y','gravity_z',\
#                    'user_acc_x','user_acc_y','user_acc_z']
    column_names = ['timestamp'] + columns
    print column_names
    data = pd.read_csv(file_path,header = None, names = column_names,comment=';')
    return data
def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma


# In[5]:


dataset = read_data('/Users/jhingran/Downloads/dipu_nose_20180116_19_57_56.csv')
for i in range(12):
    dataset[columns[i]] = feature_normalize(dataset[columns[i]])


# In[6]:


numrows = len(dataset[columns[0]])/50
#transposed is the new dataset
transposed = []
print numrows

for i in range(len(columns)):
    insertcolumn = []
    start = 0
    for j in range(numrows-1):
#        print i, j
        insertrow = []
        for k in range(100):
            insertrow.append(dataset[columns[i]][start + k])
        if (j > 5 and j < numrows-5):
            insertcolumn.append(insertrow)
        start = start + 50
    print len(insertcolumn)
    transposed.append(insertcolumn)     
       
    


# In[7]:


print transposed[0][0][50]


# In[8]:


print transposed[0][1][0]


# In[9]:


print len(transposed[0])


# In[10]:


from random import *


# In[11]:


filepath = '/Users/jhingran/work/tfwork/LSTM-Human-Activity-Recognition/mydata/'
numrows = len(transposed[0])
train_test_split = []
num_train = 0
num_test = 0
for i in range(numrows):
    if random() < 0.7:
        train_test_split.append('0')
        num_train += 1
    else:
        train_test_split.append('1')
        num_test +=1
for files in range(12):
    filename_train = filepath + 'train/Inertial Signals/' + columns[files]
    filename_test = filepath + 'test/Inertial Signals/' + columns[files]
    print filename_train, filename_test
    fp_train= open(filename_train, 'a')
    fp_test= open(filename_test,'a')
    for i in range(numrows):
        if train_test_split[i] == '0':
            fp = fp_train
        else:
            fp = fp_test
        for j in range(100):
            fp.write('  ')
            fp.write('{}'.format(transposed[0][i][j]))
        fp.write('\n')
    fp_train.close()
    fp_test.close()



    
    


# In[12]:


print numrows, num_train, num_test


# In[13]:


filename_train = filepath + 'train/' + 'y_'
filename_test = filepath + 'test/' + 'y_'
index = activities.index('nose') + 1
print filename_train, filename_test
fp_train= open(filename_train, 'a')
fp_test= open(filename_test,'a')
for i in range(numrows):
    if train_test_split[i] == '0':
        fp = fp_train
    else:
        fp = fp_test
    fp.write('{}'.format(index))
    fp.write('\n')
fp_train.close()
fp_test.close()


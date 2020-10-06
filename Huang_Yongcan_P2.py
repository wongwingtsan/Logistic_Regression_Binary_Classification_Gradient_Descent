#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install seaborn')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import codecs

f = codecs.open('./P2train.txt', mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
line = f.readline()  
x1 = []
x2=[]
y=[]
while line:
    a = line.split()
    X1= a[0:1]   
    x1.append(X1) 
    X2=a[1:2]
    x2.append(X2)
    Y=a[2:3]
    y.append(Y)
    
    
    line = f.readline()
f.close()


# In[3]:


m=int(x1[0][0])
n=int(x2[0][0])+22
del x1[0]
del x2[0]
del y[0]


# In[4]:


with open("./P2train.txt", 'a') as fout1:
    fout1.seek(0)
    fout1.truncate()
    fout1.write(str(m)+"\t"+str(n)+"\n")
    thePower =4
    for g in range(m):
        for j in range(thePower +1):
            for i in range(thePower+1):
                temp = ((float(x1[g][0]))**i)*((float(x2[g][0]))**j)
                if (temp!=1):
                    fout1.write(str(temp)+"\t")
        fout1.write(str(y[g][0])+"\n")  
fout1.close()
print("Training filename: Huang_Yongcan_P2Train.txt")


# In[5]:


df = pd.read_csv('./P2train.txt',sep='\t',names=['X1','X2','X3','X4','X5','X6','X7','X8',
                                                 'X9','X10','X11','X12','X13','X14','X15','X16',
                                                 'X17','X18','X19','X20','X21','X22','X23','X24','Y'])
df = df.drop([0])


# In[6]:


f_test = codecs.open('./P2test.txt', mode='r', encoding='utf-8')  
line_test= f_test.readline()   
x1_test = []
x2_test=[]
y_test=[]
while line_test:
    a_test = line_test.split()
    X1_test= a_test[0:1]  
    x1_test.append(X1_test) 
    X2_test=a_test[1:2]
    x2_test.append(X2_test)
    Y_test=a_test[2:3]
    y_test.append(Y_test)
    
    
    line_test = f_test.readline()
f_test.close()


# In[7]:


m_test=int(x1_test[0][0])
n_test=int(x2_test[0][0])+22
del x1_test[0]
del x2_test[0]
del y_test[0]


# In[8]:


with open("./P2test.txt", 'a') as fout2:
    fout2.seek(0)
    fout2.truncate()
    fout2.write(str(m_test)+"\t"+str(n_test)+"\n")
    thePower =4
    for g in range(m_test):
        for j in range(thePower +1):
            for i in range(thePower+1):
                temp_test = ((float(x1_test[g][0]))**i)*((float(x2_test[g][0]))**j)
                if (temp_test!=1):
                    fout2.write(str(temp_test)+"\t")
        fout2.write(str(y_test[g][0])+"\n")  
fout2.close()
print("Test filename: Huang_Yongcan_P2Test.txt")


# In[16]:


df_test = pd.read_csv('./P2test.txt',sep='\t',names=['X1_test','X2_test','X3_test','X4_test','X5_test','X6_test','X7_test','X8_test',
                                                     'X9_test','X10_test','X11_test','X12_test','X13_test','X14_test','X15_test','X16_test',
                                                     'X17_test','X18_test','X19_test','X20_test','X21_test','X22_test','X23_test','X24_test','Y_test'])
df_test = df_test.drop([0])


# Logistic Regression

# In[10]:


def logistic_regression( x , theta):
    z  = x @ (-theta.T)
    eulers = np.power(math.e , z)
    pred_y = 1 / (1 + eulers)
    
    classified = []
    for i in pred_y:
        if i >= 0.5:
            classified.append(1)
        else:
            classified.append(0)
                
    return pred_y , classified


# Cost Function

# In[11]:


def costFunction(x,y,theta):
    z = x @ (-theta.T)
    euler = np.power(math.e , z)
    pred_y = 1/(1 + euler)
    
    costF = ( (-y* (np.log(pred_y)) ) - (1-y)* (np.log(1-pred_y)) )
    
    return sum(costF)/len(x)


# In[12]:


def scale(x,lambuda):
    k=np.amax(x)-np.amin(x)
    x=x-lambuda
    
    return x/k
    


# In[18]:


X_train=df[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23','X24']].to_numpy()
X_train=scale(X_train,2.09)
Y_train=df[['Y']].to_numpy()
X_test=df_test[['X1_test','X2_test','X3_test','X4_test','X5_test','X6_test','X7_test','X8_test','X9_test','X10_test','X11_test','X12_test','X13_test','X14_test','X15_test','X16_test','X17_test','X18_test','X19_test','X20_test','X21_test','X22_test','X23_test','X24_test']].to_numpy()
X_test=scale(X_test,2.09)
Y_test=df_test[['Y_test']].to_numpy()


# In[19]:


theta=np.zeros([1,X_train.shape[1]])


# Gradient Descent

# In[20]:


def gradientDescent(x , y , theta , alpha , iterations):
    cost = np.zeros(iterations)
    for i in range(iterations):
        z = x @ (-theta.T)
        eulers = np.power(math.e , z)
        pred_y = 1/(1+eulers)
    
        partialDerivativeoftheta = (1/len(x))* (sum((pred_y - y)*x))
        
        theta = theta - (alpha*partialDerivativeoftheta)
        cost[i] = costFunction(x,y,theta)
        
    return theta , cost


# In[21]:


alpha = 0.3
iterations =10000


# In[22]:


coef , new_cost = gradientDescent(X_train , Y_train , theta , alpha , iterations)
print(coef)
print('the final J for train is {}'.format(np.amin(new_cost)))

cost_test=costFunction(X_test,Y_test,coef)
print('the final J for test is {}'.format(np.amin(cost_test)))


# In[29]:


initialcost=np.amax(new_cost)


# In[24]:



plt.plot(np.arange(iterations), new_cost, 'r')  
plt.xlabel('Iterations')  
plt.ylabel('Costs')  
plt.title('Error(cost function) vs. Training iteration')
plt.show()


# In[25]:



pred_y , y_pred= logistic_regression(X_test , coef)


# In[26]:


def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i][0]==y_pred[i]==1:
            TP += 1
        if y_pred[i]==1 and y_actual[i][0]!=y_pred[i]:
            FP += 1
        if y_actual[i][0]==y_pred[i]==0:
            TN += 1
        if y_pred[i]==0 and y_actual[i][0]!=y_pred[i]:
            FN += 1
    accuracy=(TP+TN)/(TP+TN+FP+FN)
    Precision=(TP)/(TP+FP)
    Recall=(TP)/(TP+FN)
    F1=2*(1/((1/Precision)+(1/Recall)))
    print('TP is {}, FP is {}, TN is{}, FN is {} \n accuracy is {}, Precision is {}, Recall is {}, F1 is {} '.format(TP,FP,TN
                                                                                                                     ,FN,accuracy,Precision,Recall, F1))
        #return TP, FP, TN, FN


# In[27]:


perf_measure(Y_test, y_pred)


#!/usr/bin/env python
# coding: utf-8

# # NUMPY
# 

# In[1]:


import numpy as np
x = np.array([1,2,3])
y = np.array([8,9,10])
x + y


# In[2]:


x


# In[3]:


y


# In[4]:


z = np.array([[[1.2,2],[5,6]],[[3,4],[7,8]]])
z.shape


# In[5]:


x = np.array([1,5,9])
x.sum()


# In[6]:


import numpy as np
x = np.random.normal(size=50)


# In[7]:


x


# In[8]:


y = x + np.random.normal(loc=10,scale=5,size=50)


# In[9]:


np.corrcoef(x,y)


# In[10]:


import numpy as np


# In[11]:


get_ipython().run_line_magic('pinfo', 'np.random.default_rng')


# In[12]:


rng = np.random.default_rng(1307)
print(rng.normal(scale=5,size=3))
rng2 = np.random.default_rng(1303)
print(rng2.normal(scale=5,size=3))


# In[13]:


import numpy as np
rng = np.random.default_rng(3)
y = rng.standard_normal(10)

y, np.mean(y),y.mean()


# In[14]:


np.var(y) , y.var() , np.mean((y - y.mean())**2)


# In[15]:


X = rng.standard_normal((10,3))
X


# In[16]:


X.mean(1)


# In[17]:


z = X[0,]
np.mean(z)


# In[18]:


import numpy as np
rng = np.random.default_rng(4)
X = rng.standard_normal((10,3))
X


# In[19]:


X.mean(0)


# In[20]:


X.mean(1)


# In[21]:


np.arange(-3,3,0.5)


# In[22]:


np.arange(0,5,0.5)


# # MATPLOTLIB

# In[23]:


from matplotlib.pyplot import subplots as sb


# In[24]:


fig,ax = sb(figsize = (8,8))
x = rng.standard_normal(25)
y = rng.standard_normal(25)
ax.plot(x,y,'o')


# In[25]:


ax.plot(x,y,'o')


# In[26]:


fig,ax = sb(figsize=(6,6))
x = np.arange(2,10,1)
y = x + 2
ax.plot

ax.plot(x,y,'r+');
ax.set_xlabel("X");
ax.set_ylabel("Y");
ax.set_title("X vs Y");
fig.set_size_inches(7,3)


# In[95]:


fig,axes = sb(nrows=2,ncols=2,figsize=(10,5))
y = rng.standard_normal(25)
x = rng.standard_normal(25)
axes[0,0].plot(x,y,'o');
axes[1,0].plot(x,y, color = 'orange');
axes[0,1].plot(x,y,color='green')
axes[1,1].scatter(x,y,marker='+',color="red");
fig.savefig("Figure.pdf",dpi=100)


# In[96]:


from matplotlib.pyplot import subplots
fig


# In[29]:


fig, ax = subplots(figsize=(8,8))
x = np.linspace(-np.pi,np.pi,50)
y = x
f = np.multiply.outer(np.cos(y),1/(1 + x**2))
ax.contour(x,y,f,levels=45);
ax.set_xlim([-2,2]);


# In[30]:


fig,ax = subplots(figsize=(8,8))
ax.imshow(f);


# In[31]:


import numpy as np
import pandas as pd


# In[32]:


import os


# In[35]:


os.chdir('C:/Users/LENOVO/Downloads/ALL/ALL')


# In[36]:


os.getcwd()


# In[37]:


Auto = pd.read_csv('Auto.csv')


# In[38]:


Auto


# In[39]:


np.unique(Auto['horsepower'])


# In[40]:


Auto = pd.read_csv('Auto.data', na_values = '?', delim_whitespace=True)
Auto['horsepower'].sum()


# In[41]:


Auto.shape


# In[42]:


Auto = Auto.dropna()


# In[43]:


Auto.shape


# In[44]:


Auto.columns


# In[45]:


Auto[:3]


# In[46]:


idx = Auto['year']>80
idx.sum()


# In[47]:


Auto.index


# In[48]:


Auto_re = Auto.set_index('name')
Auto_re


# In[49]:


x =['ford torino','vw pickup']
Auto_re.loc[x]


# In[50]:


Auto_re.iloc[[4,5]]


# In[51]:


Auto_re.iloc[:,[0,2,3]]


# In[52]:


Auto_re.iloc[[4,5],[0,2,3]]


# In[53]:


idx = Auto_re['year']>80
Auto_re.loc[idx,['weight','origin']].head()


# In[54]:


Auto_re.loc[lambda df: df['year']>80, ['weight','origin']].head()


# ##### suppose that we want all cars built after 1980 that achieve greater than 30 miles per gallon and weight and origin columns

# In[55]:


Auto_re.loc[lambda df: (df['year'])>80 & (df['mpg']>30), ['weight','origin']]


# #####  suppose that we want to retrieve all Ford and Datsun cars with displacement more than 300

# In[56]:


Auto_re.loc[lambda df: (df['displacement']>300) & (df.index.str.contains('ford') | df.index.str.contains('datsun')),['displacement','weight','origin']].head()


# ### FOR LOOP
# 

# In[57]:


sum = 0
for value in [3,4,5]:
   sum += value
print("total is:",sum)


# In[58]:


sum = 0
for value in [3,4,5]:
    for weight in [7,8,9]:
        sum += value * weight
sum


# In[59]:


# Perhaps a more common task would be to sum over (value, weight) pairs. For instance, to compute the average value of a
#random variable that takes on possible values 2, 3 or 19 with probability 0.2, 0.3, 0.5 respectively we would compute the 
#weighted sum. Tasks such as this can often be accomplished using the zip() function that loops over a sequence of tuples. 


# In[60]:


sum = 0 
for value, weight in zip([1,2,3],[0.3,0.2,0.7]):
    sum += value*weight
sum


# ### STRING FORMATTING 

# In[61]:


rng = np.random.default_rng(1)
A = rng.standard_normal((127,5))
M = rng.choice([0,np.nan], p = [0.8,0.2], size = A.shape)
A += M
D = pd.DataFrame(A, columns = ['food','bar','pickle','snack','popcorn'])
D.head()                    


# In[62]:


idx = D < 1
idx.sum()


# In[63]:


np.isnan(D).sum()


# In[64]:


for col in D.columns:
    template = 'Column "{0}" has {1: .2%} missing values'
    print(template.format(col,np.isnan(D[col]).mean()))


# ###### ------

# In[65]:


from matplotlib.pyplot import subplots


# In[66]:


fig, ax = subplots(figsize = (8,8))
ax.plot(Auto['horsepower'],Auto['mpg'],'o', color='green');


# In[67]:


ax = Auto.plot.scatter('horsepower','displacement');
ax.set_title('Horsepower vs Displacement');


# In[101]:


Auto.plot.scatter('horsepower','mpg',color='orange');


# In[69]:


Auto.cylinders = pd.Series(Auto.cylinders, dtype='category')
Auto.cylinders.dtype


# In[70]:


fig, ax = subplots(figsize = (8,8))
Auto.boxplot('mpg',by='cylinders',ax=ax);


# In[71]:


fig,ax = subplots(figsize = (8,8))
Auto.hist('mpg',color='red',bins=12,ax=ax);


# In[72]:


pd.plotting.scatter_matrix(Auto,figsize=(12,12),color='green');


# In[73]:


pd.plotting.scatter_matrix(Auto[['mpg','horsepower','weight']]);


# In[74]:


Auto[['mpg','horsepower']].describe()


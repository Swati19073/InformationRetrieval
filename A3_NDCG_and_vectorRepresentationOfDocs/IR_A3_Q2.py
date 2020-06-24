#!/usr/bin/env python
# coding: utf-8

# In[250]:



import nltk

from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
import string
import re
import inflect
import matplotlib.pyplot as plt


# In[251]:


f=open("assign3_ques2.txt",'r+')
myfile=f.read()


# In[252]:


with open("assign3_ques2.txt") as myfile:
    head = [next(myfile) for x in range(103)]

str1 = ""  
for i in head:  
    str1 += i 
print(str1)


# In[253]:


files=list(str1.split("\n"))
main_files=files


# In[254]:


# print(main_files)
final_unsorted=[]
for i in main_files:
    temp1=[]
    temp1=i.split(" ")
    final_unsorted.append(temp1)
print(final_unsorted)


# In[255]:


# import pandas as pd
# df = pd.DataFrame(files)
files.sort(reverse = True)
print(files)


# In[256]:


final_sorted=[]
for i in files:
    temp=[]
    temp=i.split(" ")
    final_sorted.append(temp)
    

print(final_sorted)


# In[257]:


# from __future__ import with_statement

# with open('output.txt', 'w') as f:
#     for _list in final:
#         for _string in _list:
#             #f.seek(0)
#             f.write(str(_string) + '\n')


# In[258]:


#making file of url with max dcg
# import  csv

# with open("out.csv","w") as f:
#     wr = csv.writer(f)
#     wr.writerows(final)


# In[259]:


list0=[]
list1=[]
list2=[]
list3=[]

for i in final_sorted:
    if i[0]=='3':
        list3.append(i[0])
    elif i[0]=='2':
        list2.append(i[0])
    elif i[0]=='1':
        list1.append(i[0])
    else:
        list0.append(i[0])


# In[260]:


def fact(num):
    factorial = 1
    if num < 0:
        print("Sorry, factorial does not exist for negative numbers")
    elif num == 0:
        return 1
    else:
        for i in range(1,num + 1):
            factorial = factorial*i
    return factorial

def combination(n,r):
    z1=n-r
    x=fact(n)
    y=fact(r)
    z=fact(z1)
    mul=y*z
    if(mul>0):
        com=x/mul
    return com


# In[261]:


#ans 1 
_3=len(list3)
_2=len(list2)
_1=len(list1)
_0=len(list0)

n=59
r=59
sum1=0
while(r>0):
    comb=combination(n,r)*fact(r)
    sum1+=comb
    r=r-1
f1=fact(_2)
f2=fact(_3)
f3=fact(_1)
p=f1*f2*f3

final_ans=p*sum1

print(final_ans)    


# In[ ]:





# In[262]:


import math
def DCG(list1):
    dcg_sum=0
    for i in range(len(list1)):
        l=math.log(i+1,2)
        if l!=0:
            x=list1[i]/l
            dcg_sum+=x
    return dcg_sum

# print(final)


# In[263]:


def ndcg(list1,list2):
    a=DCG(list1)
    b=DCG(list2)
    if b!=0:
        ndcg=a/b
    return ndcg
        


# In[264]:


max_dcg_list=[]
for i in final_sorted:
    for j in i:
        if(j=='1' or j=='2' or j=='3' or j=='0'):
            v=int(j)
    max_dcg_list.append(v)
print(len(max_dcg_list))
print("Max DCG is: ", DCG(max_dcg_list))
print("total number of combinations are: ",final_ans)


# In[265]:


#ndcg at 50
after_list=[]
before_list=[]
for i in final_sorted:
    for j in i:
        if(j=='1' or j=='2' or j=='3' or j=='0'):
            v1=int(j)
    after_list.append(v1)
    
for i1 in final_unsorted:
    for j1 in i1:
        if(j1=='1' or j1=='2' or j1=='3' or j1=='0'):
            v2=int(j1)
    before_list.append(v2)
top50_before=before_list[:50]
top50_after = after_list[:50]


# In[266]:


ndcg_50=ndcg(top50_before,top50_after)
print("NDCG at 50 is :", ndcg_50)


# In[267]:


ndcg_data=ndcg(before_list,after_list)
print("NDCG for whole dataset is :", ndcg_data)


# In[268]:


print(len(final_unsorted))
#part 3
# tf_idf_rel_sorted=[]
tf_idf_rel_unsorted=[]
for i in range(len(final_unsorted)-1):
    tempo=[]
    sp=final_unsorted[i][76].split(":")
    tempo.append(final_unsorted[i][0])
    tempo.append(sp[1])
    tf_idf_rel_unsorted.append(tempo)
#     tf_idf_rel_sorted.append(tempo)

cou=0
for i in tf_idf_rel_unsorted:
    if(i[0]>'0'):
        cou+=1
        


# In[269]:


# tf_idf_rel_sorted=sorted(tf_idf_rel_sorted, key = lambda x: float(x[1]), reverse= True)
# print(tf_idf_rel_unsorted)


# In[270]:


def pre_recall(list1,no_of_rel_doc):
    
    count=0
    rel_count=no_of_rel_doc
    rel=0
    pre=[]
    recall=[]
    for i in list1:
        if i[0]>'0':
            rel+=1
            count+=1
            p=rel/count
            r=rel/rel_count
            pre.append(p)
            recall.append(r)
        else:
            count+=1
            p=rel/count
            r=rel/rel_count
            pre.append(p)
            recall.append(r)
    return pre, recall
            
            
            


# In[271]:


precision, recall=pre_recall(tf_idf_rel_unsorted,cou)
print(precision)
print(recall)


# In[275]:


plt.plot(recall, precision,color="green")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.title('Precision-Recall Curve')
plt.show()


# In[ ]:





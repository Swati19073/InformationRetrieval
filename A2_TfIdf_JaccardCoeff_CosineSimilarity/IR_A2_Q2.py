#!/usr/bin/env python
# coding: utf-8

# In[1]:


import heapq
import string
import re
from collections import defaultdict
file1 = open("english2.txt","r+")
myfile=file1.read()

mylist=[]
mylist = list(myfile.split("\n"))  
type(mylist)
print(len(mylist))


# In[4]:


#https://stackoverflow.com/questions/2460177/edit-distance-in-python
def edit_distance(str1, str2):
    dictionary = {}
    for i in range(len(str1)+1):
        dictionary[i,0]=i
    for j in range(len(str2)+1):
        dictionary[0,j]=2*j
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if (str1[i-1]==str2[j-1]):
                cost=0
            else:
                cost=3
            dictionary[i,j] = min(dictionary[i, j-1]+2, dictionary[i-1, j]+1, dictionary[i-1, j-1]+cost)

    return dictionary[i,j]


# In[5]:


#sorts dictionary on the basis of edit distance
def sort_dictionary(list1):
    li=[]
    li=sorted(list1.items(), key = lambda kv:(kv[1], kv[0]))
    return li


# In[6]:


#https://stackoverflow.com/questions/41306684/get-top-5-largest-from-list-of-tuples-python
def get_k_tuples(list2,k):
    li2=[]
    li2=sorted(list2, key=lambda t: t[1], reverse=False)[:k]
    return li2


# In[10]:



temp_dict= defaultdict(list)
print("Enter the input string \n")
input_string=input()
#removing punctuations
input_string=input_string.translate(str.maketrans(" "," ",string.punctuation))
#removing numbers
input_string=re.sub(r"\d+","",input_string)
print("input the value of k")
k=int(input())
input_tokens=[]
words_not_in_dict=[]
input_tokens=input_string.split(" ")

for i in input_tokens:
    if i not in mylist:
        words_not_in_dict.append(i)
print(words_not_in_dict)
final_ans={}

for i in words_not_in_dict:
    temp={}
    list1=[]
    final_list=[]
    for j in mylist:
        x=edit_distance(i,j)
        if x!=0:
            temp[j]=x
    list1=sort_dictionary(temp)
    final_list=get_k_tuples(list1,k)
    final_ans[i]=final_list
print(final_ans)


# In[ ]:





# In[7]:





# In[ ]:





# In[ ]:





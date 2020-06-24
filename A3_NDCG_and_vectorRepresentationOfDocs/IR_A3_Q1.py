#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
import string
import re
import inflect


# In[2]:


import os
folders=os.listdir('20_newsgroups/')
# print(folders)


# In[3]:


filelist=[]
doc=[]
# count=0
for directory in folders:
    files=os.listdir('20_newsgroups/'+directory)
    files=files[1:]
    for i in files:
        doc.append(i)
    for file in files:
        f=open('20_newsgroups/'+directory+"/"+file)
        myfile=f.read()
        myfile=myfile.lower()
        myfile=myfile.translate(str.maketrans(" "," ",string.punctuation))
#         myfile=re.sub(r"\d+","",myfile)
        filelist.append(myfile)
    


# In[129]:


print(doc[0])


# In[4]:



doc_id={}
index=1
for i in doc:
    doc_id[i]=index
    index+=1
print(doc_id)


# In[5]:


print(len(filelist))


# In[6]:


import pickle
from num2words import num2words
lem=WordNetLemmatizer()
final_list=[]
idf_dict={}
docs_dictionary={}
# tf_dict={}
from nltk.tokenize import word_tokenize   
stop_words = set(stopwords.words('english')) 
# ind=0
index=0
for i in filelist:
    word_tokens = word_tokenize(i)
    mylist = [] 
    for w in word_tokens: 
        if w not in stop_words: 
            mylist.append(w) 
    temp1=[]
    temp2=[]
    for tokens in mylist:
        if(tokens.isdecimal()==True):
            tokens=num2words(tokens)
        word=lem.lemmatize(tokens)  
        temp1.append(word) 
        temp2.append(word)
    temp2=set(temp2)
    for i1 in temp2:
        if i1 not in idf_dict.keys():
            idf_dict[i1]=1
        else:
            idf_dict[i1]=idf_dict[i1]+1
    docs_dictionary[doc[index]]=temp1
    index+=1
    final_list.append(mylist)


# In[7]:


# tf_dict={}
# ind=0
# for i in final_list:
#     for j in i:
#         if j not in tf_dict.keys():
#             temp={}
#             tlist=[]
#             tf=i.count(j)
#             temp[ind]=tf
#             tlist.append((ind,tf))
#             tf_dict[j]=tlist
#         else:
#             tf_dict[j].append((ind,tf))
#     ind+=1
        


# In[8]:


# tf_dict1={}
# for i in tf_dict:
#     tf_dict1[i]=list(set(tf_dict[i]))


# In[9]:


print(len(tf_dict['peaceful']))


# In[12]:


# print(tf_dict['god'])
# tf_dict1={}
# for i in tf_dict:
#     ul=[]
#     for x in tf_dict[i]:
#         print(x)
#         if x not in ul:
#             ul.append(x)
#     tf_dict1[i]=ul
# print(docs_dictionary['49960'])


# In[13]:


# print(len(tf_dict1['peaceful']))


# In[14]:


# print(docs_dictionary["49960"])
# print(idf_dict['hello'])


# In[15]:



main_list=[]
for i in final_list:
    main_list.append(list(set(i)))


# In[16]:


file2=open("assign3_ques1.txt", 'r+')
gd=file2.read()
# print(gd)


gd_scores = list(gd.split("\n")) 

print(len(gd_scores))

gd_scores1={}
#dictionary containing doc_id as key and reviws corresponding to the docs as values
c=1
for i in range(len(gd_scores)-1):
    sp=gd_scores[i].split(" ")
    y=int(sp[1])
    gd_scores1[c]=y
    c+=1
# print(gd_scores1)

sum_ = 0
for i in gd_scores1: 
    sum_ = sum_ + gd_scores1[i] 
print(sum_)
#normalizing gd by dividing each value with the sum of all the values
c1=0   
gd_final={}
for i in gd_scores1:
    gd_final[c1]=gd_scores1[i]/sum_
    c1+=1
    


# In[17]:


# print(gd_final)


# In[18]:



def firstNpairs(dictionary, r):
    pairs = {k: dictionary[k] for k in list(dictionary)[:r]}
    return pairs
top_r=firstNpairs(gd_final,100)
# print(top_r)


# In[ ]:





# In[19]:


#dictionary with gd values for each term
dictionary={}
my_file_list=[]
for i in range(len(main_list)):
    my_file_list.append(i)
    for j in main_list[i] :
        if j not in dictionary.keys():
            list1=[]
            temp_dict={}
            temp_dict[i]=gd_final[i]
            list1.append(temp_dict)
            dictionary[j]=list1
        else:
            dictionary[j].append(temp_dict)


# In[228]:


print(len(dictionary['peaceful']))


# In[20]:


# print(dictionary['hello'])
def get_k_tuples(list2,k):
    li2=[]
    li2=sorted(list2, key=lambda t: t[1], reverse=True)[:k]
    return li2


def sort_dictionary(list1):
    li=[]
    li=sorted(list1.items(), key = lambda kv:(kv[1], kv[0]))
    return li


# In[21]:


#cosine
import math
def cosine_similarity(query,k):
    final_cosine={}
    query=query.lower()
    query=query.translate(str.maketrans("","",string.punctuation))
    query= word_tokenize(query)
    t=[]
    for w in query:
        if w not in stop_words: 
            t.append(w)
    input_txt=[]
    for i in t:
        if(i.isdecimal()==True):
            i=num2words(i)
        input_txt.append(lem.lemmatize(i))
    for x in input_txt:
        if x not in final_idf.keys():
            final_idf[x]=0
    for d in docs_dictionary:
        query_tfidf=[]
        doc_tfidf=[]
        for word in input_txt:
            c_=input_txt.count(word)
            
#             tf_=c_/len(input_txt)
            tf_=c_
            idf1_=final_idf[word]
            tf_idf_=tf_*idf1_
            query_tfidf.append(tf_idf_)
            c=docs_dictionary[d].count(word)
#             tf=c/len(docs_dictionary[d])
            tf=c
            idf1=final_idf[word]
            tf_idf=tf*idf1
            doc_tfidf.append(tf_idf)
#         print(query_tfidf)
        mul_list=[]
        q_sq_list=[]
        d_sq_list=[]
        for i in range(len(query_tfidf)):
            mul=query_tfidf[i]*doc_tfidf[i]
            mul_list.append(mul)
            qs=query_tfidf[i]*query_tfidf[i]
            ds=doc_tfidf[i]*doc_tfidf[i]
        s1=0
        s2=0
        s3=0
#         fin=0
        for s in range(len(mul_list)):
            s1=s1+mul_list[s]
            s2=s2+query_tfidf[s]
            s3=s3+doc_tfidf[s]
        if s2>0 and s3>0:
            
            s2=math.sqrt(s2)
            s3=math.sqrt(s3)
        did=doc_id[d]
        gd1=gd_final[did]
        if (s2*s3)!=0:
            fin=s1/(s2*s3)+gd1
            final_cosine[d]=fin
    list1=[]    
    list1=sort_dictionary(final_cosine)
    output=[]
    output=get_k_tuples(list1,k)
    return output


# In[22]:


#computing idf using logbase10(N/count_of_each_term)
import math
from math import log
idf={}
def idf_variation(k):
    
    if var==1:
        for i in idf_dict:
            idf[i]=len(folders)/idf_dict[i]
    elif var==2:
        for i in idf_dict:
            x=len(folders)/(idf_dict[i] +1)
            idf[i]=math.log(x,10)
    else:
        for i in idf_dict:
            x=len(folders)/idf_dict[i]
            idf[i]=math.log(x,10)
    return idf


# In[24]:


print("input query :")
query=input()
print("enter the value of k:")
k=int(input())
final_idf={}
print("which variation of idf do you want: ")
var=int(input())
final_idf=idf_variation(var)
output=cosine_similarity(query,k)
print(output)


# In[ ]:





# In[ ]:





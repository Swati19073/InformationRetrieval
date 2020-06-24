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
# import inflect


# In[2]:


#doc contains all the document list
#filelist is a list of list where each list conatins one file
#retrieving all the files inside folder 20_newsgroups


import os
folders=os.listdir('20_newsgroups/')
filelist=[]
doc=[]
# count=0
for directory in folders:
    files=os.listdir('20_newsgroups/'+directory)
    for i in files:
        doc.append(i)
    for file in files:
        f=open('20_newsgroups/'+directory+"/"+file)
        myfile=f.read()
        myfile=myfile.lower()
        myfile=myfile.translate(str.maketrans(" "," ",string.punctuation))
#         myfile=re.sub(r"\d+","",myfile)
        filelist.append(myfile)


# In[16]:


s="sci.med"
f=[]
for d in folders:
    if d ==s:
        files=os.listdir('20_newsgroups/'+d)
        for i in files:
            f.append(i)
print(len(f))      


# In[4]:


#tf_dict contains each doc as key and a dictionary of (terms, tf) values for each document as value
#idf_dict contains each term as key and ifd value as value
#vocab1 contains all the vocabulary terms
#finallist is list of list where each list contains lemmatized tokens corresponding to each file



import pickle
from num2words import num2words
lem=WordNetLemmatizer()
final_list=[]
idf_dict={}
docs_dictionary={}
# tf_dict={}
tf_dict={}
vocab1={}
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
        if i1 not in idf_dict:
            idf_dict[i1]=1
        else:
            idf_dict[i1]=idf_dict[i1]+1 
    temp_dict={}
    for i2 in temp1:
        if i2 not in vocab1:
            vocab1[i2]=1
        if i2 not in temp_dict:
            temp_dict[i2]=temp1.count(i2)
    
    tf_dict[doc[index]]=temp_dict   
    index+=1
    final_list.append(mylist)


# In[5]:


#final index is the dictionary where key is document name and value corresponds to a dictionary of size vocabulary with term
#and tf-idf values
import math
final_index={}
for i in tf_dict:
    temp={}
    for j in vocab1:
        if j in tf_dict[i]:
            x=len(doc)/(idf_dict[j] +1)
            idf=math.log(x,10)
            temp[j]=math.log(1+tf_dict[i][j])*idf
        else:
            temp[j]=0
    final_index[i]=temp


# In[7]:


print("vector corresponding to document 58044 is:\n",final_index['58044'])


# In[8]:


#function for getting top k tuples from a list of tuples
def get_k_tuples(list2,k):
    li2=[]
    li2=sorted(list2, key=lambda t: t[1], reverse=True)[:k]
    return li2


#function to sort a dictionary on the basis of values
def sort_dictionary(list1):
    li=[]
    li=sorted(list1.items(), key = lambda kv:(kv[1], kv[0]))
    return li


# In[10]:



#function to find cosine similarity between query vector and the documents and retrieving top k documents

def query_cosine(query,k):
    final_cosine={}
    query_vector={}
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
        if x not in idf_dict:
            idf_dict[x]=0
    #creating query vector
    for term in vocab1:
        if term in input_txt:
            x=len(doc)/(idf_dict[term] +1)
            idf=math.log(x,10)
            val=math.log(1+input_txt.count(term))*idf
            query_vector[term]=val
        else:
            query_vector[term]=0
    for d in final_index:
        mul_list=[]
        q_sq_list=[]
        d_sq_list=[]
        for c in query_vector:
            mul=final_index[d][c]*query_vector[c]
            mul_list.append(mul)
            qs=query_vector[c]*query_vector[c]
            q_sq_list.append(qs)
            ds=final_index[d][c]*final_index[d][c]
            d_sq_list.append(ds)
        s1=0
        s2=0
        s3=0 
        for s in range(len(mul_list)):
            s1=s1+mul_list[s]
            s2=s2+q_sq_list[s]
            s3=s3+d_sq_list[s]
        s2=math.sqrt(s2)
        s3=math.sqrt(s3)
        if (s2*s3)!=0:
            fin=s1/(s2*s3)
#             print(fin)
            final_cosine[d]=fin
    list1=[]    
    list1=sort_dictionary(final_cosine)
    output=[]
    output=get_k_tuples(list1,k)
    return output  
    


# In[11]:


query=input("enter the query \n")
k=int(input("enter the value of k \n"))
ans=query_cosine(query,k)

print("Top ",k, " documents are: \n", ans)


# In[ ]:





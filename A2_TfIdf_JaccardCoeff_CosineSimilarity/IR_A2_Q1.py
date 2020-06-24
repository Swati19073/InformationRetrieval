#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

folders=os.listdir('stories/')


# In[5]:


# print(folders)
import pickle
from num2words import num2words


# In[85]:


# li=['11122','33']
# x=[]
# for i in range(len(li)):
#     if(li[i].isdecimal()==True):
#         x.append(num2words(li[i]))
# print(x)
# u=num2words(142.9)
# print(u)


# In[21]:


import nltk
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
import string
import re
import inflect


# In[22]:


filelist=[]
#contains list of list where every index contains one document
for files in folders:
    f=open('stories/'+files,encoding='latin1')
    myfile=f.read()
    myfile=myfile.lower()
    myfile=myfile.translate(str.maketrans("","",string.punctuation))
    filelist.append(myfile)
   


# In[23]:


# print(filelist[467])


# In[53]:


lem=WordNetLemmatizer()
final_list=[]
docs_dictionary={}
idf_dict={}
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english')) 
for i in range(len(filelist)):
    count=0
    word_tokens = word_tokenize(filelist[i])
    temp = []
    for w in word_tokens:
        if w not in stop_words: 
            temp.append(w)
    temp1=[]
    temp2=[]
    for token in temp:
        if(token.isdecimal()==True):
            token=num2words(token)
        word=lem.lemmatize(token)            
        temp1.append(word) 
        temp2.append(word)
    temp2=set(temp2)
    for i1 in temp2:
        if i1 not in idf_dict.keys():
            idf_dict[i1]=1
        else:
            idf_dict[i1]=idf_dict[i1]+1
        
    docs_dictionary[folders[i]]=temp1


# In[54]:


print(idf_dict['flower'])


# In[55]:


#computing idf using logbase10(N/count_of_each_term)
import math
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


# In[ ]:





# In[56]:


def get_k_tuples(list2,k):
    li2=[]
    li2=sorted(list2, key=lambda t: t[1], reverse=True)[:k]
    return li2


def sort_dictionary(list1):
    li=[]
    li=sorted(list1.items(), key = lambda kv:(kv[1], kv[0]))
    return li


# In[86]:


#jacard coefficient

def jacard_coeff(query,k):
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
    op_dict={}    
    for val in docs_dictionary:
        union=set(docs_dictionary[val])|set(input_txt)
        intersection=set(docs_dictionary[val])&set(input_txt)
        op_dict[val]=len(intersection)/len(union)
    list1=[]    
    list1=sort_dictionary(op_dict)
    output=[]
    output=get_k_tuples(list1,k)
    return output


# In[58]:


# print(docs_dictionary["quarter.c4"])


# In[87]:


def tf_idf(query,k):
    final_tfidf={}
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
        tlist=[]
        for word in input_txt:
            if(var1==1):
                c=docs_dictionary[d].count(word)
                tf=c/len(docs_dictionary[d])
            elif var1==2:
                 #raw count
                tf=docs_dictionary[d].count(word)
            else:
                tf=math.log(1+ (docs_dictionary[d].count(word)),10)
            idf1=final_idf[word]
            tf_idf=tf*idf1
            tlist.append(tf_idf)
        sum1=0
        for i in range(len(tlist)):
            sum1=sum1+tlist[i]
        final_tfidf[d]=sum1
    list1=[]    
    list1=sort_dictionary(final_tfidf)
    output=[]
    output=get_k_tuples(list1,k)
    return output       
    


# In[60]:


print(docs_dictionary["dakota.txt"])
# print("which variation of idf do you want: ")
# var=int(input())
# idf_val=idf_variation(var)
# print(idf_val['flower'])


# In[ ]:





# In[88]:



title_dict={}

newfile=open("C:/Users/user/AppData/Local/Programs/Python/Python37-32/Scripts/stories/index.txt","r+")
tstr=newfile.readlines()
index=0
while(index<len(tstr)-1):
    split_title=tstr[index].lower().split("\t")
    w_tokens = word_tokenize(tstr[index+1].lower())
#     print(w_tokens)
    temp_=[]
    for w in w_tokens:
        if w not in stop_words: 
            temp_.append(w)
    temp1_=[]
    temp2_=[]
    for token in temp_:
        if(token.isdecimal()==True):
            token=num2words(token)
        word=lem.lemmatize(token)            
        temp1_.append(word)  
    title_dict[split_title[0]]=temp1_ 
    index=index+2
    
print(len(title_dict)) 
    
    
    #idf=idf+idf*0.7
    
        


# In[89]:


def tf_idf_with_title(query,k):
    final_tfidf={}
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
        tlist=[]
        for word in input_txt:
                if word in title_dict:
                    if var1==1:
                        c=docs_dictionary[d].count(word)
                        tf=c/len(docs_dictionary[d])
                    elif var1==2:
                        tf=docs_dictionary[d].count(word)
                    else:
                        tf=math.log(1+ (docs_dictionary[d].count(word)),10)
                    idf1=final_idf[word]
                    tf_idf=tf*idf1
                    tlist.append(tf_idf)
                else:
                    if var1==1:
                        c=docs_dictionary[d].count(word)
                        tf=c/len(docs_dictionary[d])
                    elif var1==2:
                        tf=docs_dictionary[d].count(word)
                    else:
                        tf=math.log(1+ (docs_dictionary[d].count(word)),10)
                    idf1=final_idf[word]
                    tf_idf=tf*idf1*0.7
                    tlist.append(tf_idf)
        sum1=0
        for i in range(len(tlist)):
            sum1=sum1+tlist[i]
        final_tfidf[d]=sum1
    list1=[]    
    list1=sort_dictionary(final_tfidf)
    output=[]
    output=get_k_tuples(list1,k)
    return output  


# In[ ]:





# In[90]:


#cosine
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
        s2=math.sqrt(s2)
        s3=math.sqrt(s3)
        if (s2*s3)!=0:
            fin=s1/(s2*s3)
            final_cosine[d]=fin
    list1=[]    
    list1=sort_dictionary(final_cosine)
    output=[]
    output=get_k_tuples(list1,k)
    return output    
            
            
            
        
        
    


# In[65]:


# print("input query :")
# query=input()
# print("enter the value of k:")
# k=int(input())
# final_idf={}
# # print("which variation of idf do you want: ")
# # var=int(input())
# final_idf=idf_variation(var)
# output=cosine_similarity(query,k)
# print(output)


# In[91]:


def cosine_similarity_with_title(query,k):
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
            tf_=c_/len(input_txt)
            idf1_=final_idf[word]
            tf_idf_=tf_*idf1_
            query_tfidf.append(tf_idf_)
            c=docs_dictionary[d].count(word)
            tf=c/len(docs_dictionary[d])
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
        s2=math.sqrt(s2)
        s3=math.sqrt(s3)
        if (s2*s3)!=0:
            fin=s1/(s2*s3)
            if word in title_dict:
                final_cosine[d]=fin
            else:
                final_cosine[d]=fin*0.7
    list1=[]    
    list1=sort_dictionary(final_cosine)
    output=[]
    output=get_k_tuples(list1,k)
    return output    
            
            
            
        
        
    


# In[67]:


# print("input query :")
# query=input()
# print("enter the value of k:")
# k=int(input())
# final_idf={}
# print("which variation of idf do you want: ")
# var=int(input())
# final_idf=idf_variation(var)
# output=cosine_similarity_with_title(query,k)
# print(output)


# In[102]:


print("input query :")
query=input()
print("enter the value of k:")
k=int(input())
final_idf={}
print("which variation of idf do you want: ")
var=int(input())
print("which variation of tf do you want: ")
var1=int(input())
final_idf=idf_variation(var)
# print(idf_val['flower'])
# out=tf_idf_with_title(query,k)
# final_output=jacard_coeff(query,k)
# print(final_output)   
# print(out)
# output=cosine_similarity_with_title(query,k)
# print(output)
# output=tf_idf(query,k)
# print(output)
output=tf_idf(query,k)
print(output)


# In[ ]:





# In[ ]:





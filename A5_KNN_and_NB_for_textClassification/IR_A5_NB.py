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


# In[2]:


#filelist is a dictionary where key is filename and value is lemmatized token list corresponding to that document
#classes contains all the classes names
#doc contains all the file names
#class_file is a dictionary which contains class names as key and list of all the files corresponding to that class as value
#file_vocab_dict contains filename as key and content of that file as value
import os
folders=os.listdir('20_newsgroups/')
classes=[]
filelist=[]
class_file={}
doc=[]
file_vocab_dict={}
# count=0
for directory in folders:
    if directory=='comp.graphics' or directory=='sci.med' or directory =='talk.politics.misc' or directory=='rec.sport.hockey' or directory=='sci.space':
        classes.append(directory)
        files=os.listdir('20_newsgroups/'+directory)
        files=files[1:]
        temp={}
        for i in files:
            if i not in temp:
                temp[i]=1
            doc.append(i)
        class_file[directory]=temp
        for file in files:
            f=open('20_newsgroups/'+directory+"/"+file)
            myfile=f.read()
            myfile=myfile.lower()
            myfile=myfile.translate(str.maketrans(" "," ",string.punctuation))
#         myfile=re.sub(r"\d+","",myfile)
            if file not in file_vocab_dict.keys():
                file_vocab_dict[file]=myfile
            filelist.append(myfile)


# In[190]:


import random
import sklearn
from sklearn.model_selection import train_test_split
shuffled_docs=[]
shuffled_docs=doc
random.shuffle(shuffled_docs)
print("Enter the percent of documents you want in training set: ")
amt=int(input())
test_amt=amt/100
test1, train1 = sklearn.model_selection.train_test_split(shuffled_docs,test_size=test_amt)
# print(len(test))
# print(len(train))
# print(test)
train={}
test={}
for i in train1:
    if i not in train:
        train[i]=1
for i in test1:
    if i not in test:
        test[i]=1
        


# In[191]:


print("Number of documents in training set is: ",len(train1))
print("Number of documents in testing set is: ",len(test1))


# In[192]:


# print(test)
#class_train contains class wise training document
t1={}
t2={}
t3={}
t4={}
t5={}
class_train={}

for i in train1:
    if i in class_file['comp.graphics']:
        if i not in t1:
            t1[i]=1
    elif i in class_file['sci.med']:
        if i not in t2:
            t2[i]=1
    elif i in class_file['talk.politics.misc']:
        if i not in t3:
            t3[i]=1
    elif i in class_file['rec.sport.hockey']:
        if i not in t4:
            t4[i]=1
    elif i in class_file['sci.space']:
        if i not in t5:
            t5[i]=1
            
        
class_train['comp.graphics']=t1
class_train['sci.med']=t2
class_train['talk.politics.misc']=t3
class_train['rec.sport.hockey']=t4
class_train['sci.space']=t5


# In[193]:


# print("Training documents belonging to class rec.sport.hockey are:\n",class_train['rec.sport.hockey'])


# In[194]:


# print(test)
#class_test contains class wise testing documents
tt1={}
tt2={}
tt3={}
tt4={}
tt5={}
class_test={}

for i in test1:
    if i in class_file['comp.graphics']:
        if i not in tt1:
            tt1[i]=1
    elif i in class_file['sci.med']:
        if i not in tt2:
            tt2[i]=1
    elif i in class_file['talk.politics.misc']:
        if i not in tt3:
            tt3[i]=1
    elif i in class_file['rec.sport.hockey']:
        if i not in tt4:
            tt4[i]=1
    elif i in class_file['sci.space']:
        if i not in tt5:
            tt5[i]=1
            
        
class_test['comp.graphics']=tt1
class_test['sci.med']=tt2
class_test['talk.politics.misc']=tt3
class_test['rec.sport.hockey']=tt4
class_test['sci.space']=tt5


# In[195]:


print(len(class_train['sci.med'])+len(class_train['comp.graphics'])+len(class_train['sci.space'])+len(class_train['talk.politics.misc'])+len(class_train['rec.sport.hockey']))


# In[196]:


# print(class_test['sci.space'])


# In[197]:


#vocab1 contains all the vocabulary terms
#vocab contains all the unique vocaulary terms
#final_class_test contains all the training docs and their lemmatized tokens
#final_list contains each doc as key and lemmatized tokens as values
#tf_dict contains each doc as key and a dictionary of term,tf pair for each term corresponding to each doc as value
import pickle
from num2words import num2words
lem=WordNetLemmatizer()
final_list={}
tf_dict={}
vocab1=[]
class1=[]
class2=[]
class3=[]
class4=[]
class5=[]
final_class_test={}
from nltk.tokenize import word_tokenize   
stop_words = set(stopwords.words('english')) 
# ind=0
# index=0
for i in file_vocab_dict:
    t=file_vocab_dict[i]
    word_tokens = word_tokenize(t)
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
    if i in class_train['comp.graphics']:
        for w in temp1:
            class1.append(w)
    elif i in class_train['sci.med']:
        for w in temp1:
            class2.append(w)
    elif i in class_train['talk.politics.misc']:
        for w in temp1:
            class3.append(w)
    elif i in class_train['rec.sport.hockey']:
        for w in temp1:
            class4.append(w)
    elif i in class_train['sci.space']:
        for w in temp1:
            class5.append(w)
    elif i in class_test['comp.graphics']:
        t=[]
        for w in temp1:
            t.append(w)
        final_class_test[i]=t
    elif i in class_test['sci.med']:
        tt=[]
        for w in temp1:
            tt.append(w)
        final_class_test[i]=tt
    elif i in class_test['talk.politics.misc']:
        ttt=[]
        for w in temp1:
            ttt.append(w)
        final_class_test[i]=ttt
    elif i in class_test['rec.sport.hockey']:
        tttt=[]
        for w in temp1:
            tttt.append(w)
        final_class_test[i]=tttt
    elif i in class_test['sci.space']:
        ttttt=[]
        for w in temp1:
            ttttt.append(w)
        final_class_test[i]=ttttt
#         temp2.append(word)
    temp_dict={}
    for i2 in temp1:
        vocab1.append(i2)
        if i2 not in temp_dict:
            temp_dict[i2]=temp1.count(i2)
    
    tf_dict[i]=temp_dict   
#     index+=1
    final_list[i]=temp1


# In[198]:


# print(class_file['comp.graphics'])
print(len(final_class_test))
# print(final_list['178913'])


# In[199]:


# print(tf_dict["178913"])
#class_final is a dictionary where each class is a key and list of all the terms of that class acts as value
c1={}
c2={}
c3={}
c4={}
c5={}
train_vocab=[]
for i in class1:
    train_vocab.append(i)
    if i not in c1:
        c1[i]=1
for i in class2:
    train_vocab.append(i)
    if i not in c2:
        c2[i]=1
for i in class3:
    train_vocab.append(i)
    if i not in c3:
        c3[i]=1
for i in class4:
    train_vocab.append(i)
    if i not in c4:
        c4[i]=1
for i in class5:
    train_vocab.append(i)
    if i not in c5:
        c5[i]=1
        
class_final={}
class_final['comp.graphics']=c1
class_final['sci.med']=c2
class_final['talk.politics.misc']=c3
class_final['rec.sport.hockey']=c4
class_final['sci.space']=c5

class_with_all_terms={}
class_with_all_terms['comp.graphics']=class1
class_with_all_terms['sci.med']=class2
class_with_all_terms['talk.politics.misc']=class3
class_with_all_terms['rec.sport.hockey']=class4
class_with_all_terms['sci.space']=class5


# In[200]:


# print(len(class_with_all_terms['comp.graphics']))
print("Training tokens for class sci.med are: \n",class_with_all_terms['sci.med'])


# In[201]:


# print(class_final['sci.med'])
train_vocab_unique={}
for i in train_vocab:
    if i not in train_vocab_unique:
        train_vocab_unique[i]=1
print(len(train_vocab))
print(len(train_vocab_unique))


# In[202]:



vocab={}
for i in vocab1:
    if i not in vocab:
        vocab[i]=1
# print(vocab)
print(len(vocab1))
# vocab=(set(vocab1))
print(len(vocab))


# In[203]:


tf1={}
tf2={}
tf3={}
tf4={}
tf5={}
final_train_tf={}
tft1={}
tft2={}
tft3={}
tft4={}
tft5={}
final_test_tf={}

for v in train_vocab_unique:
    s1=0
    s2=0
    s3=0
    s4=0
    s5=0
    st1=0
    st2=0
    st3=0
    st4=0
    st5=0
    for f in tf_dict:

        if f in class_train['comp.graphics']:
            if v in tf_dict[f]:
                s1+=tf_dict[f][v]
        elif f in class_train['sci.med']:
            if v in tf_dict[f]:
                s2+=tf_dict[f][v]
        elif f in class_train['talk.politics.misc']:
            if v in tf_dict[f]:
                s3+=tf_dict[f][v]
        elif f in class_train['rec.sport.hockey']:
            if v in tf_dict[f]:
                s4+=tf_dict[f][v]
        elif f in class_train['sci.space']:
            if v in tf_dict[f]:
                s5+=tf_dict[f][v]
        elif f in class_test['sci.med']:
            if v in tf_dict[f]:
                st2+=tf_dict[f][v]
        elif f in class_test['comp.graphics']:
            if v in tf_dict[f]:
                st1+=tf_dict[f][v]
        elif f in class_test['talk.politics.misc']:
            if v in tf_dict[f]:
                st3+=tf_dict[f][v]
        elif f in class_test['rec.sport.hockey']:
            if v in tf_dict[f]:
                st4+=tf_dict[f][v]
        elif f in class_test['sci.space']:
            if v in tf_dict[f]:
                st5+=tf_dict[f][v]
    tf1[v]=s1
    tf2[v]=s2
    tf3[v]=s3
    tf4[v]=s4
    tf5[v]=s5
    tft1[v]=st1
    tft2[v]=st2
    tft3[v]=st3
    tft4[v]=st4
    tft5[v]=st5
        
                


# In[204]:


#final_train_tf stores classes as keys and dictionary of term,term-frequency pair for each class having training documents as values
#final_test_tf stores classes as keys and dictionary of term,term-frequency pair for each class having testing documents as values
final_train_tf['comp.graphics']=tf1
final_train_tf['sci.med']=tf2
final_train_tf['talk.politics.misc']=tf3
final_train_tf['rec.sport.hockey']=tf4
final_train_tf['sci.space']=tf5
final_test_tf['comp.graphics']=tft1
final_test_tf['sci.med']=tft2
final_test_tf['talk.politics.misc']=tft3
final_test_tf['rec.sport.hockey']=tft4
final_test_tf['sci.space']=tft5


# In[205]:


# print("Term frequencies for each term in class talk.politics.misc are: \n",final_train_tf['talk.politics.misc'])


# In[206]:


z=0
#idf_dict contains the idf value for each vocab term
cc1={}
cc2={}
cc3={}
cc4={}
cc5={}
idf_dict={}
for w in train_vocab_unique:
    count=0
    if w in class_final['comp.graphics']:
        count+=1
    if w in class_final['sci.med']:
        count+=1
    if w in class_final['talk.politics.misc']:
        count+=1
    if w in class_final['rec.sport.hockey']:
        count+=1
    if w in class_final['sci.space']:
        count+=1
    idf_dict[w]=count
    print(z)
    c1=0
    c2=0
    c3=0
    c4=0
    c5=0
    for f in tf_dict:
        if f in class_train['comp.graphics'] and w in tf_dict[f]:
            c1+=1
        elif f in class_train['sci.med'] and w in tf_dict[f]:
            c2+=1
        elif f in class_train['talk.politics.misc'] and w in tf_dict[f]:
            c3+=1
        elif f in class_train['rec.sport.hockey'] and w in tf_dict[f]:
            c4+=1
        elif f in class_train['sci.space'] and w in tf_dict[f]:
            c5+=1
    cc1[w]=c1
    cc2[w]=c2
    cc3[w]=c3
    cc4[w]=c4
    cc5[w]=c5
    z+=1

        


# In[207]:


# print(idf_dict)


# In[208]:


#dictionary for calculatig mi. This dict contains class as key and a dict of (term, number of docs in which they appear in that particular class) as values
dict_for_mi={}
dict_for_mi['comp.graphics']=cc1
dict_for_mi['sci.med']=cc2
dict_for_mi['talk.politics.misc']=cc3
dict_for_mi['rec.sport.hockey']=cc4
dict_for_mi['sci.space']=cc5


# In[209]:


print(dict_for_mi['rec.sport.hockey'])


# In[210]:


import math
def mutual_info(n11, n01, n10, n00,N):
    a=0
    b=0
    c=0
    d=0
    if ((n11+n10)+(n01+n11))>0:
        t1=((n11*N)/((n11+n10)+(n01+n11)))
        if(t1>0):
            a=(n11/N)*math.log(t1,2)
    if ((n00+n01)*(n11+n01))>0:
        t2=((n01*N)/((n00+n01)*(n11+n01)))
        if t2>0:
            b=(n01/N)*math.log(t2,2)
    if ((n11+n10)*(n10+n00))>0:
        t3=((n10*N)/((n11+n10)*(n10+n00)))
        if t3>0:
            c=(n10/N)*math.log(t3,2)
    if ((n00+n01)*(n10+n00)) >0:
        t4=((n00*N)/((n00+n01)*(n10+n00)))
        if t4>0:
            d=(n00/N)*math.log(t4,2)
    return (a+b+c+d)


# In[211]:


#function to sort a dictionary on the basis of values
def sort_dictionary(list1):
    li=[]
    li=sorted(list1.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)
    return li

#function for getting top k tuples from a list of tuples
def get_k_tuples(list2,k):
    li2=[]
    li2=sorted(list2, key=lambda t: t[1], reverse=True)[:k]
    return li2


# In[212]:


final_mi={}
N=len(train)
def mi(k):
    mi1={}

    for i in dict_for_mi['comp.graphics']:
#     print(i)
        n11=dict_for_mi['comp.graphics'][i] #word is present and doc belongs to class comp.graphics
#         print(n11)
        n01=len(class_train['comp.graphics'])-n11 #word not present but doc belongs to comp.graphics
#         print(n01)
        #n10 is word present but docs not in class comp.graphics
        n10=dict_for_mi['sci.med'][i]+dict_for_mi['sci.space'][i]+dict_for_mi['talk.politics.misc'][i]+dict_for_mi['rec.sport.hockey'][i]
        n00=len(class_train['sci.med'])+len(class_train['sci.space'])+len(class_train['talk.politics.misc'])+len(class_train['rec.sport.hockey'])-n10
        mi=mutual_info(n11,n01,n10,n00,N)
        mi1[i]=mi
    new_mi1=[]
    new_mi1=sort_dictionary(mi1)
    output1=[]
    k1=int((k/100)*len(dict_for_mi['comp.graphics']))
    output1=get_k_tuples(new_mi1,k1)
    mi2={}
    for i in dict_for_mi['sci.med']:
#     print(i)
        n11=dict_for_mi['sci.med'][i]
        n01=len(class_train['sci.med'])-n11
        n10=dict_for_mi['comp.graphics'][i]+dict_for_mi['sci.space'][i]+dict_for_mi['talk.politics.misc'][i]+dict_for_mi['rec.sport.hockey'][i]
        n00=len(class_train['comp.graphics'])+len(class_train['sci.space'])+len(class_train['talk.politics.misc'])+len(class_train['rec.sport.hockey'])-n10
        mi=mutual_info(n11,n01,n10,n00,N)
        mi2[i]=mi
    
    new_mi2=[]
    new_mi2=sort_dictionary(mi1)
    k2=int((k/100)*len(dict_for_mi['sci.med']))
    output2=[]
    output2=get_k_tuples(new_mi2,k2)
    
    
    mi3={}
    for i in dict_for_mi['talk.politics.misc']:
#     print(i)
        n11=dict_for_mi['talk.politics.misc'][i]
        n01=len(class_train['talk.politics.misc'])-n11
        n10=dict_for_mi['comp.graphics'][i]+dict_for_mi['sci.space'][i]+dict_for_mi['sci.med'][i]+dict_for_mi['rec.sport.hockey'][i]
        n00=len(class_train['comp.graphics'])+len(class_train['sci.space'])+len(class_train['sci.med'])+len(class_train['rec.sport.hockey'])-n10
        mi=mutual_info(n11,n01,n10,n00,N)
        mi3[i]=mi
    new_mi3=[]
    new_mi3=sort_dictionary(mi3)
    output3=[]
    k3=int((k/100)*len(dict_for_mi['talk.politics.misc']))
    output3=get_k_tuples(new_mi3,k3)
    mi4={}
    for i in dict_for_mi['rec.sport.hockey']:
#     print(i)
        n11=dict_for_mi['rec.sport.hockey'][i]
        n01=len(class_train['rec.sport.hockey'])-n11
        n10=dict_for_mi['comp.graphics'][i]+dict_for_mi['sci.space'][i]+dict_for_mi['sci.med'][i]+dict_for_mi['talk.politics.misc'][i]
        n00=len(class_train['comp.graphics'])+len(class_train['sci.space'])+len(class_train['sci.med'])+len(class_train['talk.politics.misc'])-n10
        mi=mutual_info(n11,n01,n10,n00,N)
        mi4[i]=mi
    new_mi4=[]
    new_mi4=sort_dictionary(mi4)
    output4=[]
    k4=int((k/100)*len(dict_for_mi['rec.sport.hockey']))
    output4=get_k_tuples(new_mi4,k4)
    mi5={}
    for i in dict_for_mi['sci.space']:
#         print(i)
        n11=dict_for_mi['sci.space'][i]
        n01=len(class_train['sci.space'])-n11
        n10=dict_for_mi['comp.graphics'][i]+dict_for_mi['rec.sport.hockey'][i]+dict_for_mi['sci.med'][i]+dict_for_mi['talk.politics.misc'][i]
        n00=len(class_train['comp.graphics'])+len(class_train['rec.sport.hockey'])+len(class_train['sci.med'])+len(class_train['talk.politics.misc'])-n10
        mi=mutual_info(n11,n01,n10,n00,N)
        mi5[i]=mi
    new_mi5=[]
    new_mi5=sort_dictionary(mi5)
    output5=[]
    k5=int((k/100)*len(dict_for_mi['sci.space']))
    output5=get_k_tuples(new_mi5,k5)
    return output1,output2,output3,output4,output5


# In[213]:


def tf_idf(k):
    temp1={}
    for x in final_train_tf['comp.graphics']:
        y=math.log(1+final_train_tf['comp.graphics'][x])*idf_dict[x]
        temp1[x]=y
    new_mi1=[]
    k1=int((k/100)*len(final_train_tf['comp.graphics']))
#     print(k1)
    new_mi1=sort_dictionary(temp1)
    output1=[]
    output1=get_k_tuples(new_mi1,k1)
    
    temp2={}
    for x in final_train_tf['sci.med']:
        y=math.log(1+final_train_tf['sci.med'][x])*idf_dict[x]
        temp2[x]=y
    new_mi2=[]
    new_mi2=sort_dictionary(temp2)
    output2=[]
    k2=int((k/100)*len(final_train_tf['sci.med']))
    output2=get_k_tuples(new_mi2,k2)

    temp3={}
    for x in final_train_tf['talk.politics.misc']:
        y=math.log(1+final_train_tf['talk.politics.misc'][x])*idf_dict[x]
        temp3[x]=y
    new_mi3=[]
    new_mi3=sort_dictionary(temp3)
    output3=[]
    k3=int((k/100)*len(final_train_tf['talk.politics.misc']))
    output3=get_k_tuples(new_mi3,k3)

    temp4={}
    for x in final_train_tf['rec.sport.hockey']:
        y=math.log(1+final_train_tf['rec.sport.hockey'][x])*idf_dict[x]
        temp4[x]=y
    new_mi4=[]
    new_mi4=sort_dictionary(temp4)
    output4=[]
    k4=int((k/100)*len(final_train_tf['rec.sport.hockey']))
    output4=get_k_tuples(new_mi4,k4)

    temp5={}
    for x in final_train_tf['sci.space']:
        y=math.log(1+final_train_tf['sci.space'][x])*idf_dict[x]
        temp5[x]=y
    new_mi5=[]
    new_mi5=sort_dictionary(temp5)
    output5=[]
    k5=int((k/100)*len(final_train_tf['sci.space']))
    output5=get_k_tuples(new_mi5,k5)
    return output1,output2,output3,output4,output5


# In[214]:


k=int(input("Enter the percentage of k to retrieve top k terms "))

o1,o2,o3,o4,o5=mi(k)
a1,a2,a3,a4,a5=tf_idf(k)
final_mi['comp.graphics']=o1
final_mi['sci.med']=o2

final_mi['talk.politics.misc']=o3
final_mi['rec.sport.hockey']=o4

final_mi['sci.space']=o5
tf_idf={}
tf_idf['comp.graphics']=a1
tf_idf['sci.med']=a2
tf_idf['talk.politics.misc']=a3
tf_idf['rec.sport.hockey']=a4
tf_idf['sci.space']=a5


# In[215]:



# print(tf_idf['sci.med'])
#tf_idf_final contains top k words with highest tf_idf values corresponding to each class
tf_idf_final={}
t1={}
for i in range(len(tf_idf['comp.graphics'])):
    
    if (tf_idf['comp.graphics'][i][0]) not in t1:
        t1[tf_idf['comp.graphics'][i][0]]=1
tf_idf_final['comp.graphics']=t1


t2={}
for i in range(len(tf_idf['sci.med'])):
    
    if (tf_idf['sci.med'][i][0]) not in t2:
        t2[tf_idf['sci.med'][i][0]]=1
tf_idf_final['sci.med']=t2

t3={}
for i in range(len(tf_idf['talk.politics.misc'])):
    
    if (tf_idf['talk.politics.misc'][i][0]) not in t3:
        t3[tf_idf['talk.politics.misc'][i][0]]=1
tf_idf_final['talk.politics.misc']=t3

t4={}
for i in range(len(tf_idf['rec.sport.hockey'])):
    
    if (tf_idf['rec.sport.hockey'][i][0]) not in t4:
        t4[tf_idf['rec.sport.hockey'][i][0]]=1
tf_idf_final['rec.sport.hockey']=t4

t5={}
for i in range(len(tf_idf['sci.space'])):
    
    if (tf_idf['sci.space'][i][0]) not in t5:
        t5[tf_idf['sci.space'][i][0]]=1
tf_idf_final['sci.space']=t5


# In[216]:


print(final_mi['comp.graphics'])


# In[217]:


#mi final contains top k words with highest tf_idf values corresponding to each class
mi_final={}
t11={}
for i in range(len(final_mi['comp.graphics'])):
    
    if (final_mi['comp.graphics'][i][0]) not in t11:
        t11[final_mi['comp.graphics'][i][0]]=1
mi_final['comp.graphics']=t11


t22={}
for i in range(len(final_mi['sci.med'])):
    
    if (final_mi['sci.med'][i][0]) not in t22:
        t22[final_mi['sci.med'][i][0]]=1
mi_final['sci.med']=t2

t33={}
for i in range(len(final_mi['talk.politics.misc'])):
    
    if (final_mi['talk.politics.misc'][i][0]) not in t33:
        t33[final_mi['talk.politics.misc'][i][0]]=1
mi_final['talk.politics.misc']=t33

t44={}
for i in range(len(final_mi['rec.sport.hockey'])):
    
    if (final_mi['rec.sport.hockey'][i][0]) not in t44:
        t44[final_mi['rec.sport.hockey'][i][0]]=1
mi_final['rec.sport.hockey']=t44

t55={}
for i in range(len(final_mi['sci.space'])):
    
    if (final_mi['sci.space'][i][0]) not in t55:
        t55[final_mi['sci.space'][i][0]]=1
mi_final['sci.space']=t55


# In[218]:


print("Top",k,"% terms using tf_idf for feature selection for comp.graphics are: \n ",tf_idf_final['comp.graphics'])


# In[219]:


print("Top",k,"% terms using tf_idf for feature selection for sci.med are: \n ",tf_idf_final['sci.med'])


# In[220]:


print("Top",k,"% terms using tf_idf for feature selection for talk.politics.misc are: \n ",tf_idf_final['talk.politics.misc'])


# In[221]:


print("Top",k,"% terms using tf_idf for feature selection for rec.sport.hockey are: \n ",tf_idf_final['rec.sport.hockey'])


# In[222]:


print("Top",k,"% terms using tf_idf for feature selection for sci.space are: \n ",tf_idf_final['sci.space'])


# In[223]:


print("Top",k,"% terms using MI for feature selection for rec.sport.hockey are: \n ",mi_final['rec.sport.hockey'])


# In[224]:


print("Top",k,"% terms using MI for feature selection for talk.politics.misc are: \n ",mi_final['talk.politics.misc'])


# In[225]:


print("Top",k,"% terms using MI for feature selection for comp.graphics are: \n ",mi_final['comp.graphics'])


# In[226]:


print("Top",k,"5 terms using MI for feature selection for sci.med are: \n ",mi_final['sci.med'])


# In[227]:


print("Top",k,"% terms using MI for feature selection for sci.space are: \n ",mi_final['sci.space'])


# In[228]:


prior_prob={}
prior_prob['comp.graphics']=len(class_train['comp.graphics'])/len(train)
prior_prob['sci.med']=len(class_train['sci.med'])/len(train)
prior_prob['talk.politics.misc']=len(class_train['talk.politics.misc'])/len(train)
prior_prob['rec.sport.hockey']=len(class_train['rec.sport.hockey'])/len(train)
prior_prob['sci.space']=len(class_train['sci.space'])/len(train)


# In[229]:


print("Prior probability class wise are: \n" ,prior_prob)


# In[230]:


print(class_final['comp.graphics'])


# In[231]:


#cond_prob is a dictionary which contains class name as keys and word,conditional probability of each word that belongs to that 
#particular class as value
#training in naive bayes
cond_prob={}
prob1={}
# for t in class_final['comp.graphics']:
for t in train_vocab_unique:
    tf=final_train_tf['comp.graphics'][t]
    V=len(class_with_all_terms['comp.graphics'])
    B=len(train_vocab_unique)
    prob=(tf+1)/(B+V)
    prob1[t]=prob
cond_prob['comp.graphics']=prob1
    
prob2={}
for t in train_vocab_unique:
    tf=final_train_tf['sci.med'][t]
    V=len(class_with_all_terms['sci.med'])
    B=len(train_vocab_unique)
    prob=(tf+1)/(B+V)
    prob2[t]=prob
cond_prob['sci.med']=prob2

prob3={}
for t in train_vocab_unique:
    tf=final_train_tf['talk.politics.misc'][t]
    V=len(class_with_all_terms['talk.politics.misc'])
    B=len(train_vocab_unique)
    prob=(tf+1)/(B+V)
    prob3[t]=prob
cond_prob['talk.politics.misc']=prob3

prob4={}
for t in train_vocab_unique:
    tf=final_train_tf['rec.sport.hockey'][t]
    V=len(class_with_all_terms['rec.sport.hockey'])
    B=len(train_vocab_unique)
    prob=(tf+1)/(B+V)
    prob4[t]=prob
cond_prob['rec.sport.hockey']=prob4

prob5={}
for t in train_vocab_unique:
    tf=final_train_tf['sci.space'][t]
    V=len(class_with_all_terms['sci.space'])
    B=len(train_vocab_unique)
    prob=(tf+1)/(B+V)
    prob5[t]=prob
cond_prob['sci.space']=prob5


# In[232]:


# print("Conditional probability for class comp.graphics are:\n" ,cond_prob['comp.graphics'])


# In[233]:


# print("Conditional probability for class sci.med are:\n" ,cond_prob['sci.med'])


# In[234]:


# print("Conditional probability for class talk.politics.misc are:\n" ,cond_prob['talk.politics.misc'])


# In[235]:


# print("Conditional probability for class rec.sport.hockey are:\n" ,cond_prob['rec.sport.hockey'])


# In[236]:


# print("Conditional probability for class sci.space are:\n" ,cond_prob['sci.space'])


# In[237]:


final_ans={}
for t in final_class_test:
#     print(t)
    mul1=0
    mul2=0
    mul3=0
    mul4=0
    mul5=0
    for term in final_class_test[t]:
        if term in tf_idf_final['comp.graphics']:
            pro=math.log(cond_prob['comp.graphics'][term])
            mul1=mul1+pro
#             if mul1==0:
#                 print(term) 
        else:
            V=len(class_with_all_terms['comp.graphics'])
            B=len(train_vocab_unique)
            pro=math.log(1/(V+B))
            mul1=mul1+pro
            
#         print(mul1)
        if term in tf_idf_final['sci.med']:
            pro=math.log(cond_prob['sci.med'][term])
            mul2=mul2+pro
        else:
            V=len(class_with_all_terms['sci.med'])
            B=len(train_vocab_unique)
            pro=math.log(1/(V+B))
            mul2=mul2+pro 
        if term in tf_idf_final['talk.politics.misc']:
            pro=math.log(cond_prob['talk.politics.misc'][term])
            mul3=mul3+pro
        else:
            V=len(class_with_all_terms['talk.politics.misc'])
            B=len(train_vocab_unique)
            pro=math.log(1/(V+B))
            mul3=mul3+pro 
        if term in tf_idf_final['rec.sport.hockey']:
            pro=math.log(cond_prob['rec.sport.hockey'][term])
            mul4=mul4+pro
        else:
            V=len(class_with_all_terms['rec.sport.hockey'])
            B=len(train_vocab_unique)
            pro=math.log(1/(V+B))
            mul4=mul4+pro 
        if term in tf_idf_final['sci.space']:
            pro=math.log(cond_prob['sci.space'][term])
            mul5=mul5+pro
        else:
            V=len(class_with_all_terms['sci.space'])
            B=len(train_vocab_unique)
            pro=math.log(1/(V+B))
            mul5=mul5+pro
        
    rank={}
#     print(mul1)
    rank['comp.graphics']=math.log(prior_prob['comp.graphics'])+mul1
#     print(rank['comp.graphics'])
    rank['sci.med']=math.log(prior_prob['sci.med'])+mul2
#     print(rank['sci.med'])
    rank['talk.politics.misc']=math.log(prior_prob['talk.politics.misc'])+mul3
    rank['rec.sport.hockey']=math.log(prior_prob['rec.sport.hockey'])+mul4
    rank['sci.space']=math.log(prior_prob['sci.space'])+mul5
#     print(rank['sci.space'])
#     print(rank)
    final_rank=[]
    final_rank=sort_dictionary(rank)
#     print(final_rank)
    final_ans[t]=final_rank[0]
    
    
    


# In[238]:



print(final_ans)


# In[239]:


final_ans_mi={}
for t in final_class_test:
#     print(t)
    mul1=0
    mul2=0
    mul3=0
    mul4=0
    mul5=0
    for term in final_class_test[t]:
        if term in mi_final['comp.graphics']:
            pro=math.log(cond_prob['comp.graphics'][term])
            mul1=mul1+pro
#             if mul1==0:
#                 print(term) 
        else:
            V=len(class_with_all_terms['comp.graphics'])
            B=len(train_vocab_unique)
            pro=math.log(1/(V+B))
            mul1=mul1+pro
            
        if term in mi_final['sci.med']:
            if term=='compgraphics':
                continue
            else:
                pro=math.log(cond_prob['sci.med'][term])
                mul2=mul2+pro
        else:
            V=len(class_with_all_terms['sci.med'])
            B=len(train_vocab_unique)
            pro=math.log(1/(V+B))
            mul2=mul2+pro 
        if term in mi_final['talk.politics.misc']:
            if term=="compgraphics":
                continue
            else:
                pro=math.log(cond_prob['talk.politics.misc'][term])
                mul3=mul3+pro
        else:
            V=len(class_with_all_terms['talk.politics.misc'])
            B=len(train_vocab_unique)
            pro=math.log(1/(V+B))
            mul3=mul3+pro 
        if term in mi_final['rec.sport.hockey']:
            if term=="compgraphics":
                continue
            else:
                pro=math.log(cond_prob['rec.sport.hockey'][term])
                mul4=mul4+pro
        else:
            V=len(class_with_all_terms['rec.sport.hockey'])
            B=len(train_vocab_unique)
            pro=math.log(1/(V+B))
            mul4=mul4+pro 
        if term in mi_final['sci.space']:
            if term=='compgraphics':
                continue
            else:
                pro=math.log(cond_prob['sci.space'][term])
                mul5=mul5+pro
        else:
            V=len(class_with_all_terms['sci.space'])
            B=len(train_vocab_unique)
            pro=math.log(1/(V+B))
            mul5=mul5+pro
        
    rank={}

    rank['comp.graphics']=math.log(prior_prob['comp.graphics'])+mul1
#     print(rank['comp.graphics'])
    rank['sci.med']=math.log(prior_prob['sci.med'])+mul2
#     print(rank['sci.med'])
    rank['talk.politics.misc']=math.log(prior_prob['talk.politics.misc'])+mul3
    rank['rec.sport.hockey']=math.log(prior_prob['rec.sport.hockey'])+mul4
    rank['sci.space']=math.log(prior_prob['sci.space'])+mul5
#     print(rank['sci.space'])
#     print(rank)
    final_rank=[]
    final_rank=sort_dictionary(rank)
#     print(final_rank)
    final_ans_mi[t]=final_rank[0]
    
    
    


# In[ ]:





# In[240]:


actual={}
#actual contains test file name as key and actual class to which the file belongs as values
#predicted_tfidf contains test file name as keys and predicted class names (using tf_idf as feature selection) corresponding to that file as values
#predicted_tfidf contains test file name as keys and predicted class names (using  as feature selection) corresponding to that file as value
predicted_tfidf={}
predicted_mi={}
for i in test:
    if i in class_test['comp.graphics']:
        actual[i]='comp.graphics'
    elif i in class_test['sci.med']:
        actual[i]='sci.med'
    elif i in class_test['talk.politics.misc']:
        actual[i]='talk.politics.misc'
    elif i in class_test['rec.sport.hockey']:
        actual[i]='rec.sport.hockey'
    else:
        actual[i]='sci.space'
    predicted_tfidf[i]=final_ans[i][0]
    predicted_mi[i]=final_ans_mi[i][0]
    


# In[ ]:





# In[241]:


print("Actual classes for test documents are: \n" ,actual)
actual1=[]
for i in actual:
    actual1.append(actual[i])
# print(actual1)


# In[242]:


print("Predicted classes for test files using mutual information as feature selection are: \n",predicted_mi)
pred_mi=[]
for i in predicted_mi:
    pred_mi.append(predicted_mi[i])
# print(pred_mi)


# In[243]:


print("Predicted classes for test files using tf_idf as feature selection are: \n",predicted_tfidf)
pred_tfidf=[]
for i in predicted_tfidf:
    pred_tfidf.append(predicted_tfidf[i])
# print(pred_tfidf)


# In[244]:


from sklearn.metrics import accuracy_score
print("accuracy after choosing tf_idf as feature selection in naive bayes is: ",accuracy_score(actual1, pred_tfidf))


# In[245]:


print("accuracy after choosing mutual information as feature selection in naive bayes is: ",accuracy_score(actual1, pred_mi))


# In[246]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
print("accuracy after choosing mutual information as feature selection in naive bayes is: ",accuracy_score(actual1, pred_mi))
import seaborn as sns
import seaborn as sns; sns.set(style="ticks", color_codes=True)
cm_mi = confusion_matrix(actual1, pred_mi)
print("Confusion matrix for MI is: \n",cm_mi)
fig, axes = plt.subplots(figsize=(6,6))
sns.heatmap(cm_mi,annot = True, linewidths=2,fmt=".0f",axes=axes)
plt.xlabel("Predicted values")
plt.ylabel("True values")
plt.title("Confusion matrix for MI")
plt.show()


# In[247]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns; sns.set(style="ticks", color_codes=True)
print("accuracy after choosing tf_idf as feature selection in naive bayes is: ",accuracy_score(actual1, pred_tfidf))
cm_tf_idf = confusion_matrix(actual1, pred_tfidf)
print("Confusion matrix for TF-IDF is: \n",cm_tf_idf)
fig, axes = plt.subplots(figsize=(6,6))
sns.heatmap(cm_tf_idf,annot = True, linewidths=2,fmt=".0f",axes=axes)
plt.xlabel("Predicted values")
plt.ylabel("True values")
plt.title("Confusion matrix for TF-IDF")
plt.show()


# In[127]:


# k_val=["10%", "40%", "70%" ]
# a_mi=[0.972972972972973,0.9769769769769769,0.9769769769769869]
# a_tf=[0.975975975975976,0.982982982982983, 0.982982982983983]
# plt.plot(k_val, a_mi,color="red")
# plt.xlabel("Percentage of terms taken for feature selection using MI as feature selection")
# plt.ylabel("Accuracies")
# plt.title("Accuracy vs top k% terms using MI as feature selection")


# In[128]:


# plt.plot(k_val, a_tf,color="red")
# plt.xlabel("Percentage of terms taken for feature selection using TF-IDF as feature selection")
# plt.ylabel("Accuracies")
# plt.title("Accuracy vs top k% terms using TF-IDf as feature selection")


# In[251]:


a=[0.9769769769769769,0.9766381842456608,0.9703644373247897]

p=["80%", "70%", "50%"]

plt.plot(p, a,color="blue")
plt.xlabel("Percentage of documents taken for training using top k% words from feature selection")
plt.ylabel("Accuracies")
plt.title("Accuracy vs different splitting ratio using MI as feature selection")


# In[249]:


a1=[0.982982982982983,0.9819759679572764,0.97436924309171]
p=["80%", "70%", "50%"]
plt.plot(p, a1,color="blue")
plt.xlabel("Percentage of documents taken for training using top k% words from feature selection")
plt.ylabel("Accuracies")
plt.title("Accuracy vs different splitting ratio using TF-IDF as feature selection")


# In[ ]:





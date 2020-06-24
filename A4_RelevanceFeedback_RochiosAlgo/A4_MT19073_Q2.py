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


import matplotlib.pyplot as plt


# In[3]:


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
    
#doc contains all the document list
#filelist is a list of list where each list conatins one file


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
tf_dict={}
vocab1={}
from nltk.tokenize import word_tokenize   
stop_words = set(stopwords.words('english')) 
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


# In[6]:


print(final_index['58044']['marriott'])


# In[7]:


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


# In[8]:



#function to find cosine similarity between query vector and the documents and retrieving top k documents

def cosine_similarity(query_vector1,k):
    final_cosine={}
    for d in final_index:
        mul_list=[]
        q_sq_list=[]
        d_sq_list=[]
        for c in query_vector1:
            mul=final_index[d][c]*query_vector1[c]
            mul_list.append(mul)
            qs=query_vector1[c]*query_vector1[c]
            q_sq_list.append(qs)
            ds=final_index[d][c]*final_index[d][c]
            d_sq_list.append(ds)
        s1=0
        s2=0
        s3=0
        s1=sum(mul_list)
        s2=sum(q_sq_list)
        s3=sum(d_sq_list)
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
    


# In[9]:


query_vector={}


#function for initial query preprocessing
def query_preprocessing(query):
    
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
    return query_vector
        
    


# In[10]:



#function for updating query vector and again finding cosine similarity between new query vector and the documents
#and retrieving top k documents
def rel_feedback(updated_q,rel_doc,non_rel):
    new_q={}
    rel_doc_len=len(rel_doc)
    non_rel_doc_len=len(non_rel)
    for term in updated_q:
        q0=updated_q[term]*1
        temp1=[]
        temp2=[]
        avg1=0
        avg2=0
        final=0
#         for d in doc:
#             if d in rel_doc:
#                 temp1.append(0.7*final_index[d][term])
#             else:
#                 temp2.append(0.25*final_index[d][term])
        for r in rel_doc:
            temp1.append(0.7*final_index[r][term])
        avg1=sum(temp1)/rel_doc_len
        for nr in non_rel:
            temp2.append(0.25*final_index[nr][term])
        avg2=sum(temp2)/non_rel_doc_len
        final=q0+avg1-avg2
        if final<0:
            final=0
            new_q[term]=final
        else:
            new_q[term]=final
    new_ans=cosine_similarity(new_q,k)
#     p,r,m=pre_recall(new_ans,rel_doc)
    return new_ans,new_q
        


# In[11]:


#function for calculating map,precision and recall

def pre_recall(list1,rel_doc):
    
    count=0
    rel_count=len(rel_doc)
    rel=0
    pre=[]
    recall=[]
    map1=[]
    for i in range(len(list1)):
        if list1[i][0] in rel_doc:
            rel+=1
            count+=1
            p=rel/count
            r=rel/rel_count
            pre.append(p)
            map1.append(p)
            recall.append(r)
        else:
            count+=1
            p=rel/count
            r=rel/rel_count
            pre.append(p)
            recall.append(r)
    fmap=0
    if len(map1)!=0:
        fmap=sum(map1)/len(map1)
    return pre, recall,fmap


# In[106]:


query=input("enter the intial query \n ")
k=int(input("enter the value of k \n"))
ans=query_preprocessing(query)

#final_ans1 conatins top k retrieved documents after initial query
final_ans1=cosine_similarity(ans,k)

print("top ", k, "documents after initial query are: \n", final_ans1)


# In[107]:


print("Initial query vector is : \n",ans)
#final_qv contains the query vectors after each iteration 
final_qv=[]
qv1=[]
for i in ans:
    qv1.append(ans[i])
final_qv.append(qv1)


# In[108]:


print("Enter the ground truth folder: \n")
gt=input()
ground_truth=[]
for d in folders:
    if d == gt:
        files=os.listdir('20_newsgroups/'+d)
        for i in files:
            ground_truth.append(i)
# print(len(ground_truth))    


#out_list1 contains the list of relevant documents after initial query
#out_list1 is used to calculate map,precision and recall for documents retrieved after initial output
out_list1=[]
for i in range(len(final_ans1)):
    x=final_ans1[i][0]
    out_list1.append(x)
# print(out_list1)

p,r,m=pre_recall(final_ans1,ground_truth)
print("precision after initial query are: ",p)
print("recall after initial query are: ",r)
print("MAP after initial query are: ",m)


# In[109]:



map_final1=[]
map_final1.append(m)


# In[110]:



plt.plot(r, p,color="blue")
plt.xlabel("Recall after initial query")
plt.ylabel("Precision after initial query ")
plt.title(" PR curve after initial query")


# In[111]:


print(out_list1)


# In[112]:


global_rel=[]


# In[113]:


print("Enter the p % of documents you want to mark as relevant \n")
p=int(input())
number_of_rel_docs=(p/100)*k

print("Enter the ", number_of_rel_docs, "documents you want to mark as relevant \n")
rel=input()
#rel_docs1 contains all the documents a user enters as relevant
rel_docs1=rel.split(",")

non_rel_docs1=[]

#non_rel_docs1 contains all the k-p remaining documents which are non relevant

non_rel_docs1=list(set(out_list1)-set(rel_docs1))

#final_docs1 conatins top k retrieved documents after 1st iteration
#new_query1 contains updated query after 1st iteration
final_docs1,new_query1=rel_feedback(query_vector,rel_docs1,non_rel_docs1)
print("docs after 1st iteration are: \n",final_docs1)


# In[114]:


global_rel=rel.split(",")


# In[115]:


p1,r1,m1=pre_recall(final_docs1,ground_truth)
print("precision after 1st iteration are: ",p1)
print("recall after 1st iteration are: ",r1)
print("MAP after 1st iteration are: ",m1)


# In[116]:



qv2=[]
for i in new_query1:
    qv2.append(new_query1[i])
# print(qv2)
final_qv.append(qv2)


# In[117]:


print("New query vector after 1st iteration is \n", new_query1)


# In[118]:


print(len(final_qv))
map_final1.append(m1)


# In[119]:


import matplotlib.pyplot as plt

plt.plot(r1, p1,color="blue")
plt.xlabel("Recall after 1st iteration")
plt.ylabel("Precision after 1st iteration")
plt.title(" PR curve after 1st iteration")


# In[120]:


out_list2=[]
for i in range(len(final_docs1)):
    x=final_docs1[i][0]
    out_list2.append(x)
    
print(out_list2)


# In[121]:


startlist1=[]
for i in range(len(final_docs1)):
    x=final_docs1[i][0]
    if x in rel_docs1:
        l=x + "*"
        startlist1.append(l)
    else:
        startlist1.append(x)
    
# print(out_list4)
print("docs after 1st iteration are: ",startlist1)


# In[122]:


print("Enter the ", number_of_rel_docs, "documents you want to mark as relevant \n")
rel1=input()

#rel_docs2 contains all the documents a user enters as relevant
#non_rel_docs2 contains all the k-p remaining documents which are non relevant

#final_docs2 conatins top k retrieved documents after 2nd iteration
#new_query2 contains updated query after 2nd iteration
rel_docs2=rel1.split(",")
for i in rel_docs1:
    global_rel.append(i)
non_rel_docs2=[]
temp=list(set(out_list2)-set(rel_docs2))
for docs in temp:
    if docs not in global_rel:
        non_rel_docs2.append(docs)

final_docs2,new_query2=rel_feedback(new_query1,rel_docs2,non_rel_docs2)
print("docs after 2nd iteration are: ",final_docs2)
print("\n")



# In[123]:


p2,r2,m2=pre_recall(final_docs2,ground_truth)
print("precision after 2nd iteration are: ",p2)
print("\n")
print("recall after 2nd iteration are: ",r2)
print("\n")
print("MAP after 2nd iteration is: ",m2)


# In[124]:


map_final1.append(m2)


# In[125]:


print("New query vector after 2nd iteration is \n", new_query2)


# In[126]:



qv3=[]
for i in new_query2:
    qv3.append(new_query2[i])
print(qv3)
final_qv.append(qv3)


# In[127]:


print(len(final_qv))


# In[128]:


plt.plot(r2, p2,color="blue")
plt.xlabel("Recall after 2nd iteration")
plt.ylabel("Precision after 2nd iteration")
plt.title(" PR curve after 2nd iteration")


# In[129]:


print(len(final_qv))


# In[130]:


out_list3=[]
for i in range(len(final_docs2)):
    x=final_docs2[i][0]
    out_list3.append(x)
    
print(out_list3)


# In[131]:


startlist2=[]
for i in range(len(final_docs2)):
    x=final_docs2[i][0]
    if x in rel_docs2:
        l=x + "*"
        startlist2.append(l)
    else:
        startlist2.append(x)
    
# print(out_list4)
print("docs after 2nd iteration are: ",startlist2)


# In[133]:


print("Enter the ", number_of_rel_docs, "documents you want to mark as relevant \n")

#rel_docs3 contains all the documents a user enters as relevant
#non_rel_docs3 contains all the k-p remaining documents which are non relevant

#final_docs3 conatins top k retrieved documents after 3rd iteration
#new_query3 contains updated query after 3rd iteration
rel2=input()
rel_docs3=rel2.split(",")
for i in rel_docs2:
    global_rel.append(i)
non_rel_docs3=[]
temp1=list(set(out_list3)-set(rel_docs3))
for docs in temp1:
    if docs not in global_rel:
        non_rel_docs3.append(docs)
final_docs3,new_query3=rel_feedback(new_query2,rel_docs3,non_rel_docs3)
print("docs after 3rd iteration are: ",final_docs3)
print("\n")


# In[134]:


p3,r3,m3=pre_recall(final_docs3,ground_truth)
print("precision after 3rd iteration are: ",p3)
print("\n")
print("recall after 3rd iteration are: ",r3)
print("\n")
print("MAP after 3rd iteration is: ",m3)


# In[135]:


map_final1.append(m3)


# In[136]:


print("New query vector after 3rd iteration is \n", new_query3)


# In[137]:



qv4=[]

for i in new_query3:
    qv4.append(new_query3[i])
print(qv4)
final_qv.append(qv4)


# In[138]:


print(len(final_qv))


# In[139]:


plt.plot(r3, p3,color="blue")
plt.xlabel("Recall after 3rd iteration")
plt.ylabel("Precision after 3rd iteration")
plt.title("Precision Recall Curve after 3rd iteration")


# In[140]:


print(len(final_qv))


# In[141]:


out_list4=[]
for i in range(len(final_docs3)):
    x=final_docs3[i][0]
    out_list4.append(x)
    
print(out_list4)


# In[142]:


startlist3=[]
for i in range(len(final_docs3)):
    x=final_docs3[i][0]
    if x in rel_docs3:
        l=x + "*"
        startlist3.append(l)
    else:
        startlist3.append(x)
    
# print(out_list4)
print("docs after 3rd iteration are: ",startlist3)


# In[143]:


print("Enter the ", number_of_rel_docs, "documents you want to mark as relevant \n")
rel2=input()
#rel_docs4 contains all the documents a user enters as relevant
#non_rel_docs4 contains all the k-p remaining documents which are non relevant

#final_docs4 conatins top k retrieved documents after 3rd iteration
#new_query4 contains updated query after 3rd iteration
rel_docs4=rel.split(",")
for i in rel_docs3:
    global_rel.append(i)
non_rel_docs4=[]
temp3=list(set(out_list4)-set(rel_docs4))
for docs in temp3:
    if docs not in global_rel:
        non_rel_docs4.append(docs)
final_docs4,new_query4=rel_feedback(new_query3,rel_docs4,non_rel_docs4)
print("docs after 4th iteration are: ",final_docs4)
print("\n")


# In[144]:


p4,r4,m4=pre_recall(final_docs4,ground_truth)
print("precision after 4th iteration are: ",p4)
print("\n")
print("recall after 4th iteration are: ",r4)
print("\n")
print("MAP after 4th iteration is: ",m4)


# In[145]:


map_final1.append(m4)


# In[146]:


startlist4=[]
for i in range(len(final_docs4)):
    x=final_docs4[i][0]
    if x in rel_docs4:
        l=x + "*"
        startlist4.append(l)
    else:
        startlist4.append(x)
    
# print(out_list4)
print("docs after 4th iteration are: ",startlist4)


# In[147]:


final_star=[]
for i in range(len(final_docs4)):
    x=final_docs4[i][0]
    if x in global_rel:
        l=x + "*"
        final_star.append(l)
    else:
        final_star.append(x)
    
# print(out_list4)
print("Relevant documents after 4 iterations are: \n ",final_star)


# In[148]:


print("New query vector after 4th iteration is \n", new_query4)


# In[149]:



qv5=[]

for i in new_query4:
    qv5.append(new_query4[i])
final_qv.append(qv5)


# In[150]:


print(len(final_qv))


# In[151]:


plt.plot(r4, p4,color="blue")
plt.xlabel("Recall after 4th iteration")
plt.ylabel("Precision after 4th iteration")
plt.title("Precision Recall Curve")


# In[152]:


print(map_final1)
iterations=['initial query','1st itr','2nd itr','3rd itr','4th itr']
plt.plot(iterations, map_final1,color="blue")
plt.xlabel("Iterations")
plt.ylabel("MAP values")
plt.title("Different MAP values after each iteration")


# In[153]:


from sklearn.manifold import TSNE
labels=['initial query vector','query vector1', 'query vector2','query vector3','query vector4']
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=250, random_state=23)
new_values = tsne_model.fit_transform(final_qv)

x = []
y = []
for value in new_values:
    x.append(value[0])
    y.append(value[1])
        
plt.figure(figsize=(7,7)) 
for i in range(len(x)):
    plt.scatter(x[i],y[i])
    plt.annotate(labels[i],xy=(x[i], y[i]),xytext=(5, 5),textcoords='offset points',ha='right',va='bottom')
plt.show()


# 

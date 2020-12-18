#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np


# # 프린트

# In[3]:


print("Hello, world")


# In[6]:


#integer
x = 3
print("정수: %01d, %02d, %03d, %04d, %05d" % (x,x,x,x,x))


# In[7]:


# float
x = 256.123
print("실수: %.0f, %.1f, %.2f" % (x,x,x))


# In[8]:


# string
x = "Hello, world"
print("문자열: [%s]" % (x))


# # 반복문, 조건문

# In[10]:


contents = ["Regression", "Classification", "SVM", "Clustering", "Demension reduction", "NN", "CNN", "AE", "GAN", "RNN"]
for con in contents:
    if con in ["Regression", "Classification", "SVM", "Clustering", "Demension reduction"]:
        print ("%s 은(는) 기계학습 내용입니다." %con)
    elif con in ["CNN"]:
        print ("%s 은(는) convolutional neural network 입니다." %con)
    else:
        print ("%s 은(는) 심층학습 내용입니다." %con)


# # 반복문과 인덱스

# In[11]:


for (i, con) in enumerate(contents):
    print ("[%d/%d]: %s" %(i, len(contents), con))


# # 함수

# In[12]:


def sum(a, b):
    return a+b

x = 10.0
y = 20.0
print ("%.1f + %.1f = %.1f" %(x, y, sum(x, y)))


# # 리스트

# In[13]:


a = []
b = [1,2,3]
c = ["Hello", ",", "world"]
d = [1,2,3,"x","y","z"]
x = []
print(x)

x.append('a')
print(x)

x.append(123)
print(x)

x.append(["a", "b"])
print (x)


# # 딕셔너리(dictionary)

# In[14]:


dic = dict()
dic["name"] = "Heekyung"
dic["town"] = "Goyang city"
dic["job"] = "Assistant professor"
print dic


# # 클래스

# In[19]:


class Student:
    #생성자
    def __init__(self, name):
        self.name = name
    #메써드
    def study(self, hard=False):
        if hard:
            print "%s 학생은 열심히 공부합니다." %self.name
        else:
            print "%s 학생은 공부합니다." %self.name


# In[20]:


s = Student('Heekyung')
s.study()
s.study(hard=True)


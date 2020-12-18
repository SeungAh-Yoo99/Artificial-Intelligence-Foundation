#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# # 프린트

# In[4]:


def print_val(x):
    print "Type:", type(x)
    print "Shape:", x.shape
    print "값:\n", x
    print " "


# # rank 1 np array

# In[3]:


x = np.array([1, 2, 3])
print_val(x)

x[0] = 5
print_val(x)


# # rank 2 np array

# In[5]:


y = np.array([[1,2,3], [4,5,6]])
print_val(y)

#zeros
a = np.zeros((2,2))
print_val(a)

#ones
a = np.ones((3,2))
print_val(a)

#단위 행렬(identity matrix)
a = np.eye(3,3)
print_val(a)


# # 랜덤 행렬

# In[6]:


# uniform: 0~1 사이 모든 값들이 나올 확률이 같음
a = np.random.random((4,4))
print_val(a)

# Gaussian: 0을 평균으로 하는 가우시안 분포를 따르는 랜덤값
a = np.random.randn(4,4)
print_val(a)


# # np array indexing

# In[7]:


a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
print_val(a)

b = a[:2, 1:3]
print_val(b)

# 행렬의 n번째 행 얻기
row1 = a[1, :] # 1번째 행
print_val(row1)


# # 행렬의 원소별 연산

# In[10]:


m1 = np.array([[1,2], [3,4]], dtype=np.float64)
m2 = np.array([[5,6], [7,8]], dtype=np.float64)

# elementwise sum
print_val(m1 + m2)
print_val(np.add(m1, m2))

# elementwise difference
print_val(m1 - m2)
print_val(np.subtract(m1, m2))

# elementwise product
print_val(m1 * m2)
print_val(np.multiply(m1, m2))

# elementwise division
print_val(m1 / m2)
print_val(np.divide(m1, m2))

# elementwise square root
print_val(np.sqrt(m1))


# # 행렬 연산

# In[8]:


m1 = np.array([[1,2], [3,4]]) # (2,2)
m2 = np.array([[5,6], [7,8]]) # (2,2)
v1 = np.array([9,10]) # (2, 1)  # [[9,10]] (1,2)
v2 = np.array([11,12]) # (2, 1)

print_val(m1)
print_val(m2)
print_val(v1)
print_val(v2)

# 벡터-벡터 연산
print_val(v1.dot(v2))
print_val(np.dot(v1, v2))

# 벡터-행렬 연산
print_val(m1.dot(v1)) # (2,2) x (2,1) -> (2,1)
print_val(np.dot(m1, v1))

# 행렬-행렬 연산
print_val(m1.dot(m2))
print_val(np.dot(m1, m2))


# # 전치 행렬(transpose)

# In[9]:


print_val(m1)
print_val(m1.T)


# # 합

# In[11]:


print_val(np.sum(m1)) # 행렬의 모든 원소의 합
print_val(np.sum(m1, axis=0)) # shape[0] (행) 을 압축시키자. (2,2) -> (2,)
print_val(np.sum(m1, axis=1)) # shape[1] (열) 을 압축시키자. (2,2) -> (2,)

m1 = np.array([[1,2,3], [4,5,6]])
print m1
print m1.shape # (2,3)

print np.sum(m1)
print np.sum(m1, axis=0) # shape[0] (행) 을 압축시키자. (2,3) -> (3,)
print np.sum(m1, axis=1) # shape[1] (열) 을 압축시키자. (2,3) -> (2,)


# # zeros-like

# In[12]:


m1 = np.array([[1,2,3],
              [4,5,6],
              [7,8,9],
              [10,11,12]])
m2 = np.zeros_like(m1)  # m1과 같은 형태의 0으로 이루어진 np array
print_val(m1)
print_val(m2)


# # Matplot library

# In[18]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# sin 커브
x = np.arange(0, 10, 0.1)  # 0~10 까지 0.1 간격의 숫자 배열
y = np.sin(x)

plt.plot(x, y)
plt.show()

# 한 번에 두 개 그래프 그리기
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('sin and cos')
plt.legend(['sin', 'cos'])

plt.show()

# Subplot
plt.subplot(2, 1, 1) # (2,1) 형태 플랏의 첫 번째 자리에 그리겠다
plt.plot(x, y_sin)
plt.title('sin')

plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('cos')

plt.show()


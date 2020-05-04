import pandas as pd 
import numpy as np 

# data=pd.read_csv("event_category.csv")

# print(data.head())
# print(data.info())

# print(data.tail())

# print(data['category'].value_counts())

# data=np.arange(0,100)

# print(data)

# label1=np.zeros(50)
# label2=np.ones(50)

# label=np.append(label1,label2)

# print(label)

# np.random.seed(100)

# np.random.shuffle(data)
# print(data)
# np.random.seed(100)
# np.random.shuffle(label)
# print(label)


# data=pd.read_csv("pred.csv")

# print(data['0'].value_counts())


a=np.array([1,2,3])
b=np.array([[4,5,6]])


# for index, (x,y) in enumerate(zip(a,b)):
#     print("%d%d%d"%(index,x,y))

# a=np.delete(a,[0])
# print(a)
print(b.shape)
print(a.shape)

data=pd.DataFrame(data=np.array([a,b]).T)
print(data)

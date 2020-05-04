import pandas as pd 
import numpy as np 
import json
# event_data=pd.read_csv("GPU/submission_body.csv",header=None)
# category_data=pd.read_csv("GPU/pred.csv",header=None)

# event_data.columns=['id','body']
# category_data.columns=['id','category']

# event_data['category']=category_data['category']

# event_data['category']=event_data['category'].astype(int)


# event_data=event_data[['id','category','body']]

# print(event_data.shape)
# print(event_data.head())


# event_data.to_csv("result.csv",index=False,header=False,sep='\t')


data=pd.read_csv('result.csv',encoding='utf-8',header=None,sep='\t')
data.columns=['id','category','body']
data['category']=data['category'].astype('object')
print(data.head())

with open('map.json',mode='r',encoding='utf-8') as file:
    map=json.load(file)
    map1={}
    for index,(key,value) in enumerate(map.items()):
        map1[index]=value 
    print(map1)
    data['category']=data['category'].map(map1)
    # print(data.head())
    data.to_csv("result.csv",index=False,header=False,sep='\t')
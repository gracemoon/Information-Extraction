import pandas as pd
import numpy as np
import json

data=pd.read_csv("event_entity_train_data_label.csv",sep='\t',encoding='utf-8',header=0)
data.columns=['id','content','event','body']
print(data.head())
print(data.shape)
data.dropna(axis=0,inplace=True)
print(data['event'].unique())
print(len(data['event'].unique()))

map={}
for index,item in enumerate(data['event'].unique()):
    map[index]=item
    print(map)


with open("map.json",encoding='utf-8',mode='w') as file:
    json.dump(map,file)

# data['category']=data['event'].map(map)
# print(data.head())
# print(data.shape)
# data.drop(axis=1,columns=['event','body'],inplace=True)
# data.to_csv("event_category.csv",index=False)


# data.drop(axis=1,columns=['event'],inplace=True)
# data.to_csv("event_body.csv",index=False)

# data['length']=data['content'].str.len()

# print(data['length'].value_counts())

data=pd.read_csv("event_entity_dev_data.csv",sep='\t',encoding='utf-8',header=None)


data.columns=['id','content']

print(data.shape)
print(data.head())

data.to_csv("event_dev.csv",index=False)




import warnings
warnings.filterwarnings("ignore")
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pandas as pd 
import numpy as np 
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import KFold

# 加载GPU
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#默认保存路径
cache_dir="/data1/xul/models/bert/"

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',
            cache_dir=cache_dir)
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 加载Bert模型
model = BertForSequenceClassification.from_pretrained('bert-base-chinese',
    num_labels=27,
    output_attentions=False,
    output_hidden_states=False,
    cache_dir=cache_dir)


print("data is reading")
# 数据预处理
def tokenizer_data(sentences):
    input_ids=[]
    attention_mask=[]
    for sent in sentences:
        token=tokenizer.encode_plus(
            sent,
            pad_to_max_length=True,
            max_length=128,
            add_special_tokens=True,
        )
        input_ids.append(token['input_ids'])
        attention_mask.append(token['attention_mask'])
    return input_ids,attention_mask

# 计算准确率
def accuracy(y,y_pred):
    y_hat=np.argmax(y_pred,axis=1).flatten()
    y=y.flatten()
    return np.sum(y==y_hat)/len(y)

# 读取数据
data=pd.read_csv('event_category.csv',encoding='utf-8')
dev_data=pd.read_csv('event_dev.csv',encoding='utf-8')

# print(data.head())
# print(data.describe())
# print(data.info())
np.random.seed(100)

train_sentences=data['content'].values
np.random.shuffle(train_sentences)
test_sentences=dev_data['content'].values
np.random.seed(100)
train_labels=data['category'].values
ids=dev_data['id'].values
np.random.shuffle(train_labels)
# print(sentences[:10])
# print(labels[:10])

# 划分训练集与验证集
# train_sentences,test_sentences,train_labels,test_labels=train_test_split(sentences,labels,
#     test_size=0.3, random_state=42)

# kflod=KFold(n_splits=5,shuffle=True)
# k=1
# for train_index,test_index in kflod.split(sentences):
#     print("the %d flod:"%k)
#     k+=1
# train_sentences=sentences[train_index]
# test_sentences=sentences[test_index]
# train_labels=labels[train_index]
# test_labels=labels[test_index]
# print(train_sentences[:10])
# print(train_labels[:10])
# print(len(train_sentences))
# print(len(test_sentences))
# print(len(train_labels))
# print(len(test_labels))

train_input_ids,train_attention_mask=tokenizer_data(train_sentences)
test_input_ids,test_attention_mask=tokenizer_data(test_sentences)

print("inputs id is ready")

train_dataset=TensorDataset(torch.tensor(train_input_ids),
    torch.tensor(train_attention_mask),
    torch.tensor(train_labels))
train_loader=DataLoader(train_dataset,batch_size=64)

test_dataset=TensorDataset(torch.tensor(test_input_ids),
    torch.tensor(test_attention_mask))
test_loader=DataLoader(test_dataset,batch_size=64)

optimizer=AdamW(model.parameters(),lr=2e-5,eps=1e-8)

epochs=5

print("training")
model.to(device)
for epoch in range(0,epochs):
    model.train()
    for step,batch in enumerate(train_loader):
        train_input_ids=batch[0].to(device)
        train_attention_mask=batch[1].to(device)
        train_labels=batch[2].to(device)
        model.zero_grad()
        output=model(input_ids=train_input_ids,
                attention_mask=train_attention_mask,
                labels=train_labels)
        loss,logit=output[:2]
        loss.backward()
        optimizer.step()
        if(step%50==0):
            print("the loss is"+str(loss.detach().cpu().numpy()))
            train_accuracy=accuracy(train_labels.detach().cpu().numpy(),logit.detach().cpu().numpy())
            # print(logit.detach().cpu().numpy())

model.eval()
pred=[]
for step,batch in enumerate(test_loader):
    test_input_ids=batch[0].to(device)
    test_attention_mask=batch[1].to(device)
    with torch.no_grad():
        output=model(input_ids=test_input_ids,
                attention_mask=test_attention_mask)
        logit=output[0]
        # print(logit.detach().cpu().numpy())
        step_pred=np.argmax(logit.detach().cpu().numpy(),axis=1).flatten()
        # print(logit.detach().cpu().numpy())
        pred=np.append(pred,step_pred)
pred=pd.DataFrame(data=np.array([ids,pred]).T)
pred.to_csv("pred.csv",index=False,header=False)
torch.save(model,"./model1")







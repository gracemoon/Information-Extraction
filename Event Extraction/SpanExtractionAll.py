import warnings
warnings.filterwarnings("ignore")
from transformers import BertModel,BertTokenizer 
import torch
import pandas as pd
import numpy as np 
from torch.nn import CrossEntropyLoss, MSELoss
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset,DataLoader
from model import BertForSpanExtraction
# model=BertModel.from_pretrained("bert-base-chinese")

model=BertForSpanExtraction()
tokenizer=BertTokenizer.from_pretrained("bert-base-chinese")

def Loss():
    pass 


def list_find(list1, list2):
    """在list1中寻找子串list2，如果找到，返回第一个下标；
    如果找不到，返回-1。
    """
    n_list2 = len(list2)
    for i in range(len(list1)):
        if list1[i: i+n_list2] == list2:
            return i
    return -1
 
def groundTruth(sentences,labels):
    start_positions=[]
    end_positions=[]
    index=[]
    for i,(sentence,label) in enumerate(zip(sentences,labels)):
        # sp=np.zeros(128)
        # ep=np.zeros(128)
        
        all_tokens = tokenizer.tokenize(sentence)
        tokens=tokenizer.tokenize(label)
        start=list_find(all_tokens,tokens)+1
        end=start+len(tokens)-1
        if start!=-1 and end<128:
        #     sp[start]=1
        #     ep[end]=1
        # start_positions.append(sp)
        # end_positions.append(ep)
            start_positions.append(start)
            end_positions.append(end)
        else:
            # print("------------>%d%d"%(start,end))
            index.append(i)
    return index,start_positions,end_positions


# data=pd.read_csv("event_body.csv")
# sentences=data['content']
# labels=data['body']

# train_sentences,test_sentences,train_labels,test_labels=train_test_split(sentences,labels,test_size=0.3)

data=pd.read_csv("event_body.csv")
train_sentences=data['content']
train_labels=data['body']

dev_data=pd.read_csv("event_dev.csv")
test_sentences=dev_data['content']
test_ids=dev_data['id']

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

train_index,train_start_positions,train_end_positions=groundTruth(train_sentences,train_labels)
print(len(train_index))
# print(train_index)
train_sentences=np.delete(train_sentences,train_index)
train_labels=np.delete(train_labels,train_index)
train_input_ids,train_attention_mask=tokenizer_data(train_sentences)

# test_index,test_start_positions,test_end_positions=groundTruth(test_sentences,test_labels)
# test_sentences=np.delete(test_sentences,test_index)
# test_labels=np.delete(test_labels,test_index)
test_input_ids,test_attention_mask=tokenizer_data(test_sentences)
# print(train_start_positions[:5])


# loss,start_scores, end_scores = BertForSpanExtraction(
#     input_ids=torch.tensor(input_ids), 
#     attention_mask=torch.tensor(attention_mask))




def extraction(input_ids,start_scores,end_scores):
    answers=[]
    for index,input_id in enumerate(input_ids):
        all_tokens = tokenizer.convert_ids_to_tokens(input_id)
        answer = ''.join(all_tokens[np.argmax(start_scores[index]):np.argmax(end_scores[index])+1])
        answers.append(answer)
    return answers

# answers=extraction(start_scores,end_scores)

# print(answers)



epochs=2

train_tensorDataset=TensorDataset(torch.tensor(train_input_ids),
                    torch.tensor(train_attention_mask),
                    torch.tensor(train_start_positions),
                    torch.tensor(train_end_positions))
train_dataLoader=DataLoader(train_tensorDataset,batch_size=64)

test_tensorDataset=TensorDataset(torch.tensor(test_input_ids),
                    torch.tensor(test_attention_mask))
test_dataLoader=DataLoader(test_tensorDataset,batch_size=64)

optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,eps=1e-8)

# model.to(device)
for epoch in range(epochs):
    model.train()
    for step,batch in enumerate(train_dataLoader):
        batch_train_input_ids=batch[0]
        batch_train_attention_mask=batch[1]
        batch_train_start_position=batch[2]
        batch_train_end_position=batch[3]
        if(step==0):
            all_tokens=tokenizer.convert_ids_to_tokens(batch[0][0])
            print(all_tokens)
            answer = ''.join(all_tokens[batch[2][0] : batch[3][0]])
            print(answer)

        model.zero_grad()
        output=model(input_ids=batch_train_input_ids,
                        attention_mask=batch_train_attention_mask,
                        start_positions=batch_train_start_position,
                        end_positions=batch_train_end_position)
        loss,start_logits,end_logits=output[:3]
        loss.backward()
        optimizer.step()
        if step%50==0:
            answers=extraction(batch_train_input_ids.detach().numpy(),start_logits.detach().numpy(),end_logits.detach().numpy())
            # print(answers)
            print("%d"%loss.detach().numpy())

model.eval()
answers=[]
for step,batch in enumerate(test_dataLoader):
    batch_train_input_ids=batch[0]
    batch_train_attention_mask=batch[1]
    with torch.no_grad():
        output=model(input_ids=batch_train_input_ids,
                    attention_mask=batch_train_attention_mask)
        start_logits,end_logits=output[:2]
        step_answers=extraction(batch_train_input_ids.detach().numpy(),start_logits.detach().numpy(),end_logits.detach().numpy())
        answers=np.append(answers,step_answers)
print(answers.shape)

pred=pd.DataFrame(data=np.array([test_ids,answers]).T)
pred.to_csv("submission_body.csv",index=False,header=False)





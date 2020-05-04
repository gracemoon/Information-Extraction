from transformers import BertTokenizer,BertForSequenceClassification,AdamW
import torch
import argparse
import os
from torch.utils.data import DataLoader,TensorDataset
from reader import reader
import torch.nn as nn
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--cache-dir',type=str,default=None)
args=parser.parse_args()

if args.cache_dir:
    if os.path.exists(args.cache_dir):
        print('ok')
    else:
        print('the directory is not exist:' + args.cache_dir)
else:
    print('please input the cache dir')



model=BertForSequenceClassification.from_pretrained('bert-base-chinese',
    num_labels=3,
    output_attentions=False,
    output_hidden_states=False,
    cache_dir=args.cache_dir)
#model = nn.DataParallel(model)

tokenizer=BertTokenizer.from_pretrained('bert-base-chinese',
    cache_dir=args.cache_dir)


def tokenizer_data(data):
    input_ids=[]
    attention_mask=[]
    for sequence in data:
        try:
            token=tokenizer.encode_plus(sequence,add_special_tokens=True,
                                       max_length=128,
                                       pad_to_max_length=True)
        except ValueError:
            print(sequence)
        input_ids.append(token['input_ids'])
        attention_mask.append(token['attention_mask'])
    return input_ids,attention_mask

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

print('read csv...............')
train_data,labels=reader('train')
#test_data=reader('test')
print('tokenizer..............')
input_ids,attention_mask=tokenizer_data(train_data)


optimizer=AdamW(model.parameters(),lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8)
num_epochs=10
dataset=TensorDataset(torch.tensor(input_ids),torch.tensor(attention_mask),torch.tensor(labels))
train_dataloader=DataLoader(dataset,batch_size=64)
print('training...............')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(0,num_epochs):
    model.train()
    for step,batch in enumerate(train_dataloader):
        train_input_ids=batch[0].to(device)
        train_attention_mask=batch[1].to(device)
        train_label=batch[2].to(device)
        model.zero_grad()
        output=model(input_ids=train_input_ids,
                     attention_mask=train_attention_mask,
                     labels=train_label)
        loss,logit=output[:2]
        #loss.mean().backward()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step%1000==0:
            print('loss:'+str(loss.item()))
            print('logit:'+str(logit.detach().cpu().numpy()))
            print(flat_accuracy(logit.detach().cpu().numpy(),train_label.detach().cpu().numpy()))

    val_accuracy=0
    val_step=0
    model.eval()
    for step,batch in enumerate(train_dataloader):
        val_input_ids=batch[0].to(device)
        val_attention_mask=batch[1].to(device)
        val_label=batch[2].numpy()
        
        with torch.no_grad():
            output=model(input_ids=val_input_ids,
                         attention_mask=val_attention_mask)
            logits=output[0]
            if step%1000==0:
                print(val_input_ids)
                print(logits)
                print( np.argmax(logits.detach().cpu().numpy(), axis=1).flatten())
            logits=logits.detach().cpu().numpy()
            val_accuracy+=flat_accuracy(logits,val_label)
            
            val_step+=1
    val_accuracy/=val_step
    print('the validation accuracy:'+str(val_accuracy))
            
torch.save(model,'/data1/xul/data/netizen_sentiment_analysis/model')












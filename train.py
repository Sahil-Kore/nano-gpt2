# import requests

# url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# response = requests.get(url)
# # Save to file
# with open("tinyshakespeare.txt", "w", encoding='utf-8') as f:
#     f.write(response.text)
import os
os.getcwd()
with open('./tinyshakespeare.txt',"r") as f:
    text=f.read()
len(text)

print(text[:1000])

chars=sorted(list(set(text)))
vocab_size=len(chars)
print(''.join(chars))
print(vocab_size)

stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for ch,i in stoi.items()}
encode=lambda s:[stoi[c] for c in s] #encode a string output the integer
decode=lambda  l:''.join(itos[int(i)] for i in l)  #decode a list of integers and return the string

print(encode("Hii there"))
print(decode([20, 47, 47, 1, 58, 46, 43, 56, 43]))

import torch
data=torch.tensor(encode(text) , dtype=torch.long)
data.shape,data.dtype
data[:1000]


n=int(len(data)*0.9)
train_data=data[:n]
val_data=data[n:]

block_size=8
train_data[:block_size+1]

#in a example of block size 8 there are 8 subexamples
#18, 47, 56, 57, 58,  1, 15, 47, 58]
#first example given 18 47 follows 
#2nd given 18 and 47 56 follows and so on

x=train_data[:block_size]
y=train_data[1:block_size+1]

for t in range( block_size):
    context=x[:t+1]
    target=y[t]
    print(f"When input is {context} out is {target}")

torch.manual_seed(1337)
batch_size=64 #how many independent sequences will be processed
block_size=256 #maximum conteext length for prediction

def get_batch (split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data) - block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1]for i in ix])
    return x.to("cuda"),y.to("cuda")

xb,yb =get_batch("train")
print("Inputs")
print(xb.shape)
print(xb)
print("targets")
print(yb.shape)
print(yb)

for b in range(batch_size):
    for t in range(block_size):
        context=xb[b,: t+1]
        target=yb[b,t]
        print(f"When input is {context} output is {target}")
        
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)
        
    def forward(self,idx,targets=None):
        #idx and targets both are (B,T) shape tensors
        #initially the every character has their embedding added in the third dimension in logits
        #that is for xb cell there is vocab_size dimensional vector coming out of the page
        #this vocab_size vector will then act as logits 
        #so every position will be the probability of the next character 
        #the loss function will then be used to optimize the loss
        logits=self.token_embedding_table(idx)  #(B,T,C)

        #cross does requires the logits to be (B,C,T) or a 2d array of (T,C)
        #so to simply we just convert the logits to a 2d array same for target
        if targets is None:
            loss=None
        else:    
            B,T,C =logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #get the prediction
            logits,loss=self(idx)
            #focus only on the last time step
            logits=logits[: ,-1,:] #becomes (B,C)
            #apply softmax to get probabilites
            probs=F.softmax(logits,dim=-1) #(B,C)
            #sample from the distribution
            idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
            #append the sampled index to the running sequence
            idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx

m=BigramLanguageModel(vocab_size)
logits,loss=m(xb,yb)

logits.shape
loss

idx=torch.zeros((1,1),dtype=torch.long)
idx
print(decode(m.generate(idx,max_new_tokens=100)[0].tolist()))


#training the model

optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)
batch_size=64
for steps in range(10000):
    #sample a batch of data
    xb,yb= get_batch('train')
    
    #evaulate the loss
    logits,loss=m(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(m.generate(idx,max_new_tokens=500)[0].tolist()))


#transformer

#toy example
torch.manual_seed(1337)
B,T,C=4,8,2
x=torch.randn(B,T,C)
x.shape

#creating a attention example that works like bag of words
#lets say we want to predict the 5th token so for that we will take the average of the channel(predictions) of the previous 4 tokens 
#as we are doing average the order of the tokens doesnt matter hence bagofwords

#we want x[b,t]=mean{i<=t} x[b,i]
xbow=torch.zeros((B,T,C))
for b in range (B):
    for t in range (T):
        xprev=x[b,:t+1] #(t,C)
        xbow[b,t]=torch.mean(xprev,0)
        
x[0]
xbow[0]
xbow

#to do it more efficiently we can do
'''
1 0 0
1 1 0
1 1 1

multiplied by any matrix
2 7
6 6
6 5

the result will be 
2 7
8 13
14 18

the first row will be sum of first 1 row
second row will be sum of first 2 rows
and so on
a=torch.tril(torch.ones(3,3)) # creates a lower triangular matrix
a=[1,0,0
   1,1,0
   1,1,1]
similiarly to get the average for the sum of previous rows and average them we can just normalize the rows of the triangylar matrix
'''

a=torch.tril(torch.ones(3,3))
a=a/torch.sum(a,1,keepdim=True)
a
b=torch.randint(0,10,(3,2)).float()
c=a@b
c

#weights
wei=torch.tril(torch.ones(T,T))
wei=wei/wei.sum(1,keepdim=True)
xbow2=wei @ x #(B,T,C) @ (T,T) x will be broadcasted to (B,T,T)
torch.allclose(xbow,xbow2,atol=1e-6)

#another method of doing the same is softmax
tril=torch.tril(torch.ones(T,T))
wei=torch.zeros((T,T))
wei=wei.masked_fill(tril==0,float('-inf'))
wei=torch.softmax(wei,dim=-1)
xbow3=wei @ x
torch.allclose(xbow,xbow3,atol=1e-6)

n_embd=384
class BigramLanguageModel(nn.Module):
    
        #now the ebedding size is not the same as vocab size so we use a linear layer to convert embeddings to logits
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.lm_head=nn.Linear(n_embd ,vocab_size)
        
        
    def forward(self,idx,targets=None):
        B,T=idx.shape

        token_embd=self.token_embedding_table(idx)  #(B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T)) #(T,C)
        x=token_embd + pos_emb
        logits=self.lm_head(x) #(B,T,vocab_size)
        if targets is None:
            loss=None
        else:    
            B,T,C =logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #get the prediction
            logits,loss=self(idx)
            #focus only on the last time step
            logits=logits[: ,-1,:] #becomes (B,C)
            #apply softmax to get probabilites
            probs=F.softmax(logits,dim=-1) #(B,C)
            #sample from the distribution
            idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
            #append the sampled index to the running sequence
            idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx


#self attention
torch.manual_seed(1337)
B,T,C=4,8,32
x=torch.randn(B,T,C)

#single head of self attention
head_size=16
key=nn.Linear(C,head_size,bias=False)
query=nn.Linear(C,head_size,bias=False)


#what i am
k=key(x)    #(B,T,head_size)

#what i want 
q=query(x)  #(B,T,head_size)

#if you find me interesting what i will communicate
#used for flexibility earlier people used x itself
value=nn.Linear(C,head_size,bias=False)

#dot product of queries and keys of all the tokens in a batch
wei=q @ k.transpose(-2,-1) #(B,T,16) @(B,16,T) 

#wew use scaling because q and k are taken from snd but wei has a variance of 17 so we need to scale it
wei=q @ k.transpose(-2,-1) * head_size**-0.5 #(B,T,16) @(B,16,T)  

#if we do not scale wei ,after softmax wei will ocnverge to one hot vectors
# 


wei.shape
tril=torch.tril(torch.ones(T,T))
wei=wei.masked_fill(tril==0,float("-inf"))
wei=F.softmax(wei,dim=-1)
v=value(x)

out=wei @ v
out
wei[0]


class Head(nn.Module):
    #one head of self attention
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
    
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        #compute attention scores(affinities)
        wei=q @ k.transpose(-2,-1) * head_size**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1)
        
        v=self.value(x)
        out=wei @ v
        return out
        
        

class BigramLanguageModelAttention(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.sa_head=Head(n_embd)
        self.lm_head=nn.Linear(n_embd ,vocab_size)
        
        
    def forward(self,idx,targets=None):
        B,T=idx.shape

        token_embd=self.token_embedding_table(idx)  #(B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T)) #(T,C)
        x=token_embd + pos_emb
        x=self.sa_head(x) #apply one head of self attention
        logits=self.lm_head(x) #(B,T,vocab_size)
        if targets is None:
            loss=None
        else:    
            B,T,C =logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond=idx[:,-block_size:]
            #get the prediction
            logits,loss=self(idx_cond)
            #focus only on the last time step
            logits=logits[: ,-1,:] #becomes (B,C)
            #apply softmax to get probabilites
            probs=F.softmax(logits,dim=-1) #(B,C)
            #sample from the distribution
            idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
            #append the sampled index to the running sequence
            idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx

model=BigramLanguageModelAttention()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
batch_size=64
for steps in range(5000):
    #sample a batch of data
    xb,yb= get_batch('train')
    
    #evaulate the loss
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(model.generate(idx,max_new_tokens=500)[0].tolist()))

class MultiHeadAttention(nn.Module):
    'multiple heads of self attention in parallel'

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range (num_heads)])
    
    def forward(self,x):
        return torch.cat([h(x) for h in self.heads],dim=-1)
    


class BigramLanguageModelMultiHead(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.sa_heads=MultiHeadAttention(4,n_embd//4)
        self.lm_head=nn.Linear(n_embd ,vocab_size)
        
        
    def forward(self,idx,targets=None):
        B,T=idx.shape

        token_embd=self.token_embedding_table(idx)  #(B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T)) #(T,C)
        x=token_embd + pos_emb
        x=self.sa_heads(x) #apply one head of self attention
        logits=self.lm_head(x) #(B,T,vocab_size)
        if targets is None:
            loss=None
        else:    
            B,T,C =logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond=idx[:,-block_size:]
            #get the prediction
            logits,loss=self(idx_cond)
            #focus only on the last time step
            logits=logits[: ,-1,:] #becomes (B,C)
            #apply softmax to get probabilites
            probs=F.softmax(logits,dim=-1) #(B,C)
            #sample from the distribution
            idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
            #append the sampled index to the running sequence
            idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx


model=BigramLanguageModelMultiHead()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
batch_size=64
for steps in range(5000):
    #sample a batch of data
    xb,yb= get_batch('train')
    
    #evaulate the loss
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(model.generate(idx,max_new_tokens=500)[0].tolist()))


class FeedForward(nn.Module):
    ' a single linear layer followed by a non linearity'
    
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_embd , n_embd)
                               ,nn.ReLU())
        
    def forward(self,x):
        return self.net(x)
    

class BigramLanguageModelMultiHead_ff(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.sa_heads=MultiHeadAttention(4,n_embd//4)
        self.ffwd=FeedForward(n_embd)
        self.lm_head=nn.Linear(n_embd ,vocab_size)
        
        
    def forward(self,idx,targets=None):
        B,T=idx.shape

        token_embd=self.token_embedding_table(idx)  #(B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T)) #(T,C)
        x=token_embd + pos_emb
        x=self.sa_heads(x) #apply one head of self attention
        x=self.ffwd(x)
        logits=self.lm_head(x) #(B,T,vocab_size)
        if targets is None:
            loss=None
        else:    
            B,T,C =logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond=idx[:,-block_size:]
            #get the prediction
            logits,loss=self(idx_cond)
            #focus only on the last time step
            logits=logits[: ,-1,:] #becomes (B,C)
            #apply softmax to get probabilites
            probs=F.softmax(logits,dim=-1) #(B,C)
            #sample from the distribution
            idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
            #append the sampled index to the running sequence
            idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx


model=BigramLanguageModelMultiHead_ff()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
batch_size=32
for steps in range(5000):
    #sample a batch of data
    xb,yb= get_batch('train')
    
    #evaulate the loss
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(model.generate(idx,max_new_tokens=500)[0].tolist()))


#the above model uses a multihead attention followed by a ff nn only once
#this is called a block 
#normally we have ultiple such blocks

class Block(nn.Module):
    #transformer block communication followed by computation
    #communication->self-attention
    #computation -ff neural network
    def __init__(self,n_embd,n_head):
        
        super().__init__()
        head_size=n_embd//n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embd)
        
    def forward(self,x):
        x=self.sa(x)
        x=self.ffwd(x)
        return x
    

class MultiBlockBigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)

        self.blocks=nn.Sequential(
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4)
        )
        self.lm_head=nn.Linear(n_embd,vocab_size)
     
    def forward(self,idx,targets=None):
        B,T=idx.shape

        token_embd=self.token_embedding_table(idx)  #(B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T)) #(T,C)
        x=token_embd + pos_emb
        x=self.blocks(x) #apply one head of self attention
        logits=self.lm_head(x) #(B,T,vocab_size)
        if targets is None:
            loss=None
        else:    
            B,T,C =logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond=idx[:,-block_size:]
            #get the prediction
            logits,loss=self(idx_cond)
            #focus only on the last time step
            logits=logits[: ,-1,:] #becomes (B,C)
            #apply softmax to get probabilites
            probs=F.softmax(logits,dim=-1) #(B,C)
            #sample from the distribution
            idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
            #append the sampled index to the running sequence
            idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx

model=MultiBlockBigramModel()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
batch_size=64
for steps in range(5000):
    #sample a batch of data
    xb,yb= get_batch('train')
    
    #evaulate the loss
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(model.generate(idx,max_new_tokens=500)[0].tolist()))

# this takes a lot of time to train and the accuracy is also low
#deep neural nets require more time to train and more epochs too
#to tackle this we use residual connection

class Block(nn.Module):
    #transformer block communication followed by computation
    #communication->self-attention
    #computation -ff neural network
    def __init__(self,n_embd,n_head):
        
        super().__init__()
        head_size=n_embd//n_head
        self.sa=MultiHeadAttention(n_head,head_size)
        self.ffwd=FeedForward(n_embd)
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)
        
        
    def forward(self,x):
        x= x + self.sa(self.ln1(x))
        x= x + self.ffwd(self.ln2(x))
        return x
    
class MultiHeadAttention(nn.Module):
    'multiple heads of self attention in parallel'

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range (num_heads)])
        self.proj=nn.Linear(n_embd,n_embd)
    
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.proj(out)
        return out
    

class FeedForward(nn.Module):
    ' a single linear layer followed by a non linearity'
    
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_embd , n_embd*4),
                               nn.ReLU(),
                               nn.Linear(n_embd*4,n_embd) 
                               )
        
    def forward(self,x):
        return self.net(x)

class MultiBlockBigramModel_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)

        self.blocks=nn.Sequential(
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),
            Block(n_embd,n_head=4),
            nn.LayerNorm(n_embd)
        )
        self.lm_head=nn.Linear(n_embd,vocab_size)
     
    def forward(self,idx,targets=None):
        B,T=idx.shape

        token_embd=self.token_embedding_table(idx)  #(B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T)) #(T,C)
        x=token_embd + pos_emb
        x=self.blocks(x) #apply one head of self attention
        logits=self.lm_head(x) #(B,T,vocab_size)
        if targets is None:
            loss=None
        else:    
            B,T,C =logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond=idx[:,-block_size:]
            #get the prediction
            logits,loss=self(idx_cond)
            #focus only on the last time step
            logits=logits[: ,-1,:] #becomes (B,C)
            #apply softmax to get probabilites
            probs=F.softmax(logits,dim=-1) #(B,C)
            #sample from the distribution
            idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
            #append the sampled index to the running sequence
            idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx

model=MultiBlockBigramModel_resnet()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
batch_size=64
for steps in range(5000):
    #sample a batch of data
    xb,yb= get_batch('train')
    
    #evaulate the loss
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(model.generate(idx,max_new_tokens=500)[0].tolist()))



#changing some lines to make it easy to increase the size of the model
device="cuda"
n_layer=6
dropout=0.2
n_head=6
n_embd=384
block_size=256
learning_rate=3e-4
torch.set_float32_matmul_precision('high')
class Head(nn.Module):
    #one head of self attention
    def __init__(self,head_size):
        super().__init__()
        self.head_size=head_size
        self.key=nn.Linear(n_embd,head_size,bias=False)
        self.query=nn.Linear(n_embd,head_size,bias=False)
        self.value=nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout=nn.Dropout(dropout)
    
    def forward(self,x):
        B,T,C=x.shape
        k=self.key(x)
        q=self.query(x)
        #compute attention scores(affinities)
        wei=q @ k.transpose(-2,-1) * self.head_size**-0.5
        wei=wei.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        wei=F.softmax(wei,dim=-1)
        wei=self.dropout(wei)
        v=self.value(x)
        out=wei @ v
        return out
        


class MultiHeadAttention(nn.Module):
    'multiple heads of self attention in parallel'

    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range (num_heads)])
        self.proj=nn.Linear(n_embd,n_embd)
        self.droupout=nn.Dropout(dropout)
    
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.droupout(self.proj(out))
        return out
    
    
class FeedForward(nn.Module):
    ' a single linear layer followed by a non linearity'
    
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd , n_embd*4),
            nn.ReLU(),
            nn.Linear(n_embd*4,n_embd),
            nn.Dropout(dropout))
        
    def forward(self,x):
        return self.net(x)


class MultiBlockBigramModel_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)

        self.blocks=nn.Sequential(*[
            Block(n_embd,n_head=n_head)for _ in range (n_layer)])
        self.ln_f=nn.LayerNorm(n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
     
    def forward(self,idx,targets=None):
        B,T=idx.shape

        token_embd=self.token_embedding_table(idx)  #(B,T,C)
        pos_emb=self.position_embedding_table(torch.arange(T)) #(T,C)
        x=token_embd + pos_emb
        x=self.blocks(x) #apply one head of self attention
        logits=self.lm_head(x) #(B,T,vocab_size)
        if targets is None:
            loss=None
        else:    
            B,T,C =logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond=idx[:,-block_size:]
            #get the prediction
            logits,loss=self(idx_cond)
            #focus only on the last time step
            logits=logits[: ,-1,:] #becomes (B,C)
            #apply softmax to get probabilites
            probs=F.softmax(logits,dim=-1) #(B,C)
            #sample from the distribution
            idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
            #append the sampled index to the running sequence
            idx=torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx

model=MultiBlockBigramModel_resnet()
optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate)
for steps in range(100):
    #sample a batch of data
    xb,yb= get_batch('train')
    
    #evaulate the loss
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(f"epoch {steps}")

print(loss.item())
print(decode(model.generate(idx,max_new_tokens=500)[0].tolist()))



import torch
import torch.nn as nn
import torch.nn.functional as F

# Define constants
n_layer = 6
dropout = 0.2
n_head = 6
n_embd = 384
block_size = 256
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_float32_matmul_precision('high')


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size=head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MultiBlockBigramModel_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embd = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_embd + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Instantiate model and send to GPU
model = MultiBlockBigramModel_resnet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
model=torch.compile(model)
# Training loop
for step in range(5000):
    xb, yb = get_batch('train')
    with torch.autocast(device_type=device):
        logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step%100==0:
        print(f"Epoch {step}, Loss: {loss.item()}")

# Text generation
idx = torch.zeros((1, 1), dtype=torch.long, device=device)  # starting token
output = model.generate(idx, max_new_tokens=10000)
decoded_text=decode(output[0].tolist())
print(decoded_text)


with open("generated_tiny_shakespere.txt","w",encoding='utf-8') as f:
    f.write(decoded_text)
    
torch.save(model.state_dict(),"model.pth")

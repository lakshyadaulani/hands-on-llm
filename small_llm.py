


import os
from datasets import load_dataset
from dataclasses import dataclass
import tiktoken
import numpy as np
from tqdm import tqdm
# 1. Load the dataset
ds = load_dataset("roneneldan/TinyStories")
#explore dataset
print("-------------")
print(ds)
# Overview (splits + sizes)
print(ds.keys())
train_ds = ds["train"]
print(train_ds) # Train split summary
print(train_ds.features)
# First example
print("FIRST EXAMPLE IN TRAIN SPLIT",train_ds[0])# Dict with 'text'
print("-------------")
## we will use BPE used in gpt2 from tiktoken library
enc = tiktoken.get_encoding("gpt2")
'''
In below code,
- Function process takes a sample row/text/doc and creates its encoding
- ds has splits train and validation.
- map is applied to every split.
- process is run on every row in each split.
- remove_columns=['text'] drops the original 'text' column after processing, leaving only 'ids' and 'len'. (Saves RAM/disk.)
- desc provides a progress bar label: tokenizing the splits.
- num_proc=8 spawns 8 worker processes for parallel tokenization. Faster, but higher CPU + memory + temporary disk usage.
Result (tokenized) is a new DatasetDict with the same splits, each split now having columns: ids (list<int>) and len (int).
'''
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    out = {'ids': ids, 'len': len(ids)}
    return out
tokenized = ds.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=8,
    )

print("-------tokenised output------")
print(tokenized)# Overview (splits + sizes)
print(tokenized.keys())
print(tokenized["train"][0]["len"],tokenized["train"][10]["len"])
# length is the number of words in a document, hence it will be different for different rows
# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = f'{split}.bin' #we are using .bin file because we don't want to store this on RAM, we want to store it on disk.
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    #np.memmap creates a memory-mapped file, allowing us to work with arrays larger than RAM.
    #memmap creates a binary file on disk that can be accessed like a numpy array, but doesn't load the entire file into memory at once. You can write to it chunk by chunk without holding everything in RAM.
    # 'w+' mode creates the file if it doesn't exist, or overwrites it if
    total_batches = 1024
    # we will split our entire dataset into 1024 batches and write each batch to the memmap file one by one.
 
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush() #arr.flush will ensure everythin is written to disk
 
    # Some functions from https://github.com/karpathy/nanoGPT/blob/master/train.py with slight modifications
# this function will give a batch of input and output data, where y is x shifted right
#block size = context window
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/6147212…
    if split == 'train':
        data = np.memmap('train.bin', dtype=np.uint16, mode='r')
    else:
        data = np.memmap('validation.bin', dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,)) #random position to start from for taking input output
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        # pin_memory locks memory of tensor in RAM, preventing it from being swapped to disk. This enables faster and non-blocking transfers to GPU.
        # this simply means that we ask OS to not move this memory to disk, because GPU will soon need this.
 
        # to_device by default copies tensors to GPU synchronously (blocking). This means CPU will wait until the copy is done before proceeding.
        # non_blocking=True allows CPU to continue executing subsequent code while the copy to GPU happens
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
 
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
 
 
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias = config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))
            
 
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
 
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
 
## feed forward layer in the end of transformer block (expansion+compression)
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
 
# transformer block
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    def forward(self, x):
        x_ln1 = self.ln_1(x)
        x_attn = self.attn(x_ln1)
        x = x + x_attn
        # x = x + self.attn(self.ln_1(x))
        x_ln2 = self.ln_2(x)
        x_mlp = self.mlp(x_ln2)
        x = x + x_mlp
        # x = x + self.mlp(self.ln_2(x))
        return x
    
@dataclass
class GPTConfig:
    block_size: int #should be 4 here
    vocab_size: int #for example total number of words in english language
    n_layer: int
    n_head: int
    n_embd: int # dimension of word embeddings
    dropout: float = 0.0
    bias: bool = True
 
 
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            #word token embedding:learns a dense vector of size n_embd for each token in the vocabulary of size vocab_size
            # Input shape (Batch, Tokens) -> output (Batch, Tokens, n_embd)
            'wte': nn.Embedding(config.vocab_size, config.n_embd),
            # word position embedding: Learns a vector per position index [0 .. block_size-1]. Same embedding dim n_embd.
            # Added to wte output to inject order: x = wte(tokens) + wpe(positions).
            'wpe': nn.Embedding(config.block_size, config.n_embd),
            'drop': nn.Dropout(config.dropout),
            # h: nn.ModuleList of transformer Blocks (your Block class). Length = n_layer.
            # They are iterated sequentially: for block in self.transformer['h']: x = block(x).
            'h': nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # Final LayerNorm applied after last block (standard GPT practice) to stabilize and normalize before projecting to logits (you will typically add a Linear head: nn.Linear(n_embd, vocab_size) on top of this).
            'ln_f': LayerNorm(config.n_embd, bias=config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
 
        # nn.Module.apply(fn) walks the entire module tree (the model plus all its children) depth‑first.
        #For every submodule it calls fn(submodule).
        # so every Linear and Embedding inside the model (embeddings, attention projections, MLP layers, lm_head, etc.) gets initialized uniformly.
        self.apply(self._init_weights)
 
        #Post‑pass selective re‑initialization loop
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'): #MLP layer in output projection
                # Reinitializes those weights with a reduced standard deviation: 0.02 / sqrt(2 * n_layer).
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
                # Dividing by sqrt(2 * n_layer) approximates a He‑style / residual scaling to keep activations’ variance roughly constant across depth.
                # More stable gradients in deeper models; mitigates training instabilities in deep transformers.
 
    def _init_weights(self, module):
        # This matches GPT‑2 style initialization (small Gaussian, zero biases) giving stable early training and avoiding large activation magnitudes.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
 
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size
        pos = torch.arange(0, t, dtype=torch.long, device=device)
 
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
 
        # Prevents overfitting by randomly zeroing embedding dimensions, forcing the model to not rely on precise co‑adapted feature combinations.
        # Acts like embedding dimension noise: Encourages more distributed, robust token representations.
        x = self.transformer.drop(tok_emb + pos_emb) #this will go as input to transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
 
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:, [-1], :])
            return logits, None
 
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens given a conditioning sequence.
        idx: Tensor of shape (B, T)
        """
        for _ in range(max_new_tokens):
            # generate() method autoregressively generates new tokens. As generation proceeds, idx grows by concatenating newly predicted tokens
            # below code Truncates the input sequence to fit within the model's maximum context window (block_size) during generation
            # idx.size(1): Current sequence length T (dim 1 is the token dimension; dim 0 is batch).
            # Else (idx.size(1) > block_size): Take only the last block_size tokens: idx[:, -self.config.block_size:]
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            #forward pass. returns logits and loss (ignore loss here in inferencing)
            logits, _ = self(idx_cond)
            # Takes only the last token position's logits ./ temperature: Controls randomness
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
                
# ?? Why apply then override?
 
# First pass gives uniform baseline to all modules.
# Second pass selectively tightens variance only for residual output projections (c_proj) where accumulation risk is highest.
# Doing override after apply ensures the specialized initialization is not undone later.
 
config = GPTConfig(
    vocab_size=50257,     # use the tokenizer's vocab size
    block_size=128,       # or whatever context size you're training with
    n_layer=6, #number of transformer blocks
    n_head=6, #number of attention heads
    n_embd=384,
    dropout=0.1,
    bias=True
)
 
model = GPT(config)
 
def estimate_loss(model):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out
 
# Training Config
import torch
from contextlib import nullcontext
 
learning_rate = 1e-4 #more stable training, earlier 1e-4
max_iters = 10000  #increase from 5000
warmup_steps = 1000 #smoother initial train, earlier 100
min_lr = 5e-4 #lower rate, earlier 5e-4
eval_iters = 500 # increased from 100
batch_size = 32 # changed from 16, better gradient estimate
block_size = 128 #changed from 64, capture longer range dependencies
 
gradient_accumulation_steps = 32 # reduced from 50
 
device =  "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
 
# How to use autocast https://wandb.ai/wandb_fc/tips/reports/How-To-Use-Autocast-in-PyTorch--VmlldzoyMTk4NTky
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
 
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
 
torch.set_default_device(device)
torch.manual_seed(42)
 
from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR
 
##PUT IN WEIGHT DECAY, CHANGED BETA2 to 0.95
optimizer =  torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9) #weight decay for regularization
 
scheduler_warmup = LinearLR(optimizer, total_iters = warmup_steps) #Implement linear warmup
scheduler_decay = CosineAnnealingLR(optimizer,T_max = max_iters - warmup_steps, eta_min = min_lr) #Implement lr decay
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps]) #Switching from warmup to decay
 
# https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-…
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
 
 
best_val_loss = float('inf')
best_model_params_path =  "best_model_params_10k.pt"
train_loss_list, validation_loss_list = [], []
 
# Ensure model is on the correct device
model = model.to(device)
 
# In your training loop
for epoch in tqdm(range(max_iters)):
    if epoch % eval_iters == 0 and epoch != 0:
        # Ensure estimate_loss uses the correct device
        losses = estimate_loss(model)
        print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        train_loss_list += [losses['train']]
        validation_loss_list += [losses['val']]
 
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(), best_model_params_path)
 
    # Ensure X and y are on the correct device
    X, y = get_batch("train")
    X, y = X.to(device), y.to(device)
 
    with ctx:
        logits, loss = model(X, y)
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
 
    if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    scheduler.step()
 
    #Load the model
model = GPT(config)  # re-create the model with same config
device =  "cuda" if torch.cuda.is_available() else "cpu"
best_model_params_path =  "best_model_params_5k.pt"
model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device))) # load best model states
 
#inference
sentence = "Once upon a time there was a pumpkin."
context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim = 0))
y = model.generate(context, 200)
print(enc.decode(y.squeeze().tolist()))
# %%
# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F

# %%
# Load input data
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(f"length of dataset in characters: {len(text)}")
# %%
# Get unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)
# %%
# Define encoder decoder for the tokenizer
c2i = {ch: i for i, ch in enumerate(chars)}
i2c = {i: ch for ch, i in c2i.items()}
encode = lambda s: [c2i[c] for c in s]  # Encoder: takes a string and outputs list of integers.
decode = lambda l: "".join([i2c[i] for i in l])  # Decoder: takes a list of integers and outputs a string.
print(encode("Hi there!, how is it going?"))
print(decode([20, 47, 1, 58, 46, 43, 56, 43, 2, 6, 1, 46, 53, 61, 1, 47, 57, 1, 47, 58, 1, 45, 53, 47, 52, 45, 12]))

# %%
# Encode text input using torch.Tensor

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])
# %%
# Split in train and val data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
# %%
# Define block size
block_size = 8
print(train_data[: block_size + 1])
# %%
x = train_data[:block_size]
y = train_data[1 : block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    print(f"When input is {context}, the target is {target}")
# %%
# Generate batch
torch.manual_seed(1337)
batch_size = 4
block_size = 8


def get_batch(split):
    # Generate small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


xb, yb = get_batch("train")
print("Inputs:")
print(xb.shape)
print(xb)
print("Targets:")
print(yb.shape)
print(yb)
print("-----")

for b in range(batch_size):  # Loop through batch dimension
    for t in range(block_size):  # Loop through block dimension
        context = xb[b, : t + 1]
        target = yb[b, t]
        print(f"When input is {context.tolist()}, the target is {target}")

# %%
torch.manual_seed(1337)


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # Alternative way:
            # logits = logits.transpose(1, 2)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get predictions
            logits, _ = self(idx)
            # Focus only on last time step
            logits = logits[:, -1, :]
            # Apply softmax
            probs = F.softmax(logits, dim=1)  # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T)
        return idx


model = BigramLanguageModel(vocab_size=vocab_size)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss)

# %%
idx = torch.zeros((1, 1))

#%%
#===========================
# Mathematical trick in self-attention
#

# Toy example:
torch.manual_seed(seed=1337)
B, T, C = 4, 8, 2
x = torch.randn(B, T, C)
x.shape

#%%
# Goal: x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] #(t, C)
        xbow[b, t] = torch.mean(xprev, 0)

#%%
# version 2
wei = torch.tril(torch.ones(T, T))
wei = wei/wei.sum(1, keepdim=True)
xbow2 = wei @ x #(B, T, T) @ (B, T, C) ---> (B, T, C)
torch.allclose(xbow, xbow2)
#%%
# version 3
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
print(wei)
wei = wei.masked_fill(tril == 0, float("-inf"))
print(wei)
wei = F.softmax(wei, dim=-1)
print(wei)
xbow3 = wei @ x
torch.allclose(xbow, xbow3)
# %%
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a/torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b
print("a=")
print(a)
print("--")
print("b=")
print(b)
print("--")
print("c=")
print(c)
# %%
# version 4: self-attention!
torch.manual_seed(1337)
B, T, C = 4, 8, 32 # Batch, Time, # of Channels
x = torch.randn(B, T, C)

# Every token emits two vectors: Q and K.
# Query: What am I looking for?
# Key: What do I contain?
# Value: Value of the token

# Single head self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, head_size)
q = query(x) # (B, T, head_size)
wei = q @ k.transpose(-2, -1) # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)

#%%

tril = torch.tril(torch.ones(T, T))
# wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v



# %%
# scaled attention
k = torch.randn(B, T, head_size)
q = torch.randn(B, T, head_size)
wei = q @ k.transpose(-2, -1)
print(k.var())
print(q.var())
print(wei.var())
# %%
print(f"Original softmax output: {torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)}")
for n in range(1, 100, 10):
    print(f"If n factor is {n}, then softmax output is {torch.softmax(n*torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)}")
# %%

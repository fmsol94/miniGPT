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

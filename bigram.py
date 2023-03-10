# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 30000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
# --------------

# Define seed to reproduce results in future
torch.manual_seed(1337)

# Load input data
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Get unique characters
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Define encoder decoder for the tokenizer
c2i = {ch: i for i, ch in enumerate(chars)}
i2c = {i: ch for ch, i in c2i.items()}
encode = lambda s: [c2i[c] for c in s]  # Encoder: takes a string and outputs list of integers.
decode = lambda l: "".join([i2c[i] for i in l])  # Decoder: takes a list of integers and outputs a string.

# Encode text input using torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split in train and val data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data loading
def get_batch(split):
    # Generate small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


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
m = model.to(device)

# Define optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in tqdm(range(max_iters)):
    # Every eval_interval evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch("train")

    # Evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

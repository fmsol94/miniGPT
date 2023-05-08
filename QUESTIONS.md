# Questions to answer
1. Why variance of Q @ K.T is in the order of head_size?

2. Why is it important to keep that variance in the order of 1?

It is important for this product to be fairly diffuse when computing the softmax. Especially for initialization purposes. If Q@K.T has very extreme values, softmax will converge towards one hot vectors.
Quick check
```python
print(f"Original softmax output: {torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)}")
for n in range(1, 100, 10):
    print(f"If n factor is {n}, then softmax output is {torch.softmax(n*torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)}")
```
This will cause information to be aggregated for one main node, and that is something we want to avoid. Especially during initialization.
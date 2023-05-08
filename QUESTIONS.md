# Questions to answer
## 1. Why variance of Q @ K.T is in the order of head_size?

Let's assume that the elements of the Q and K vectors are independent random variables with mean 0 and variance 1. The dot product of Q and K can be written as:

```Q @ K.T = sum(q_i * k_i) for i = 1 to head_size```

As the dot product is a sum of the products of the corresponding elements, the variance of the dot product scales with the head size.

```Var(Q @ K.T) = sum(Var(q_i * k_i)) for i = 1 to head_size```

Under the assumptions made above, the variance of each product term (q_i * k_i) is 1, so the variance of the dot product is approximately equal to the head size:

```Var(Q @ K.T) ≈ head_size```

This is the reason for dividing the dot product by the square root of the head size, as it helps normalize the variance and maintain numerical stability during the training process.

## 2. Why is it important to keep that variance in the order of 1?

It is important for this product to be fairly diffuse when computing the softmax. Especially for initialization purposes. If Q@K.T has very extreme values, softmax will converge towards one hot vectors.
Quick check
```python
print(f"Original softmax output: {torch.softmax(torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)}")
for n in range(1, 100, 10):
    print(f"If n factor is {n}, then softmax output is {torch.softmax(n*torch.tensor([0.1, -0.2, 0.3, -0.2, 0.5]), dim=-1)}")
```
This will cause information to be aggregated for one main node, and that is something we want to avoid. Especially during initialization.

## 3. Why 0 here: x.mean(0, keepdim=True)?

```x.mean(0, keepdim=True)``` calculates the mean along the first dimension (rows) and retains the reduced dimension as a singleton dimension in the output tensor. This is particularly useful if you need to perform element-wise operations between the original tensor and the mean tensor.

## 4. Why do we use ```__call__``` in LayerNorm
`__call__` is a special method in Python that allows a class instance to be called as if it were a function. When you define the __call__ method in a class, it gets executed when an object of that class is called as a function. This makes the syntax for using the class more concise and intuitive.
## 5. Review additional papers

## 6. Why n_embd 384?
384/6 heads we get a 64 dimension as a standard. In practice, many successful Transformer-based models, such as BERT and GPT-2, use similar head sizes. BERT-base, for example, has an embedding dimension of 768 and uses 12 attention heads, resulting in a head size of 64 (768/12). These models have demonstrated strong performance across a wide range of NLP tasks, suggesting that such configurations can work well in practice.
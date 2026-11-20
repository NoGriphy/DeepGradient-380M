### (The Application of Gamma and Beta):
While many machine-learning developers already know the purpose of Gamma (γ) and Beta (β), it is worth breaking down how they work in this architecture. These learnable parameters highlight or reinforce the significance of any feature (column) that the model considers important. Specifically, γ scales up positive values, while β slightly shifts the final output to make key features easier to detect. At the same time, unimportant features are scaled down so the model is able to safely ignore them. Occasionally, β even shifts a negative feature value into the positive range, preserving its importance in the final calculation. Overall, this mechanism helps the model catch the crucial relationships and filter out the unrelated context.

In the code below, I have defined both Gamma and Beta for every layer because it requires their own G&B values in order to know which direction it should adjust on their features.
```python
# These pre-defined variables, Gamma and Beta for attention and feed-forward network, stay outside of the for-loop list, to initialize every neural layer's weights later.
self.attention_gamma_values = []
self.attention_beta_values = []
self.ffn_gamma_values = []
self.ffn_beta_values = []
```

Applying ones and zeros for Gamma and Beta, respectively, helps to determine how large the directional step should take. In other words, base on the math calculation '(layer_normalization_result * Gamma) + (Beta)', if Gamma is a 1-dimensional of zeros, it will absolutely destroy what the layer-normalization has been calculating along the way and add whatever beta value afterwards - this final result is completely meaningless. Because it has the multiplication step, we define Gamma as ones for directional scaling to prevent multiplying by zero and ruin every calculated result. 

Every neural layer's G&B adjusts their number correspondingly from backpropagation; each layer's γ and β parameters are updated independently based on the gradients computed from the subsequent operations.
```python
# Using a for-loop degrades the processing speed; however, to make this entire model engine work, I will stick to standard Python loop - for now.
# 'self.model_neural_layer' is the number of layers the model have - currently 12. So, each layer receives their own G&B values so they can update independently without cross-contamination. 
for _ in range(self.model_neural_layer):
  self.attention_gamma_values.append(t.ones(1536, dtype = t.float32, device = 'cuda').requires_grad_(True))
  self.attention_beta_values.append(t.zeros(1536, dtype = t.float32, device = 'cuda').requires_grad_(True))
  self.ffn_gamma_values.append(t.ones(1536, dtype = t.float32, device = 'cuda').requires_grad_(True))
  self.ffn_beta_values.append(t.zeros(1536, dtype = t.float32, device = 'cuda').requires_grad_(True))
```
This is ready for the heavy, complex matrix multiplication on the later stage. 

## Development of DeepGradient AI Model *(380 Million Parameters)*
This records my personal progress on building my own text-language model, DeepGradient, from scratch by implementing formal mathematical approach to natural language processing. Any modifications or changes have made to the entire language model would be updated in this repository.

### (Encountered Challenges with Passable Solutions):
The most difficult section when building a neural network is considering how to prevent the gradient from either reaching almost 0.0 or a massive number (causing infinity). Because a model cannot learn anything, obviously, from zeros or infinity. Initially, applying Xavior initialization is to stabilize the matrix scalars to prevent exploding numbers. The method is '1 / sqrt(input_num))'. However, this assists the raw tensor matrices from increasing in forward-pass computation, but it does not permanently solve the 'inf' issue during backward-pass. I multiply both raw decompression and output matrices by '2' and the amount of neural layers to produce even smaller results. Because these computations are what altered most of the scalars' size in my code. I have attached an example below in the terminal.

```bash
Note: The '12' here is the number of neural layers that the model currently have.

1 / sqrt(3072) = 0.01804219591
1 / sqrt(3072 * 2 * 12) = 0.00368284781

Although the numbers have become tiny after this extra modification, but I personally still recommend this change in the code. 
```
```python
# First Thought: Standard decrease by leveraging Xavior method:
self.raw_feature_compression_grid.append((t.randn(3072, 1536, dtype = t.float32, device = 'cuda') * (1 / (3072) ** 0.5)).requires_grad_(True))

# Modification Made: Scalars' size could unpredictable sometimes, so I reduce to even smaller scale.
self.raw_feature_compression_grid.append((t.randn(3072, 1536, dtype = t.float32, device = 'cuda') * (1 / (3072 * 2 * self.model_neural_layer) ** 0.5)).requires_grad_(True))
```


### THIS REPOSITORY IS WAITING FOR UPDATES.

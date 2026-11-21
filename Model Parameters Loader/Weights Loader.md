**It is recommended to check out the 'Model Parameters Saver' folder before viewing this explanation.**
**I used PyTorch's load function, which has the ability to load the raw weights without any unwanted data - although I am the one who often 
saves the file, keeping 'weight_only' to True is always safe. Additionally, setting the device location to CUDA for parallel processing instead
of CPU to increase the computing speed (if the data was processed previously on GPU, adding this parameter is redundant; however, defining
every parameter is for better clarity).**

**1.** Three main parameters to pass in 'torch.load()' is the file location, device location, and whether to load only the raw tensors. Because
"load_model_parameters" contains all the elements from 'self.model_parameters' (specified in 'Model Parameters Saver'), we are going to run
through three for-loops to conduct in-place copy operation without allocating new memory in computer's RAM.
```python
load_model_parameters = t.load(current_saved_model_file, map_location = 'cuda', weights_only = True)
```

**2.** 'with torch.no_grad()' is to cancel the computation dependency-tree or graph so the weights can be updated without calculating the gradients
for this specific block; wrapping 'with torch.no_grad()' to stop tracking the backward-pass calculation. Normally, storing immediate values cause the 
GPU or RAM to grow significantly, which is why disabling it prevent OoM (Out-of-Memory) error. Momentum and variance buffers, and general 
weights are copied independently by using in-place copy operation - 'torch.copy()'. Using Python's built-in 'zip()' speeds up the unpacking process. The 
PyTorch function takes the passed weights and inject them directly to the raw, random weights. 
```python
with t.no_grad():
  for current_matrix, saved_matrix in zip(self.model_parameters, load_model_parameters['model_weights']):
    current_matrix.copy_(saved_matrix)
  
  for current_mt_buffers, saved_mt_buffers in zip(self.m_buffers, load_model_parameters['momentum_buffers']):
    current_mt_buffers.copy_(saved_mt_buffers)
  
  for current_var_buffers, saved_var_buffers in zip(self.v_buffers, load_model_parameters['variance_buffers']):
    current_var_buffers.copy_(saved_var_buffers)
```

**3.** Lastly, directly modify 'self.current_step' with its clone and print out the final message to ensure
the operations have been completed. 
```python
self.current_step = load_model_parameters['current_stage']
print("\n[Previously Trained Weights Have Been Successfully Loaded]\n")
```

**(FULL-SECTION CODE BELOW):**
```python
def LoadModelWeight(self, current_saved_model_file = 'anygradient_mini_model_0.5.pt'):
  load_model_parameters = t.load(current_saved_model_file, map_location = 'cuda', weights_only = True)
  
  with t.no_grad():
    for current_matrix, saved_matrix in zip(self.model_parameters, load_model_parameters['model_weights']):
        current_matrix.copy_(saved_matrix)
  
    for current_mt_buffers, saved_mt_buffers in zip(self.m_buffers, load_model_parameters['momentum_buffers']):
        current_mt_buffers.copy_(saved_mt_buffers)
  
    for current_var_buffers, saved_var_buffers in zip(self.v_buffers, load_model_parameters['variance_buffers']):
        current_var_buffers.copy_(saved_var_buffers)
  
  self.current_step = load_model_parameters['current_stage']
  print("\n[Previously Trained Weights Have Been Successfully Loaded]\n")
```

**To save the model's previously trained weights without learning from-scratch, I packed everything into a dictionary and use PyTorch's save function
to store the data into a designated file - 'current_saved_model_file'. I choose to use PyTorch's save function because it is able to precisely
store every weight and improve the security, flexibility, and storage speed. 'torch.save()' helps to writes from raw metadata separatelyto a compressed
zip archive. Additionally, it is really helpful when 'torch.load()' is utilized to load the model's weights, setting its parameters, 'weight_only', to 
True to prevent injecting malicious, unwanted data (e.g., classes, functions, modules, or etc) into the program. This only inserts the raw tensors into the model
without executing arbitrary code.**

**1.** 'model_saved_parameters' contains momentum and variance buffers from Adaptive Optimizer (Adaptive Moment Estimation/Adam method), each neural layer's Query,
Key, Value, feature compression, feature decompression, attention gamma & beta, feed-forward network gamma & beta, current step in training, and raw output weight 
(for consolidating the calculated meanings between each multi-head attention after they have successfully reconnected using reshaping method). This is ready to save.
```python
model_saved_parameters = {
  "model_weights": self.model_parameters,
  "momentum_buffers": self.m_buffers,
  "variance_buffers": self.v_buffers,
  "current_stage": self.current_step
}
```

**2.** As mentioned above, PyTorch's save function stores a given argument with the file location - 'torch.save(model_content, file_location_name)'. Lastly, using a
print statement to check if it has finished the saving operation. 
```python
t.save(model_saved_parameters, current_saved_model_file)
print(f"\n[Saved DeepGradient's Internal Weights to '{current_saved_model_file}']\n")
```

**(FULL-SECTION CODE BELOW)**
```python
def SaveModelWeights(self, current_saved_model_file = 'anygradient_mini_model_0.5.pt'):
  model_saved_parameters = {
    "model_weights": self.model_parameters,
    "momentum_buffers": self.m_buffers,
    "variance_buffers": self.v_buffers,
    "current_stage": self.current_step
  }
  
  t.save(model_saved_parameters, current_saved_model_file)
  print(f"\n[Saved DeepGradient's Internal Weights to '{current_saved_model_file}']\n")
```

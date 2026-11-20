**To prevent the tokenized file, "tokens_file_name", occupying excessive GPU (VRAM) space, I utilized Numpy's built-in
memory map (memmap) function that builds a connection or "bridge" between the file to computer's memory. So, everytime
when the program accesses the file, the system extracts the size required in SSD (disk) without using memory. This approach
is efficient when you come to large datasets that come along with multi-gigabytes (GB) - this avoids the risk of running out of 
memory while using computer's disk space.**

**1.** I chose to use unsigned 16-bits (2^16 = 65,536 distinct values and use 2-bytes) to fit Tiktoken's (library from OpenAI) GPT-2 vocabulary range,
which has a maximum value of 50,257 (65,536 > 50,257). Above unsigned 16-bits integers like 18-bits or 32-bits remain too much unused values. 
```python
token_file_access = np.memmap(tokens_file_name, dtype = np.uint16, mode = 'r')
```

**2.** To store what the model will be learning and the correct next-words, I defined two standard lists. 
```python
x_input_batches = []
y_target_batches = []
```

**3.** I was wondering how to randomly choose a random starting position or index so the model learns in an consolidated way, instead of choosing
sequentially. The purpose is that a text-language model needs to learn from unpredictable batches to force its weights to be at a "middle-ground" 
where it has most of the knowledge to answer a user's question. PyTorch's 't.randint(low, high, size, data_type, device)' generates from 0 (inclusive)
to 'len(token_file_access)' (exclusive). 'model_context_window' is subtracted from 'len(token_file_access)' to prevent out of boundary error. For
example, if the dataset file has 8,000 tokens and the model's context window is 512, when 'randint' selects Index 7,950, that means it is going to
extract 512 tokens according to that position. 7,950 + 512 = 8,462, which exceeds the file's maximum size: 8,000. That is why the subtraction is
crucial for this function because the system can only choose between 0 to (8,000 - 512 = 7,488), and Index 7,488 + 512 is 8,000, which is exactly
right on the maximum edge of the file, preventing crashing the code. The amount of integers 'randint' can choose is 4 due to hardware limit. So,
there will be 4 indices getting randomly chosen. 
```python
random_context_position = t.randint(0, len(token_file_access) - model_context_window, (phrase_batch_size,), dtype = t.int64, device=  'cuda')
```

**4.** The for-loop goes through those 4 randomly chosen integers and extract their corresponding batch from that position. "x_input_tokens" is
responsible for acquiring new batches, while "y_target_tokens" fetches the correct next-words for that specific batch. Then, all of the sentences
are appended to their respective list. 
```python
for training_starting_position in random_context_position:
  x_input_tokens = token_file_access[training_starting_position:training_starting_position + model_context_window]
  y_target_tokens = token_file_access[training_starting_position + 1:training_starting_position + model_context_window + 1]
  
  x_input_batches.append(x_input_tokens)
  y_target_batches.append(y_target_tokens)
```

**5.** In computer's memory (RAM), those selected batches (data) are completely scattered around. I used Numpy's stack function to gather all the data
together so it increases CPU's processing speed; unlike standard Python lists, Numpy packs the raw data into one contiguous, uniform block that boosts
computer's overall speed. Lastly, the code returns the stored data after converting Numpy data to tensors data type, along with device location.
```python
organized_data_x = np.stack(x_input_batches)
organized_data_y = np.stack(y_target_batches)

return (t.from_numpy(organized_data_x).to(dtype = t.int64, device = 'cuda'),
        t.from_numpy(organized_data_y).to(dtype = t.int64, device = 'cuda'))
```

**(FULL-SECTION CODE BELOW)**
```python
def DataTokensFromSSD(self, tokens_file_name, phrase_batch_size, model_context_window):
  token_file_access = np.memmap(tokens_file_name, dtype = np.uint16, mode = 'r')
  
  x_input_batches = []
  y_target_batches = []
  
  random_context_position = t.randint(0, len(token_file_access) - model_context_window, (phrase_batch_size,), dtype = t.int64, device=  'cuda')
  for training_starting_position in random_context_position:
    x_input_tokens = token_file_access[training_starting_position:training_starting_position + model_context_window]
    y_target_tokens = token_file_access[training_starting_position + 1:training_starting_position + model_context_window + 1]
  
    x_input_batches.append(x_input_tokens)
    y_target_batches.append(y_target_tokens)
  
  organized_data_x = np.stack(x_input_batches)
  organized_data_y = np.stack(y_target_batches)
  
  return (t.from_numpy(organized_data_x).to(dtype = t.int64, device = 'cuda'),
          t.from_numpy(organized_data_y).to(dtype = t.int64, device = 'cuda'))
```

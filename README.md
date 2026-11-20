## Development of DeepGradient AI Model *(380 Million Parameters)*
This records my personal progress on building my own text-language model, DeepGradient, from scratch by implementing formal mathematical approach to natural language processing. Any modifications or changes have made to the entire language model would be updated in this repository.

### (General Notes): 
1. The model's name is DeepGradient.
2. The model's context window is 512 tokens.
3. The model processes 16,384 tokens per episode (8 sets of batch, 4 batches/sentences in every set, 512 tokens in every batch).
4. The model has around 381M (381,000,000) parameters/weights - close to OpenAI's GPT2-Medium text-language model.

### (Training Plan/Strategy):
1. To prevent overfitting, the text-language model goes through 5 epochs for each dataset. 
2. Making the model understand English grammar before learning other topics.

### (Training History):
1. The model has learned from **'deepgradient_data_file.bin' (Stage 1: More than 80% are English instructions)** tokenized dataset for 7,650 episodes. 
The file contains 25,064,202 (~25M) tokens. Because each episode computes 16,384 tokens, the text-language
model requires around 1,530 episodes (25,064,202 / 16,384) to achieve one completion (or learn the entire dataset once). 
The GOAL is to let the model learns 5 epochs while not overfitting or memorizing the exact tokens. The total amount of
training episodes is 7,650 (1,530 * 5). 

**(Validation)**: 7,650 * 16,384 = 125,337,600 tokens. Dividing the result by 5 is around 25,067,520, which is very near
the file's token size. 

The final loss score for Episode 7,650 is 2.64 (3 significant figures). Keep in mind that this loss score could be an average speculation of how well
the model predicts the next token to match the context of the sentence. Different batch receives different loss scores. 'deepgradient_data_file.bin' is
making the model to master English grammar first before diving into next topic (e.g., coding, mathematics, physics, etc). 

**[Next Step]**: Since the model has a bit context on English instructions, it will go through 70% coding dataset to see how well it performs. 


$$\color{#00BFFF}{\texttt{( ---------- SECOND TRAINING RECORD IS BELOW ---------- )}}$$


2. The model has learned through **'anygradient_TS.bin' (1) (Stage 2: 70% coding (12,600 rows), 20% English instruction/casual conversations (3,600 rows), ~0.2% Basic Greetings (50 rows))**
with 1,325 episodes. Since the current model has the ability to process 16,384 tokens per episode, according to "General Notes", and the total number of tokens for file is 4,343,270. 
The GOAL is to finish 5 epochs. To achieve one completion, the number of episodes is 265 (rounded to 3 significant figures; 4,343,270 / 16,384). Five completions is 1,325 episodes (265 * 5). 

**(Event Occurred - Low Score)**: The model appears to have a loss score around 0.6 to 0.9 on Episode 1,000. This could be caused by analyzing the English phrases that have previously learned 
from the first training. According to the file's dataset, 20% are English instructions, because the model needs to preserve the normal conversation knowledge without forgetting. Coding occupies 
a large portion of the data (70%), which provides enough chance for the model to learn Python syntax. Overall, it is expected the model to have a moderately lower score on the English batches, 
and a high score in coding.

The final loss for Episode 1,325 is 0.6 (1 significant figure).

**[Next Step]**: The model needs to review general English instructions.


$$\color{#00BFFF}{\texttt{( ---------- THIRD TRAINING RECORD IS BELOW ---------- )}}$$


3. Revising English instructions with partial coding dataset via **'anygradient_TS.bin' (2) (Stage 2.5: 65% English instructions (11,700 rows), 34% Coding (6,120 rows), 1% Basic Greetings (180 rows))**.
The model replies with its pure reasoning rather than memorizing a specific sentence. Although there is a slight overfitting on self-introduction, decreasing it to 1% so it has tiny chance of 
getting chosen by the custom batch selector while preserving its previous knowledge/identity. 

The total tokens for the file is 4,115,108. By following the 5-epochs rule, the total amount of episodes is 1,255 (4,115,108 / 16,384 = 251 (3sf), 255 * 5 = 1,255).

The final loss score for Episode 1,255 is 1.7 (2sf).

**[Next Step]**: The model requires common sense/knowledge in order to reason logically on math and coding later. 


$$\color{#00BFFF}{\texttt{( ---------- FOURTH TRAINING RECORD IS BELOW ---------- )}}$$


4. The model answers some of the user's questions logically but still lacks of common sense. To fix this, the model will go through this 'HuggingFaceTB/cosmopedia' to build up general knowledge. 
The file **'anygradient_TS.bin' (3) (Stage 2.8: World Knowledge (~56%), Fresh Python (~26%), Intro Math (~16%), Greetings (~1%))** has 15,409,026 tokens. This time, decreasing the epoch counts to
2. The model trains with 2 epochs to see if it is able to master simple concepts. 

One completion: 15,409,026 / 16,384 = 940 (2sf). Two epochs is 1,880 (940 * 2). 

The final loss score for Episode 1,880 is 2.4 (2sf).


$$\color{#00BFFF}{\texttt{( ---------- FIFTH TRAINING RECORD IS BELOW ---------- )}}$$


...


### THIS REPOSITORY IS WAITING UPDATES FROM THE CREATOR...

# llm4090
Prototype LLM based on the GPT2 architecture that can be trained on a laptop 4090. Written in pytorch.

# features
- optimizations: CUDA graph replay + jit compilation
- dataset for prototyping: wikitext-103-raw-v1 (0.5 GB) - https://huggingface.co/datasets/Salesforce/wikitext
- sequence packing + custom attention mask
- 6 transformer layers/blocks with 6 attention heads - https://medium.com/@hsinhungw/gpt-2-detailed-model-architecture-6b1aad33d16b
- ~50M parameters

# results
training and validation loss:
- 1000 epochs, 100 steps/epoch, batch size = 64, cosine learning rate annealing

![image](https://github.com/user-attachments/assets/3a9fdf3b-44b2-4438-9c68-09007a4a054c)
![image](https://github.com/user-attachments/assets/74da70f6-cc67-4960-89ac-26a1f2d5269c)

# TODO:
- Gradient accumulation: batch size 64 -> 256
- Larger dataset to reach the 40 token/model param target
- Verification using deterministic sequences
- Investigation of predicted token distributions

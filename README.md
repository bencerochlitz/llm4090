Prototype LLM based on standard transformer architecture that can be trained on a laptop 4090. Written in pytorch.

# Features
- optimizations: CUDA graph replay + torch.compile
- dataset: 8 GB of https://huggingface.co/datasets/Skylion007/openwebtext
- 6 transformer layers/blocks with 6 attention heads - https://pub.towardsai.net/llama-explained-a70e71e706e9
- ~90M parameters (bf16)
- batch size = 128 with gradient accumulation

# Results
training and validation loss:
- 50k steps, cosine learning rate annealing

![image](https://github.com/user-attachments/assets/c943a9bb-c819-418e-b6af-c8c7e381a4b8)


# Inference example from the test set
- run inference: cd python && python runner.py --mode=infer
- input: "...Coming out to my immediate family and close friends was easy, but taking my journey to the next level could not "
- next 50 tokens: " be nonsense. Firas is Trump’s last person, sure, but the truth is that he really stood in them, in uniform throughout the country, and often has ties to western world politics. tour of Firas’ military endeavors will need to be"

# TODO:
- Async data loader, new data to device during training
- Rotary embeddings

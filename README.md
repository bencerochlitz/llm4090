# llm4090
Prototype LLM based on the GPT2 architecture that can be trained on a laptop 4090. Written in pytorch.

- dataset ready for training and inference checkpoint are available in the repo
- run training: cd python && python runner.py --mode=train
- run inference: cd python && python runner.py --mode=infer

# features
- optimizations: CUDA graph replay + torch.compile
- dataset for prototyping: wikitext-103-raw-v1 (0.5 GB) - https://huggingface.co/datasets/Salesforce/wikitext
- sequence packing + custom attention mask
- 6 transformer layers/blocks with 6 attention heads - https://medium.com/@hsinhungw/gpt-2-detailed-model-architecture-6b1aad33d16b
- ~50M parameters (bf16)
- batch size = 2 * 128 with gradient accumulation

# results
training and validation loss:
- 50k steps, cosine learning rate annealing

<img src="https://github.com/user-attachments/assets/3a9fdf3b-44b2-4438-9c68-09007a4a054c" width="400" height="250">
<img src="https://github.com/user-attachments/assets/74da70f6-cc67-4960-89ac-26a1f2d5269c" width="400" height="250">

# inference example from the test set
- input: ..."How to Curse was performed at Bush Theatre in the"
- next 20 tokens: "  North of England on Valentine 's Day . In 2001 , he noted the controversial portrayal of the role"
- word context and grammar seem to be close to OK

# TODO:
- Larger dataset to reach the 40 token/model param target
- Rotary embeddings
- Verification using deterministic sequences
- Investigation of predicted token distributions

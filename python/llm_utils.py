import torch

@torch.jit.script
def compute_targets(data, target_buf, eot_token: int):
    # roll left
    # RuntimeError: "roll_cuda" not implemented for 'UInt16'
    targets = torch.roll(data, -1, dims=-1)
    
    # predict eot tokens after eot tokens
    mask = data == eot_token
    print("compute_targets mask shape: ", mask.shape)
    
    targets[mask] = eot_token
    
    # copy result
    target_buf.copy_(targets)

@torch.jit.script
def sample_batch(batch, batch_ids, data, data_ids, targets, batch_targets):
    device = batch.device
    N = len(data)
    B = len(batch)
    
    # sizes are const, so it's graph-safe
    ids = torch.randint(0, N, (B, ), device=device)
    
    # batch tokens and pos embeddings
    # RuntimeError: "index_cuda" not implemented for 'UInt16'
    batch.copy_(data[ids])
    batch_ids.copy_(data_ids[ids])
    
    # token targets
    batch_targets.copy_(targets[ids])
    
    
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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

# @torch.jit.script
def att_mask_packed_seq(tokens, eot: int):
    eot_mask = tokens == eot
    
    cumsum = torch.cumsum(eot_mask, -1)
    cumsum = torch.roll(cumsum, 1)
    cumsum[:, 0] = 0
    
    # print(eot_mask)
    # print(cumsum)
    
    T = tokens.shape[-1]
    mask = cumsum.unsqueeze(-1).repeat(1, 1, T)
    
    # print(mask.shape)
    
    # NOTE: cuda graph capture doesn't like this line if jitted
    mask = mask == torch.transpose(mask, dim0=-2, dim1=-1)
    
    mask = torch.tril(mask, diagonal=0)
    
    # torch multihead attention: does not attend if True
    return ~mask
    
    # # return torch.ones((8, T, T), device=tokens.device)

assert torch.cuda.is_available()
device = torch.device('cuda:0')

if __name__ == '__main__':
    
    # test att_mask_packed_seq
    eot = 50256
    
    tokens = torch.tensor([[eot, 1, 2, eot, 3, 4, 5, eot],
                           [0, 1, eot, 2, 3, 4, 5, eot]],
                          dtype=torch.int, device=device)
    
    mask = att_mask_packed_seq(tokens, eot)
    
    print(mask)
    
    
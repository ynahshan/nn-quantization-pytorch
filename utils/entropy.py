import torch


def shannon_entropy(t, handle_negative=False):
    # workaround for out of memory issue
    torch.cuda.empty_cache()

    pk = torch.unique(t.flatten(), return_counts=True)[1]

    probs = pk.float() / pk.sum()
    probs[probs == 0] = 1
    entropy = -probs * torch.log2(probs)
    res = entropy.sum()

    return res

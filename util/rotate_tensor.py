import torch
import torch.nn.functional as F

def single_rotate(tensor):
    """
    Rotate 90 degrees of tensor.
    tensor : [bz, 3, H, W]
    """
    tensor = torch.flip(tensor, (2,))
    tensor = torch.permute(tensor, (0, 1, 3, 2))
    return tensor


def multiple_rotate(tensor, rotate_times=1):
    """
    Conduct multiple single_rotate
    tensor : [bz, 3, H, W]
    """
    for _ in range(rotate_times):
        tensor = single_rotate(tensor)
    return tensor


def multiple_rotate_all(tensor, all_rotate_times=[0, 1, 2, 3]):
    """
    tensor : [bz, 3, H, W]
    """
    all_tensors = []
    for rotate_times in all_rotate_times:
        rotate_tensor = multiple_rotate(tensor, rotate_times)
        all_tensors.append(rotate_tensor)
    output_tensor = torch.cat(all_tensors, dim=0)
    return output_tensor


def mask_tensor(tensor, fea_size, mask_num=10):
    bz, num_channels, cur_fea_size = tensor.shape[0], tensor.shape[1], tensor.shape[-1]
    fea_len = int(fea_size ** 2)
    rand_values = torch.randn(bz, fea_len).cuda()
    reserve_mask = torch.zeros(bz, fea_len).cuda()
    reserve_indexes = rand_values.argsort(dim=-1)[:, :fea_len - mask_num]
    reserve_mask.scatter_(1, reserve_indexes, 1)    # (bz, fea_len)
    reserve_mask = reserve_mask.reshape(bz, fea_size, fea_size) # (bz, fea_size, fea_size)
    out_reserve_mask = reserve_mask
    reserve_mask = reserve_mask[:, None].repeat(1, num_channels, 1, 1)    # (bz, num_channels, fea_size, fea_size)
    if cur_fea_size != fea_size:
        reserve_mask = F.interpolate(reserve_mask, (cur_fea_size, cur_fea_size), mode='nearest')    # (bz, num_channels, cur_fea_size, cur_fea_size)
    out_tensor = tensor.mul(reserve_mask)

    return out_tensor, out_reserve_mask
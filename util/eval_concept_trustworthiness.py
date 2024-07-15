import torch
import numpy as np
from tqdm import tqdm
from util.local_parts import attributes_names, part_attributes_names, id_to_attributes, id_to_bbox, id_to_part_loc, part_name_to_part_indexes


def get_activation_maps(ppnet, test_loader, corre_proto_num=10):
    # Get the averaged activation maps of all prototypes
    ppnet.eval()
    if hasattr(ppnet, 'module'):
        ppnet = ppnet.module
    all_proto_acts, all_targets, all_img_ids = [], [], []
    for _, (data, targets, img_ids) in tqdm(enumerate(test_loader)):
        data = data.cuda()
        targets = targets.cuda()
        _, proto_acts = ppnet.push_forward(data)

        all_proto_acts.append(proto_acts.detach())
        all_targets.append(targets)
        all_img_ids.append(img_ids)
    all_proto_acts = torch.cat(all_proto_acts, dim=0)   # (n_samples, 2000, fea_h, fea_w)
    all_targets = torch.cat(all_targets, dim=0)
    all_img_ids = torch.cat(all_img_ids, dim=0)

    # Get the ground-truth attributes of the all test images
    all_attributes = []
    for img_id in all_img_ids:
        attributes = id_to_attributes[img_id.item()]
        all_attributes.append(attributes)
    all_attributes = np.stack(all_attributes, axis=0)
    all_attributes = torch.from_numpy(all_attributes).cuda()
    n_attributes = all_attributes.shape[-1]

    # Get the averaged activation map of each attribute
    corre_proto_num = 10
    a_weight = ppnet.attributes_predictor.weight.detach() # (112, 2000)
    a_highest_indexes = torch.argsort(a_weight, dim=1, descending=True)   # (112, 2000), Get the indexes of the sorted weights
    corre_proto_indexes = a_highest_indexes[:, :corre_proto_num]    # (112, corre_proto_num)

    all_activation_maps = []    # all_activation_maps[i] : (n_select_samples, fea_h, fea_w) for the i-th attribute
    for attri_idx in tqdm(range(n_attributes)):
        attri_labels = all_attributes[:, attri_idx]
        select_img_indexes = torch.nonzero(attri_labels == 1).squeeze(dim=1)   # Select all the test images containing this attribute
        cur_corre_proto_indexes = corre_proto_indexes[attri_idx]
        select_proto_acts = all_proto_acts[select_img_indexes][:, cur_corre_proto_indexes]  # (n_select_samples, corre_proto_num, fea_h, fea_w)

        activation_maps = select_proto_acts.mean(dim=1)    # (n_select_samples, fea_h, fea_w)
        all_activation_maps.append(activation_maps)
    return all_activation_maps, all_img_ids


def evaluate_concept_trustworthiness(all_activation_maps, all_img_ids, bbox_half_size=36, img_size=224):
    """
    all_activation_maps[i] : (n_select_samples, fea_h, fea_w) for the i-th attribute
    all_img_ids[i] : img_id for the i-th image
    """
    # Get the ground-truth attributes of the all test images
    all_attributes = []
    for img_id in all_img_ids:
        attributes = id_to_attributes[img_id.item()]
        all_attributes.append(attributes)
    all_attributes = np.stack(all_attributes, axis=0)
    all_attributes = torch.from_numpy(all_attributes).cuda()
    n_attributes = all_attributes.shape[-1]

    # Gather the part locs of each image
    all_img_num, part_num = all_img_ids.shape[0], 15
    all_part_locs = np.zeros((all_img_num, part_num, 2)) - 1   # Each element is (gt_y, gt_x)
    all_img_ids = all_img_ids.cpu().numpy()
    for idx, img_id in enumerate(all_img_ids):
        img_id = img_id.item()
        bbox = id_to_bbox[img_id]
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        part_locs = id_to_part_loc[img_id]
        for part_loc in part_locs:
            part_id = part_loc[0] - 1   # Begin From 0
            loc_x, loc_y = part_loc[1] - bbox_x1, part_loc[2] - bbox_y1
            ratio_x, ratio_y = loc_x / (bbox_x2 - bbox_x1), loc_y / (bbox_y2 - bbox_y1)
            re_loc_x, re_loc_y = int(img_size * ratio_x), int(img_size * ratio_y)
            all_part_locs[idx, part_id, 0] = re_loc_y
            all_part_locs[idx, part_id, 1] = re_loc_x
    all_part_locs = torch.from_numpy(all_part_locs).cuda()

    # Only evaluate the part attributes
    all_loc_acc, all_attri_idx, all_num_samples = [], [], []
    for attri_idx in tqdm(range(n_attributes)):
        attribute_name = attributes_names[attri_idx]   # The name of current attribute
        if attribute_name not in part_attributes_names:    # Only evaluate the part attributes, eliminate the attributes for the whole body
            continue
        part_indexes = torch.LongTensor(part_name_to_part_indexes[attribute_name]).cuda()

        attri_labels = all_attributes[:, attri_idx]
        select_img_indexes = torch.nonzero(attri_labels == 1).squeeze(dim=1)   # Select all the test images containing this attribute

        n_select_samples = select_img_indexes.shape[0]
        select_activaton_maps = all_activation_maps[attri_idx] # (n_select_samples, fea_h, fea_w)
        
        # Get the activation maps
        select_activaton_maps = select_activaton_maps[:, None]   # (n_select_samples, 1, fea_h, fea_w)
        upsampled_activation_maps = torch.nn.functional.interpolate(select_activaton_maps, size=(img_size, img_size), mode='bicubic')
        upsampled_activation_maps = upsampled_activation_maps.squeeze(dim=1)    # (n_select_samples, img_h, img_w)
        
        # Get the prediction bboxes
        max_indice = upsampled_activation_maps.flatten(start_dim=1).argmax(dim=-1)
        mi_h, mi_w = torch.div(max_indice, img_size, rounding_mode='trunc'), max_indice % img_size
        bhz = bbox_half_size
        pred_y1, pred_y2, pred_x1, pred_x2 = torch.where(mi_h - bhz >= 0, mi_h - bhz, 0), \
                                            torch.where(mi_h + bhz <= img_size, mi_h + bhz, img_size), \
                                            torch.where(mi_w - bhz >= 0, mi_w - bhz, 0), \
                                            torch.where(mi_w + bhz <= img_size, mi_w + bhz, img_size)
        pred_bboxes = torch.stack([pred_y1, pred_y2, pred_x1, pred_x2], dim=1)  # (n_select_samples, 4)

        # Get the ground-truth part locations
        part_locs = all_part_locs[select_img_indexes]   # (n_select_samples, 15, 2)
        part_indexes = part_indexes[None, :, None].repeat(n_select_samples, 1, 2)
        part_locs = torch.gather(part_locs, 1, part_indexes)    # (n_select_samples, Np, 2), Np is the number of part location annotations of current attribute in the image
        part_exist = part_locs.sum(dim=-1) > 0 # (n_select_samples, Np)
        sample_exist = part_exist.sum(dim=-1) > 0 # (n_select_samples)
        cal_img_indexes = torch.nonzero(sample_exist == 1).squeeze(dim=1)   # The images without part location annotations are eliminated
        # n_cal_samples = cal_img_indexes.shape[0]
        cal_pred_bboxes, cal_part_locs, cal_part_exist = pred_bboxes[cal_img_indexes], part_locs[cal_img_indexes], part_exist[cal_img_indexes]  # (n_cal_samples, 4), (n_cal_samples, Np, 2), (n_cal_samples, Np)

        cal_pred_bboxes = cal_pred_bboxes[:, None]  # (n_cal_samples, 1, 4)
        cal_cond1, cal_cond2, cal_cond3, cal_cond4 = cal_part_locs[:, :, 0] - cal_pred_bboxes[:, :, 0] >= 0, \
                                                    cal_part_locs[:, :, 0] - cal_pred_bboxes[:, :, 1] <= 0, \
                                                    cal_part_locs[:, :, 1] - cal_pred_bboxes[:, :, 2] >= 0, \
                                                    cal_part_locs[:, :, 1] - cal_pred_bboxes[:, :, 3] <= 0  # Each one: (n_cal_samples, Np)
        cal_part_inside = torch.stack([cal_cond1, cal_cond2, cal_cond3, cal_cond4], dim=2).sum(dim=-1) == 4 # (n_cal_samples, Np), estimate whether the ground-truth part location is inside the prediction bbox
        cal_part_inside = cal_part_inside.sum(dim=-1) > 0   # (n_cal_samples,)

        loc_acc = cal_part_inside.sum(dim=0) / cal_part_inside.shape[0] # Calculate the ratio of images that the ground-truth part location is inside the prediction bbox

        all_loc_acc.append(loc_acc.item())
        all_attri_idx.append(attri_idx)
        all_num_samples.append(n_select_samples)
    all_loc_acc = np.array(all_loc_acc)
    all_attri_idx = np.array(all_attri_idx)
    all_num_samples = np.array(all_num_samples)
    mean_loc_acc = all_loc_acc.mean()

    return mean_loc_acc * 100, (all_loc_acc, all_attri_idx, all_num_samples)
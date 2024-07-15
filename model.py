import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.resnet_cub_features_all import resnet18_cub_features
from models.resnet_features_all import resnet18_features, resnet34_features, resnet50_features, resnet50_inat_features, resnet101_features, resnet152_features
from models.deit_features_all import deit_tiny_features, deit_small_features, deit_base_features
from util.local_parts import part_attributes_names, attributes_names, id_to_attributes
from util.rotate_tensor import multiple_rotate_all, mask_tensor

base_architecture_to_features = {'resnet18': resnet18_cub_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_inat_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'deit_tiny': deit_tiny_features,
                                 'deit_small': deit_small_features,
                                 'deit_base': deit_base_features,}

class NewNet(nn.Module):
    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 num_attributes=112,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(NewNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.epsilon = 1e-4

        self.prototype_activation_function = prototype_activation_function #log

        self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                    self.num_classes)
        self.num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('RES'):
            self.shallow_layer_idx = 0
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            self.shallow_layer_idx = 4
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        elif features_name.startswith('MYVISION'):
            self.shallow_layer_idx = 3
            first_add_on_layer_in_channels = self.features.embed_dim
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape),
                                              requires_grad=True)

        self.ones = nn.Parameter(torch.ones(self.prototype_shape),
                                 requires_grad=False)


        # Prototypes -> Attributes & Attributes -> Classes
        self.attributes_predictor = nn.Linear(self.num_prototypes, self.num_attributes)
        self.class_predictor = nn.Linear(self.num_attributes, self.num_classes)

        # Part Attributes
        self.concept_groups = [['forehead', 'eye', 'crown', 'beak'], ['belly', 'back', 'leg']]
        self.num_concept_groups = len(self.concept_groups)
        self.mask_a_groups = [torch.zeros(self.num_attributes,) for _  in range(self.num_concept_groups)]
        self.indexes_groups = []
        for grp_idx in range(self.num_concept_groups):
            for part_name in self.concept_groups[grp_idx]:
                self.mask_a_groups[grp_idx] += torch.FloatTensor(np.array(attributes_names) == part_name)
            self.indexes_groups.append(torch.nonzero(self.mask_a_groups[grp_idx] == 1).squeeze(dim=1).cuda())
        
        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):

        x, all_feas = self.features.forward_all(x)
        x = self.add_on_layers(x)

        return x, all_feas

    def _cosine_convolution(self, x):

        x = F.normalize(x,p=2,dim=1)
        now_prototype_vectors = F.normalize(self.prototype_vectors,p=2,dim=1)
        distances = F.conv2d(input=x, weight=now_prototype_vectors)
        distances = -distances

        return distances

    def _project2basis(self,x):

        now_prototype_vectors = F.normalize(self.prototype_vectors, p=2, dim=1)
        distances = F.conv2d(input=x, weight=now_prototype_vectors)
        return distances

    def prototype_distances(self, x):

        conv_features, all_feas = self.conv_features(x)
        cosine_distances = self._cosine_convolution(conv_features)
        project_distances = self._project2basis(conv_features)

        return project_distances, cosine_distances, all_feas

    def global_min_pooling(self,distances):

        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)

        return min_distances

    def global_max_pooling(self,distances):

        max_distances = F.max_pool2d(distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        max_distances = max_distances.view(-1, self.num_prototypes)

        return max_distances

    def get_ortho_loss(self,):
        cur_basis_matrix = torch.squeeze(self.prototype_vectors)
        subspace_basis_matrix = cur_basis_matrix.reshape(self.num_classes, self.num_prototypes_per_class, self.prototype_shape[1])
        subspace_basis_matrix_T = torch.transpose(subspace_basis_matrix,1,2)
        orth_operator = torch.matmul(subspace_basis_matrix, subspace_basis_matrix_T)
        I_operator = torch.eye(subspace_basis_matrix.size(1), subspace_basis_matrix.size(1)).cuda()
        difference_value = orth_operator - I_operator
        ortho_cost = torch.sum(torch.relu(torch.norm(difference_value,p=1,dim=[1,2]) - 0))

        return ortho_cost

    def get_CLA_loss(self, shallow_feas, deep_feas, scales=[1, 2, 3], consis_thresh=None):
        shallow_feas = shallow_feas.detach()
        bz, fea_len = shallow_feas.shape[0], shallow_feas.shape[-1]
        fea_size = int(fea_len ** (1/2))
        shallow_feas, deep_feas = shallow_feas.reshape(bz, -1, fea_size, fea_size), deep_feas.reshape(bz, -1, fea_size, fea_size)   # (bz, d, fea_size, fea_size)

        all_consis_cost = 0
        for scale in scales:
            cur_s_feas, cur_d_feas, cur_fea_size = [], [], fea_size - scale + 1
            for idx_w in range(cur_fea_size):
                for idx_h in range(cur_fea_size):
                    cur_s_feas.append(shallow_feas[:, :, idx_h : idx_h + scale, idx_w : idx_w + scale].permute(0, 2, 3, 1).flatten(start_dim=1))
                    cur_d_feas.append(deep_feas[:, :, idx_h : idx_h + scale, idx_w : idx_w + scale].permute(0, 2, 3, 1).flatten(start_dim=1))
            cur_s_feas, cur_d_feas = torch.stack(cur_s_feas, dim=1), torch.stack(cur_d_feas, dim=1)
            shallow_norm = cur_s_feas / cur_s_feas.norm(dim=-1, p=2).unsqueeze(dim=-1)
            shallow_simi = shallow_norm.bmm(shallow_norm.permute(0, 2, 1))   # (B, 49, 49)
            deep_norm = cur_d_feas / cur_d_feas.norm(dim=-1, p=2).unsqueeze(dim=-1)
            deep_simi = deep_norm.bmm(deep_norm.permute(0, 2, 1))    # (B, 49, 49)

            consis_cost = F.relu(torch.abs(deep_simi - shallow_simi) - consis_thresh).mean()
            all_consis_cost += consis_cost

        return all_consis_cost

    def get_CIA_loss(self, all_feas, bz, layer_idx=3):
        """
        all_feas : list<tensor>
        bz : batch size
        """
        cur_feas = all_feas[layer_idx]
        feas, feas_r1 = cur_feas[:bz], cur_feas[bz:2*bz]
        nfeas_r_all = multiple_rotate_all(feas, all_rotate_times=[1])

        feas_r_all = torch.cat([feas_r1], dim=0)
        nfeas_r_all = nfeas_r_all.detach()    # This works
        mse_cost = F.mse_loss(feas_r_all, nfeas_r_all)

        return mse_cost
        
    def get_PA_loss(self, proto_acts, corre_proto_num=10, cls_dis_thresh=2, sep_dis_thresh=2):
        B = proto_acts.shape[0]
        a_weight = self.attributes_predictor.weight
        a_highest_indexes = torch.argsort(a_weight, dim=1, descending=True) # (112, 2000)
        corre_proto_indexes = a_highest_indexes[:, :corre_proto_num]    # (112, corre_proto_num)
        
        fea_size = int(proto_acts.shape[-1] ** (1/2))
        proto_acts = proto_acts.reshape(B, -1, fea_size, fea_size).unsqueeze(dim=1).repeat(1, 112, 1, 1, 1)    # (B, 112, 2000, fea_h, fea_w)
        corre_proto_indexes = corre_proto_indexes[None, :, :, None, None].repeat(B, 1, 1, fea_size, fea_size) # (B, 112, 10, fea_h, fea_w)
        proto_acts = torch.gather(proto_acts, 2, corre_proto_indexes)   # (B, 112, 10, fea_h, fea_w)

        proto_acts_max, proto_acts_min = proto_acts.amax(dim=(3, 4), keepdim=True), proto_acts.amin(dim=(3, 4), keepdim=True)
        proto_acts = (proto_acts - proto_acts_min) / (proto_acts_max - proto_acts_min)  # (B, 112, 10, fea_h, fea_w)

        posx_values, posy_values = torch.arange(fea_size)[None, :, None].repeat(fea_size, 1, 1).cuda(), \
                                    torch.arange(fea_size)[:, None, None].repeat(1, fea_size, 1).cuda() # (fea_h, fea_w, 1), (fea_h, fea_w, 1)
        pos_values = torch.cat([posx_values, posy_values], dim=-1)[None, None, None, :]  # (1, 1, 1, fea_h, fea_w, 2)
        pos_weights = proto_acts.unsqueeze(dim=-1) # (B, 112, 10, fea_h, fea_w, 1)
        pos_weights = pos_weights / pos_weights.sum(dim=(3, 4), keepdim=True)
        pos_centers = pos_weights.mul(pos_values).sum(dim=(3, 4))  # (B, 112, 10, 2)

        # Concept Groups
        group1_indexes = self.indexes_groups[0][None, :, None, None].repeat(B, 1, corre_proto_num, 2)  # (B, n1, 10, 2)
        group1_pcs = torch.gather(pos_centers, 1, group1_indexes) # (B, n1, 10, 2)
        group1_pcs = group1_pcs.reshape(B, -1, 2)   # (B, n1, 2)
        
        group2_indexes = self.indexes_groups[1][None, :, None, None].repeat(B, 1, corre_proto_num, 2)
        group2_pcs = torch.gather(pos_centers, 1, group2_indexes) # (B, n2, 10, 2)
        group2_pcs = group2_pcs.reshape(B, -1, 2)   # (B, n2 * 10, 2)
        
        # Group loss
        grp_dis = torch.abs(group1_pcs.unsqueeze(dim=2) - group1_pcs.unsqueeze(dim=1)).sum(dim=-1)
        grp_dis_cost = F.relu(grp_dis - cls_dis_thresh).mean()
        
        # Division loss
        group2_pcs = group2_pcs.detach()
        div_dis = torch.abs(group1_pcs.unsqueeze(dim=2) - group2_pcs.unsqueeze(dim=1)).sum(dim=-1)
        div_dis_cost = F.relu(sep_dis_thresh - div_dis).mean()

        return grp_dis_cost, div_dis_cost
        
    def forward(self, x):

        project_distances, cosine_distances, all_feas = self.prototype_distances(x)
        cosine_min_distances = self.global_min_pooling(cosine_distances)

        project_max_distances = self.global_max_pooling(project_distances)
        prototype_activations = project_max_distances
        logits = prototype_activations.reshape(-1, self.num_classes, self.num_prototypes_per_class).sum(dim=-1) # (B, 200)
        
        attributes_logits = self.attributes_predictor(prototype_activations)
        logits_attri = self.class_predictor(attributes_logits)

        fea_size = project_distances.shape[-1]
        project_distances = project_distances.flatten(start_dim=2)
        shallow_feas = all_feas[self.shallow_layer_idx]
        batch_size, dim, shallow_size = shallow_feas.shape[0], shallow_feas.shape[1], shallow_feas.shape[-1]
        shallow_feas = shallow_feas.reshape(batch_size, dim, fea_size, shallow_size // fea_size, fea_size, shallow_size // fea_size)
        shallow_feas = shallow_feas.permute(0, 1, 3, 5, 2, 4)   # (B, dim, 8, 8, 7, 7)
        shallow_feas = shallow_feas.reshape(batch_size, -1, fea_size, fea_size)
        shallow_feas = shallow_feas.flatten(start_dim=2)
        deep_feas = all_feas[-1].flatten(start_dim=2)

        return (logits, logits_attri, attributes_logits), (cosine_min_distances, project_distances, shallow_feas, deep_feas, all_feas)
    
    def push_forward(self, x):

        conv_output, _ = self.conv_features(x) #[batchsize, 128, 14, 14]

        distances = self._project2basis(conv_output)
        return conv_output, distances
    
    def push_forward_all(self, x):

        conv_output, all_feas = self.conv_features(x) #[batchsize, 128, 14, 14]

        distances = self._project2basis(conv_output)
        return conv_output, distances, all_feas

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def construct_CBMNet(base_architecture, pretrained=True, img_size=224,
                    prototype_shape=(2000, 128, 1, 1), num_classes=200,
                    prototype_activation_function='log',
                    add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    proto_layer_rf_info = None
                                                         
    return NewNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)
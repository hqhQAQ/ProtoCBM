import os
import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
from tqdm import tqdm

import model
from util.utils import str2bool
from util.preprocess import mean, std
from util.datasets import Cub2011Eval
from util.local_parts import id_to_attributes, attributes_names, part_attributes_names
from util.eval_concept_trustworthiness import evaluate_concept_trustworthiness

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', type=str, default='0')
parser.add_argument('--data_set', default='CUB2011',
    choices=['CUB2011A', 'CUB2011U', 'Car', 'Dogs', 'CUB2011'], type=str)
parser.add_argument('--data_path', default='datasets/cub200_cropped', type=str)
parser.add_argument('--imgclass', type=int, nargs=1)
parser.add_argument('--out_dir', type=str, default='output_view')
parser.add_argument('--batch_size', type=int, default=30)
parser.add_argument('--check_test', type=str2bool, default=False)

# Model
parser.add_argument('--base_architecture', type=str, default='vgg16')
parser.add_argument('--input_size', default=224, type=int, help='images input size')
parser.add_argument('--prototype_shape', nargs='+', type=int, default=[2000, 64, 1, 1])
parser.add_argument('--prototype_activation_function', type=str, default='log')
parser.add_argument('--add_on_layers_type', type=str, default='regular')

parser.add_argument('--resume', type=str)
args = parser.parse_args()

args.vis_image = True
args.vis_test = True

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

img_size = args.input_size
args.nb_classes = 200
device = torch.device('cuda')

# Load the model
args.test_batch_size = args.batch_size
check_test_accu = args.check_test
base_architecture = args.base_architecture

state_dir = 'test' if args.vis_test else 'train'
save_analysis_path = os.path.join(args.out_dir, base_architecture, state_dir)

ppnet = model.construct_CBMNet(base_architecture=args.base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=args.prototype_shape,
                              num_classes=args.nb_classes,
                              prototype_activation_function=args.prototype_activation_function,
                              add_on_layers_type=args.add_on_layers_type)
if args.resume:
    checkpoint = torch.load(args.resume, map_location='cpu')
ppnet.load_state_dict(checkpoint['model'], strict=False)

ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
ppnet.eval()

img_size = ppnet_multi.module.img_size
prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    normalize
])

test_dataset = Cub2011Eval(args.data_path, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=10, pin_memory=True, drop_last=False, shuffle=False)
loader = test_loader

# Get the averaged activation maps of all prototypes
all_proto_acts, all_targets, all_img_ids = [], [], []
with torch.no_grad():
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
fea_h, fea_w = all_proto_acts.shape[-2], all_proto_acts.shape[-1]

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

all_activation_maps = []    # all_activation_maps[i] : (n_select_samples, fea_h, fea_w) for the i-th attribute, the length is 112
for attri_idx in tqdm(range(n_attributes)):
    attri_labels = all_attributes[:, attri_idx]
    select_img_indexes = torch.nonzero(attri_labels == 1).squeeze(dim=1)   # Select all the test images containing this attribute
    cur_corre_proto_indexes = corre_proto_indexes[attri_idx]

    select_proto_acts = all_proto_acts[select_img_indexes][:, cur_corre_proto_indexes]  # (n_select_samples, corre_proto_num, fea_h, fea_w)
    activation_maps = select_proto_acts.mean(dim=1)    # (n_select_samples, fea_h, fea_w)
    all_activation_maps.append(activation_maps)

mean_loc_acc, (all_loc_acc, all_attri_idx, all_num_samples) = evaluate_concept_trustworthiness(all_activation_maps, all_img_ids, bbox_half_size=45)
attributes_names = np.array(attributes_names)
select_attribute_names = attributes_names[all_attri_idx]

np.set_printoptions(precision=2)
print('Mean Loc Accuracy : %.2f' % (mean_loc_acc))
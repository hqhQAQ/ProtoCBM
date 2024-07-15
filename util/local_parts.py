import os
import copy
import pickle
import numpy as np

def draw_point(img, point, bbox_size=10, color=(0, 0, 255)):
    img[point[1] - bbox_size // 2: point[1] + bbox_size // 2, point[0] - bbox_size // 2: point[0] + bbox_size // 2] = color

    return img

def in_bbox(loc, bbox):
    return loc[0] >= bbox[0] and loc[0] <= bbox[1] and loc[1] >= bbox[2] and loc[1] <= bbox[3]


data_root = 'datasets/CUB_200_2011'
seg_root = 'datasets/segmentations'
out_dir = 'output_view/parts'

img_txt = os.path.join(data_root, 'images.txt')
cls_txt = os.path.join(data_root, 'image_class_labels.txt')
bbox_txt = os.path.join(data_root, 'bounding_boxes.txt')
attributes_txt = os.path.join(data_root, 'attributes.txt')
image_attributes_txt = os.path.join(data_root, 'attributes', 'image_attribute_labels.txt')
train_txt = os.path.join(data_root, 'train_test_split.txt')
part_name_txt = os.path.join(data_root, 'parts', 'parts.txt')
part_cls_txt = os.path.join(data_root, 'parts', 'parts.txt')
part_loc_txt = os.path.join(data_root, 'parts', 'part_locs.txt')

id_to_path, id_to_name, name_to_id = {}, {}, {}
with open(img_txt, 'r') as f:
    img_lines = f.readlines()
for img_line in img_lines:
    img_id, img_path = int(img_line.split(' ')[0]), img_line.split(' ')[1][:-1]
    img_folder, img_name = img_path.split('/')[0], img_path.split('/')[1]
    id_to_path[img_id] = (img_folder, img_name)
    id_to_name[img_id] = img_name
    name_to_id[img_name] = img_id

id_to_bbox = {}
with open(bbox_txt, 'r') as f:
    bbox_lines = f.readlines()
for bbox_line in bbox_lines:
    cts = bbox_line.split(' ')
    img_id, bbox_x, bbox_y, bbox_width, bbox_height = int(cts[0]), int(cts[1].split('.')[0]), int(cts[2].split('.')[0]), int(cts[3].split('.')[0]), int(cts[4].split('.')[0])
    bbox_x2, bbox_y2 = bbox_x + bbox_width, bbox_y + bbox_height
    id_to_bbox[img_id] = (bbox_x, bbox_y, bbox_x2, bbox_y2)

id_to_cls = {}
cls_to_id = {}
with open(cls_txt, 'r') as f:
    cls_lines = f.readlines()
for cls_line in cls_lines:
    img_id, cls_id = int(cls_line.split(' ')[0]), int(cls_line.split(' ')[1]) - 1   # 0 -> 199
    if cls_id not in cls_to_id.keys():
        cls_to_id[cls_id] = []
    cls_to_id[cls_id].append(img_id)
    id_to_cls[img_id] = cls_id

id_to_train = {}
with open(train_txt, 'r') as f:
    train_lines = f.readlines()
for train_line in train_lines:
    img_id, is_train = int(train_line.split(' ')[0]), int(train_line.split(' ')[1][:-1])
    id_to_train[img_id] = is_train

part_id_to_part = {}
with open(part_cls_txt, 'r') as f:
    part_cls_lines = f.readlines()
for part_cls_line in part_cls_lines:
    id_len = len(part_cls_line.split(' ')[0])
    part_id, part_name = part_cls_line[:id_len], part_cls_line[id_len + 1:]
    part_id_to_part[part_id] = part_name

id_to_part_loc = {}
with open(part_loc_txt, 'r') as f:
    part_loc_lines = f.readlines()
for part_loc_line in part_loc_lines:
    content = part_loc_line.split(' ')
    img_id, part_id, loc_x, loc_y, visible = int(content[0]), int(content[1]), int(float(content[2])), int(float(content[3])), int(content[4])
    if img_id not in id_to_part_loc.keys():
        id_to_part_loc[img_id] = []
    if visible == 1:
        id_to_part_loc[img_id].append([part_id, loc_x, loc_y])

id_to_label, id_to_attributes = {}, {}
train_ids = []
CUB_PROCESSED_DIR = 'datasets/class_attr_data_10'
TRAIN_PKL = os.path.join(CUB_PROCESSED_DIR, 'train.pkl')
TEST_PKL = os.path.join(CUB_PROCESSED_DIR, 'test.pkl')
train_metadata = pickle.load(open(TRAIN_PKL, 'rb'))
test_metadata = pickle.load(open(TEST_PKL, 'rb'))
all_metadata = copy.deepcopy(train_metadata)
all_metadata.extend(test_metadata)
for meta_item in all_metadata:
    img_id = meta_item['id']
    label = meta_item['class_label']
    attributes = meta_item['attribute_label']
    id_to_label[img_id] = label
    id_to_attributes[img_id] = attributes
for meta_item in train_metadata:
    img_id = meta_item['id']
    train_ids.append(img_id)

# Get all the attributes for train images & Pos Weight
train_attributes = []
for img_id in train_ids:
    cur_attri = id_to_attributes[img_id]
    train_attributes.append(cur_attri)
train_attributes = np.stack(train_attributes)
num_train_imgs = train_attributes.shape[0]
train_pos_weights = (num_train_imgs - train_attributes.sum(axis=0)) / train_attributes.sum(axis=0)

# Get the attributes for each class
cls_to_attributes = {}
num_classes = 200
for img_id in id_to_attributes.keys():
    cur_attributes = id_to_attributes[img_id]
    cur_cls = id_to_cls[img_id]
    if cur_cls not in cls_to_attributes.keys():
        cls_to_attributes[cur_cls] = cur_attributes
    
# Read Part Names
part_names = []
with open(part_name_txt, 'r') as f:
    part_name_lines = f.readlines()
for part_name_line in part_name_lines:
    content = part_name_line.split(' ')
    part_id, part_name = int(content[0]), content[-1][:-1]
    # part_names.append((part_id, part_name))
    part_names.append(part_name)

# Read Attributes
attributes_names = []
with open(attributes_txt, 'r') as f:
    attributes_lines = f.readlines()
for attributes_line in attributes_lines:
    content = attributes_line.split(' ')
    attribute_id, attribute_content = int(content[0]), content[1]
    attribute_part = attribute_content.split('::')[0].split('_')[1]
    attributes_names.append((attribute_id, attribute_part))
attributes_indexes = [2, 5, 7, 8, 11, 15, 16, 21, 22, 24,   # Copied from the vanilla CBM
                        26, 30, 31, 36, 37, 39, 41, 45, 46, 51,
                        52, 54, 55, 57, 58, 60, 64, 65, 70, 71,
                        73, 76, 81, 85, 91, 92, 94, 100, 102, 107,
                        111, 112, 117, 118, 120, 126, 127, 132, 133, 135,
                        146, 150, 152, 153, 154, 158, 159, 164, 165, 169,
                        173, 179, 180, 182, 184, 188, 189, 194, 195, 197,
                        199, 203, 204, 209, 210, 212, 213, 214, 219, 221,
                        222, 226, 236, 237, 239, 240, 241, 243, 244, 245,
                        250, 254, 255, 260, 261, 263, 269, 275, 278, 284,
                        290, 293, 294, 295, 299, 300, 305, 306, 309, 310,
                        311, 312]
attributes_indexes = np.array(attributes_indexes) - 1
original_attributes_indexes = attributes_indexes + 1
attributes_names = [attributes_names[k][1] for k in attributes_indexes]
attributes_names = list(map(lambda x: x.replace('bill', 'beak'), attributes_names)) # Total 112 attributes

part_attributes_names = ['breast', 'forehead', 'nape', 'throat', 'eye', 'belly', 'back', 'leg', 'crown', 'beak']    # The names of part attributes
part_name_to_part_indexes = {}
for part_attri_name in part_attributes_names:
    part_indexes = np.nonzero(np.array(part_names) == part_attri_name)[0]
    part_name_to_part_indexes[part_attri_name] = part_indexes

part_name_to_attri_indexes = {}
for part_attri_name in part_attributes_names:
    attri_indexes = np.nonzero(np.array(attributes_names) == part_attri_name)[0]
    part_name_to_attri_indexes[part_attri_name] = attri_indexes
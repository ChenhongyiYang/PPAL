import os
import json
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='AL Dataset')
parser.add_argument('oracle-path', type=str, required=True, help='dataset root')
parser.add_argument('out-root', type=str, required=True, help='output json path')
parser.add_argument('n-diff', type=int, required=True, help='number of different initial set')
parser.add_argument('n-labeled', type=int, required=True, help='n labeled images')
parser.add_argument('dataset', choices=['coco', 'voc'], required=True, help='dataset type')
args = parser.parse_args()

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')


voc_classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
       'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
       'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')



def generate_active_learning_dataset(oracle_json, n_labeled_img, out_labeled_json, out_unlabeled_json, valid_classes):
    with open(oracle_json) as f:
        data = json.load(f)

    all_images = data['images']
    all_annotations = data['annotations']

    class_id2name = dict()
    class_name2id = dict()
    for c in data['categories']:
        class_id2name[c['id']] = c['name']
        class_name2id[c['name']] = c['id']

    inds = np.random.permutation(len(all_images))
    labeled_inds = inds[:n_labeled_img].tolist()
    unlabeled_inds = inds[n_labeled_img:].tolist()

    labeled_images = []
    labeled_img_ids = []
    labeled_annotations = []

    for ind in labeled_inds:
        labeled_images.append(all_images[ind])
        labeled_img_ids.append(all_images[ind]['id'])
    for ann in all_annotations:
        if (class_id2name[ann['category_id']] in valid_classes) and (ann['image_id'] in labeled_img_ids):
            labeled_annotations.append(ann)

    unlabeled_images = []
    for ind in unlabeled_inds:
        unlabeled_images.append(all_images[ind])

    out_labeled_data = dict(
        categories=data['categories'],
        images=labeled_images,
        annotations=labeled_annotations)

    out_unlabeled_data = dict(
        categories=data['categories'],
        images=unlabeled_images,
        annotations=[])

    with open(out_labeled_json, 'w') as fl:
        json.dump(out_labeled_data, fl)

    with open(out_unlabeled_json, 'w') as fu:
        json.dump(out_unlabeled_data, fu)

    print('------------------------------------------------------')
    print('Labeled data:')
    print('Output path: %s'%out_labeled_json)
    print('Number of images: %d'%len(labeled_images))
    print('Number of objects: %d'%len(labeled_annotations))

    print('Unlabeled data:')
    print('Output path: %s' % out_unlabeled_json)
    print('Number of images: %d' % len(unlabeled_images))
    print('------------------------------------------------------')


if __name__ == '__main__':
    if args.dataset == 'coco':
        valid_classes = CLASSES
    elif args.dataset == 'voc':
        valid_classes = voc_classes
    else:
        raise NotImplementedError

    N = args.n_diff
    for i in range(N):
        data_prefix = args.dataset + '_' + args.n_labeled
        generate_active_learning_dataset(
            oracle_json=args.oracle_path,
            n_labeled_img=args.n_labeled,
            out_labeled_json=os.path.join(args.out_root, data_prefix+'_labeled_%d.json'%(i+1)),
            out_unlabeled_json=os.path.join(args.out_root, data_prefix+'_unlabeled_%d.json'%(i+1)),
            valid_classes=valid_classes
        )

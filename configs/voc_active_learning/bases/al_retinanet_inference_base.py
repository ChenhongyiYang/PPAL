_base_ = "al_retinanet_base.py"
data_root = 'data/VOC0712/'

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


data = dict(
    test=dict(
        type='ALCocoDataset',
        img_prefix='data/VOC0712/images/',
        classes=CLASSES
    )
)


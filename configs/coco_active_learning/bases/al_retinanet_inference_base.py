_base_ = "al_retinanet_base.py"
data_root = 'data/coco/'
data = dict(
    test=dict(
        type='ALCocoDataset',
        img_prefix=data_root + 'train2017/',
    )
)


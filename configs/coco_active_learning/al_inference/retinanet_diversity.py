_base_ = "../bases/al_retinanet_inference_base.py"

model = dict(
    type='ALRetinaNet',
    bbox_head=dict(
        type='RetinaHeadFeat',
        total_images=0,  # placeholder
        max_det=100,
        feat_dim=256,
        output_path=''  # placeholder
    ),
)
data = dict(
    test=dict(ann_file=None)
)
unlabeled_data = ''

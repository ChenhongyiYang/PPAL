_base_ = "../bases/al_retinanet_base.py"

labeled_data = ''
unlabeled_data = ''

model = dict(
    bbox_head=dict(
        type='RetinaQualityEMAHead',
        base_momentum=0.99
    )
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        ann_file=None
    ),
)

evaluation=dict(interval=999999999, metric='bbox')
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[20])

runner = dict(type='EpochBasedRunner', max_epochs=26)
checkpoint_config = dict(interval=26, max_keep_ckpts=1, by_epoch=True)



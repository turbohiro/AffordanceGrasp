import os
from os.path import join as opj

exp_name = "DGCNN_ROTATE_SO3_Release"
work_dir = opj("./log/dgcnn", exp_name)
try:
    os.makedirs(work_dir)
except:
    print('Working Dir is already existed!')

scheduler = dict(
    type='cos',
    T_max=100,
    eta_min=1e-3
)

optimizer = dict(
    type='sgd',
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

model = dict(
    type='dgcnn',
    k=40,
    emb_dims=1024
)

training_cfg = dict(
    model=model,
    estimate=False,
    partial=True,
    rotate='z',  # z,so3
    semi=False,
    rotate_type=None,
    batch_size=30,#14
    epoch=200,
    seed=1,
    dropout=0.5,
    gpu='0,1', 
    workflow=dict(
        train=1,
        val=1
    )    
)

data = dict(
    data_root='/data/wchen/3D_affordance/partial',
    #category=['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
    #         'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
    #         'listen', 'wear', 'press', 'cut', 'stab']
    category=['grasp', 'contain', 'lift', 'openable',  'wrap_grasp', 'pourable','wear','press','cut']
)

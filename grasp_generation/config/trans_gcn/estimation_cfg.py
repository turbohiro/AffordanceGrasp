import os
from os.path import join as opj
from utils import PN2_BNMomentum, PN2_Scheduler

exp_name = "Trans_GCN_ESTIMATION_Release"
work_dir = opj("./log/trans_gcn", exp_name)
seed = 1
try:
    os.makedirs(work_dir)
except:
    print('Working Dir is already existed!')

scheduler = dict(
    #type='cos',
    #T_max=200,
    #eta_min=1e-3

    #type='lr_lambda',
    #lr_lambda=PN2_Scheduler(init_lr=0.0001, step=15,
    #                        decay_rate=0.5, min_lr=1e-6)

    type = 'step',
    step_size = 20
)

optimizer = dict(
    #type='adam',
#    lr=0.0001,
    #betas=(0.9, 0.999),
    #eps=1e-08,
    #weight_decay=1e-5

    type='sgd',
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

model = dict(
    type='trans_gcn',
    k=40,
    emb_dims=512
)

training_cfg = dict(
    model=model,
    estimate=True,
    partial=False,
    rotate='None',  # z,so3
    semi=False,
    rotate_type=None,
    batch_size=14,
    epoch=60,
    seed=1,
    dropout=0.5,
    gpu='0,1',
    workflow=dict(
        train=1,
        val=1
    )
)

data = dict(
    data_root='/data/wchen/3D_affordance/full-shape',
    #category=['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
    #         'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
    #         'listen', 'wear', 'press', 'cut', 'stab']
    category=['grasp', 'contain',  'openable', 'sittable','support', 'wrap_grasp', 'pourable', 'move', 'displaY']
    
)

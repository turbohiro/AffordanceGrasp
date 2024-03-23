import os
from os.path import join as opj
from utils import PN2_BNMomentum, PN2_Scheduler

exp_name = "PN_ESTIMATION_Release"
work_dir = opj("./log/pn", exp_name)
try:
    os.makedirs(work_dir)
except:
    print('Working Dir is already existed!')

scheduler = dict(
    type='lr_lambda',
    lr_lambda=PN2_Scheduler(init_lr=0.001, step=20,
                            decay_rate=0.5, min_lr=1e-6)
    #type = 'step',
    #step_size = 40

    ##
    #type = 'cos',
    #T_max = 200,
    #eta_min = 1e-4
)

optimizer = dict(
    type='adam',
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=1e-4
)

model = dict(
    type='pn',
    weights_init='pn2_init'
)

training_cfg = dict(
    model=model,
    estimate=True,
    partial=False,
    rotate='None',  # z,so3
    semi=False,
    rotate_type=None,
    batch_size=14,
    epoch=100,
    seed=1,
    dropout=0.5,
    gpu='0,1',
    workflow=dict(
        train=1,
        val=1
    ),
    bn_momentum=PN2_BNMomentum(origin_m=0.1, m_decay=0.5, step=20)
)

data = dict(
    data_root='/data/wchen/3D_affordance/full-shape',
    #category=['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
    #          'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
    #          'listen', 'wear', 'press', 'cut', 'stab']
    category=['grasp', 'contain',  'openable', 'sittable','support', 'wrap_grasp', 'pourable', 'move', 'displaY']

    
)

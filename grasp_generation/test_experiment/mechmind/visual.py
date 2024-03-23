import mayavi.mlab as mlab
import numpy as np
import trimesh
import pdb

a = np.load('test_experiment/mechmind/a.npy',allow_pickle=True)

pc = a[1]
pts_visual = a[0]
np.savetxt('test2.xyz',pc)
pdb.set_trace()
mlab.figure(bgcolor=(1, 1, 1))
tube_radius = 0.001
for ii in range(10):
    mlab.points3d(pc[:, 0],pc[:, 1],pc[:, 2],color=(0.1, 0.1, 1),scale_factor=0.01)
    mlab.plot3d(pts_visual[ii][:, 0],pts_visual[ii][:, 1],pts_visual[ii][:, 2],color=(0.5,0,1),tube_radius=tube_radius,opacity=1)
mlab.show()
This is affordance-based dataset of  simulated parallel-jaw grasps from ShapeNet objects. It was generated based on [ACONYM](https://sites.google.com/nvidia.com/graspdataset) dataset and Pybullet simulator.

# Requirements and  Installation (Following acronym_tools)
* Python3
* `python -m pip install -r requirements.txt`
* `python -m pip install -e .`

# Use Cases

### Visualize Grasps
Successful affordance-based grasps. Grasp markers are colored green/red based on whether the simulation result was a success/failure:

`acronym_visualize_grasps.py --mesh_root /mnt/mydrive/acronym-affordance/data/grasp_data2  /mnt/mydrive/acronym-affordance/data/grasp_data2/grasps/Mug_62634df2ad8f19b87d1b7935311a2ed0_0.02328042176991366.h5`

![alt text](https://github.com/turbohiro/AffordanceGrasp/blob/master/data/fig/1.png)
### Generate Random Scenes and Visualize Grasps

Successful affordance-based grasps based on a table(filtering those that are in collision):

`acronym_generate_scene.py --mesh_root /mnt/mydrive/acronym-affordance/data/grasp_data2 --objects /mnt/mydrive/acronym-affordance/data/grasp_data2/grasps/Mug_128ecbc10df5b05d96eaf1340564a4de_0.0017788254548529554.h5  /mnt/mydrive/acronym-affordance/data/grasp_data2/grasps/Mug_62634df2ad8f19b87d1b7935311a2ed0_0.02328042176991366.h5 --support /mnt/mydrive/acronym-affordance/data/grasp_data2/grasps/Table_99cf659ae2fe4b87b72437fd995483b_0.009700376721042367.h5 --show_grasps`

![alt text](https://github.com/turbohiro/AffordanceGrasp/blob/master/data/fig/scene.png)






# Citation
If you use the dataset please cite:
```
@inproceedings{chen2022learning,
  title={Learning 6-DoF Task-oriented Grasp Detection via Implicit Estimation and Visual Affordance},
  author={Chen, Wenkai and Liang, Hongzhuo and Chen, Zhaopeng and Sun, Fuchun and Zhang, Jianwei},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={762--769},
  year={2022},
  organization={IEEE}
}
```

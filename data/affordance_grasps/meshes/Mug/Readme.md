# Make affordance-based grasp label (Mug):
## To test grasp in simulator, We need to first transform the ShapeNet Objects into the same axis coordinate one by one, different objects represent different transformation.
# 1.
b7841572364fd9ce1249ffc39a0c3c0b.obj 
5b0c679eb8a2156c4314179664d18101.obj
5b515d0b0c6740d5c5fe7f65b41f3b19
10f6e09036350e92b3f21f1137c3c347
23fb2a2231263e261a9ac99425d3b306
46ed9dad0440c043d33646b0990bb4a
547b3284875a6ce535deb3b0c692a2a
24651c3767aa5089e19f4cee87249aca
62634df2ad8f19b87d1b7935311a2ed0
414772162ef70ec29109ad7f9c200d62
b6f30c63c946c286cf6897d8875cfd5e
df026976dc03197213ac44947b92716e
transform: R1 =  trimesh.transformations.euler_matrix(-math.pi/2,0,0,'rxyz')
	       R2 = trimesh.transformations.euler_matrix(0,-math.pi,0,'rxyz')
	       R = np.dot(R2,R1)

# 2.
1ea9ea99ac8ed233bf355ac8109b9988
40f9a6cc6b2c3b3a78060a3a3a55e18f
128ecbc10df5b05d96eaf1340564a4de
159e56c18906830278d8f8c02c47cde0
187859d3c3a2fd23f54e1b6f41fdd78a
71995893d717598c9de7b195ccfa970
b88bcf33f25c6cb15b4f129f868dedb
ba10400c108e5c3f54e1b6f41fdd78a
c51b79493419eccdc1584fff35347dc6
cf777e14ca2c7a19b4aad3cc5ce7ee8
d309d5f8038df4121198791a5b8655c
e79d807e1093c6174e716404e6ec3a5f
f1c5b9bb744afd96d6e1954365b10b52
transform: R1 =  trimesh.transformations.euler_matrix(0,math.pi/2,0,'rxyz')
           R2 = trimesh.transformations.euler_matrix(0,0,math.pi/2,'rxyz')
           R = np.dot(R2,R1)
# 3.
37f56901a07da69dac6b8e58caf61f95
85d5e7be548357baee0fa456543b8166
d0a3fdd33c7e1eb040bc4e38b9ba163e
d75af64aa166c24eacbe2257d0988c9c
ea33ad442b032208d778b73d04298f62
transform: R1 =  trimesh.transformations.euler_matrix(-math.pi/2,0,0,'rxyz')
           R2 = trimesh.transformations.euler_matrix(0,0,0,'rxyz')
           R = np.dot(R2,R1)
# 4.
57f73714cbc425e44ae022a8f6e258a7
7223820f07fd6b55e453535335057818
transform: R1 =  trimesh.transformations.euler_matrix(0,0,-math.pi/2,'rxyz')
           R2 = trimesh.transformations.euler_matrix(-math.pi/2,0,0,'rxyz')
           R = np.dot(R2,R1)


# new added object attributes:

data["grasps/transforms_adjust"]: new grasp pose correspoding with new object

data["grasps/good_grasp"]   :grasp affordance 
data["grasps/good_pour"]    :pour affordance
data["grasps/good_contain"] :contain affordance
data["grasps/good_wrap"]    :wrap affordance

data["grasps/bad_grasp"]
data["grasps/bad_pour"]
data["grasps/bad_contain"]
data["grasps/bad_wrap"]

test: 85d5e7be548357baee0fa456543b8166.obj
187859d3c3a2fd23f54e1b6f41fdd78a.obj
b6f30c63c946c286cf6897d8875cfd5e.obj



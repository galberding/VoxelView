"""
This script is used for visualizing the generated voxeldata.
"""
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from create_dataset_with_occs import qube_path, sphere_path, pen_path
import os
from main import generate_cloud_pen, generate_cloud_sphere, generate_cloud_cube, transform_cloud, cloud2voxel

def vis_qube():
    for root, dirs, files in os.walk(qube_path):
        print(dirs)
        for dir in dirs:
            name = os.path.join(root,dir, "sample.npz")
            print("Name: ", name)
            data = np.load(name)
            voxel = np.array(data["voxel"], dtype=np.float)
            print(voxel.shape)
            # voxel = generate_cloud_cube()
            # ma = np.random.choice([0, 1], size=(16, 16, 16), p=[0.99, 0.01])
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
            ax.voxels(voxel,edgecolor="k")
            plt.show()
            # print(ma)

        break

if __name__ == '__main__':
    vis_qube()
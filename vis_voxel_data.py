"""
This script is used for visualizing the generated voxeldata.
"""
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


im_out_path = "../out/pen/plots/"

def vis_qube(path):
    for root, dirs, files in os.walk(path):
        print(dirs)
        dirs = sorted(dirs)
        for dir in dirs:
            name = os.path.join(root,dir, "sample.npz")
            print("Name: ", name)
            data = np.load(name)
            voxel = np.array(data["voxel"], dtype=np.float)
            print(voxel)
            # voxel = generate_cloud_cube()
            # ma = np.random.choice([0, 1], size=(16, 16, 16), p=[0.99, 0.01])
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_aspect('equal')
            ax.voxels(voxel,edgecolor="k")
            ax.view_init(elev=0, azim=0)
            ax.dist =3
            ax.set_axis_off()
            # plt.show()
            fig.savefig(im_out_path + dir)
            # break
            # print(ma)

        break

def getImage(path):
    return OffsetImage(plt.imread(path), zoom=0.06)


def vis_latent_space(path, sample_path):
    samples = np.load(sample_path)
    count = 0
    fig, ax = plt.subplots()
    ax.scatter(samples[:, 0], samples[:, 1])
    for root, dirs, files in os.walk(path):
        files = sorted(files)
        print(files)
        for file in files:
            print(os.path.join(root,file))
            ab = AnnotationBbox(getImage(os.path.join(root,file)), (samples[count, 0], samples[count, 1]), frameon=False)
            ax.add_artist(ab)
            count += 1
    # fig.savefig("../out/pen/Latent_visualization")
    plt.show()
if __name__ == '__main__':
    # vis_qube("../data/dataset/pen/train")
    vis_latent_space(im_out_path, "../out/pen/latent_samples.npy")
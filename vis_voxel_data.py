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
from VoxelView.main import generate_cloud_pen, generate_cloud_sphere, generate_cloud_cube, transform_cloud, cloud2voxel


def vis_data(dir_in, dir_out):
    for root, dirs, files in os.walk(dir_in):
        print(dirs)
        dirs = sorted(dirs)
        for dir in dirs:
            name = os.path.join(root, dir, "sample.npz")
            print("Name: ", name)
            data = np.load(name)
            voxel = np.array(data["voxel"], dtype=np.float)
            points = data["points"]
            occ = data["occ"]
            transl = data["transl"]
            print(transl)
            print(points.shape)
            print(occ)
            occ_points = points[occ == 1]
            print(occ_points.shape)
            voxel_rec = cloud2voxel(occ_points,1.0, 32)
            # voxel = generate_cloud_cube()
            # ma = np.random.choice([0, 1], size=(16, 16, 16), p=[0.99, 0.01])
            # fig = plt.figure()
            fix, axes = plt.subplots(1,3,subplot_kw=dict(projection='3d'), figsize=(25,20))
            # ax = fig.gca(projection='3d')
            axes[0].set_aspect('equal')
            axes[1].set_aspect('equal')
            axes[2].set_aspect('equal')
            axes[0].voxels(voxel, edgecolor="k")
            axes[1].scatter(occ_points[:,0], occ_points[:,1], occ_points[:,2])
            axes[2].voxels(voxel_rec, edgecolor="k")
            # ax.view_init(elev=0, azim=0)
            # ax.dist = 3
            # ax.set_axis_off()
            plt.show()
            # fig.savefig(dir_out + dir)
            # break
            # print(ma)

        break


def getImage(path):
    return OffsetImage(plt.imread(path), zoom=0.07)


def vis_latent_space(path, sample_path):
    samples = np.load(sample_path)
    count = 0
    fig, ax = plt.subplots()
    ax.scatter(samples[:, 0], samples[:, 1])
    for root, dirs, files in os.walk(path):
        files = sorted(files)
        print(files)
        for file in files:
            print(os.path.join(root, file))
            ab = AnnotationBbox(getImage(os.path.join(root, file)), (samples[count, 0], samples[count, 1]),
                                frameon=False)
            ax.add_artist(ab)
            count += 1
    fig.savefig("../assets/Latent_visualization_spheres")
    plt.show()


if __name__ == '__main__':
    dir_out = "../out/sphere/plots/"
    dir_in = "dataset/train"
    vis_data(dir_in, dir_out)
    # vis_latent_space(dir_out, "../assets/latent_samples_spheres.npy")

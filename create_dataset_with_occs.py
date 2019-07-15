from main import generate_cloud_pen, generate_cloud_sphere, generate_cloud_cube, transform_cloud, cloud2voxel
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from time import gmtime, strftime
from multiprocessing import Pool

import matplotlib.pyplot as plt
import os
from pyquaternion import Quaternion
from scipy.spatial import Delaunay

voxelSpaceSize = 16
voxelRange = 2.0

# How many occ/points should be generated
point_samples = 100000

# Paths to datasets
qube_path = "dataset/qube/"
sphere_path = "dataset/sphere/"
pen_path = "dataset/pen/"

range_quat_cube = 10
range_quat_pen = 10 #* 1000


def create_dataset_dir():
    '''
    Check if necessary files are existent.
    If not the missing directories will be created.
    :return:
    '''
    if not os.path.exists(qube_path):
        os.makedirs(qube_path)
    if not os.path.exists(sphere_path):
        os.makedirs(sphere_path)
    if not os.path.exists(pen_path):
        os.makedirs(pen_path)


def gen_points(cloud, samples):
    '''
    Sample points from uniform distribution in unit qube and calculate the corresponding occupancies to the
    given pointcloud,
    :param cloud: Pointcloud of the specific shape
    :param samples: amount of points to generate
    :return: Tupel of: [0] points, [1] occ
    '''
    points = np.random.uniform([-1, -1, -1], [1, 1, 1], (samples, 3))

    if not isinstance(cloud, Delaunay):
        hull = Delaunay(cloud)
        print(hull.points.shape)
    else:
        hull = cloud
    occ = hull.find_simplex(points)
    occ[occ >= 0] = 1
    occ[occ < 0] = 0
    # shape_occ = np.ones((clound_arr.shape[0]))
    return points, occ


def safe_sample(points, occ, voxel, path):
    '''
    Safe data to a sample.npz file.
    The file will be stored in the given path.
    A new directory will be created which contains the sample npz file.
    :param points:
    :param occ:
    :param voxel:
    :param path:
    '''
    files = (len([name for name in os.listdir(path)]))
    print(files)
    safe_dir = path + str(files).zfill(5) + "/"
    if not os.path.exists(safe_dir):
        os.makedirs(safe_dir)
    np.savez(safe_dir + "sample.npz", points=points, occ=occ, voxel=voxel)


# Helper function to generate occs and points as well as returning the correct path to store the samples
def qube(cloud):
    points, occs = gen_points(cloud, point_samples)
    voxel = cloud2voxel(cloud, voxelRange, size=voxelSpaceSize)
    # print("done!")
    return (points, occs, voxel, qube_path)


def sphere(cloud):
    points, occs = gen_points(cloud, point_samples)
    voxel = cloud2voxel(cloud, voxelRange, size=voxelSpaceSize)
    # print("done!")
    return (points, occs, voxel, sphere_path)


def pen(cloud):
    points, occs = gen_points(cloud, point_samples)
    voxel = cloud2voxel(cloud, voxelRange, size=voxelSpaceSize)
    # print("done!")
    return (points, occs, voxel, pen_path)


# Parallelizes the generation of occupancies.
# This is done by testing 100000 if they are inside the convex hull of the previous generated Pintcloud
#  Subsequently the voxels will be generated from the pointcloud.
#  points, occupancies and the voxel data will all bestored in a singele .npz file
def gen_samples(generator, config_list):
    '''
    Used for parallel calculation of the occupancies.
    :param generator: helper methodname which calcupates occs (qube | sphere | pen)
    :param config_list: list of transformed pointclouds
    '''
    pool = Pool(4)
    results = pool.map(generator, config_list)
    pool.close()
    pool.join()
    for res in results:
        safe_sample(*res)


if __name__ == '__main__':

    create_dataset_dir()

    # qube
    # for shapeSize in [8, 9, 10, 11]:
    #     dimension = voxelRange * shapeSize / voxelSpaceSize
    #     cloud = generate_cloud_cube(size=voxelSpaceSize, dimension=dimension)
    #
    #     conf_list = []
    #     for i in range(range_quat_cube):
    #         quat = Quaternion.random()
    #         if i == 0: quat = Quaternion()  # first quat = 1 + 0i + 0j + 0k
    #         if i % 100 == 0: print(strftime("%H%M%S", gmtime()), 'cube', '\t', 'shapeSize', shapeSize, '\t', 'i', i,
    #                                '\t',
    #                                'quat', quat)
    #         transformed = transform_cloud(cloud, quat)
    #         conf_list.append(transformed)
    #     gen_samples(qube, conf_list)

    # # Sphere
    # transf_spheres = []
    # for k in range(7):
    #     for shapeSize in [8, 9, 10, 11, 12, 13, 14, 15, 16]:
    #         dimension = voxelRange * shapeSize / voxelSpaceSize
    #         cloud = generate_cloud_sphere(size=voxelSpaceSize, dimension=dimension)

    #         quat = Quaternion.random()
    #         cloud_transformed = transform_cloud(cloud, quat)  #

    #         # voxel = cloud2voxel(cloud_transformed, voxelRange, size=voxelSpaceSize)
    #         transf_spheres.append(cloud_transformed)
    # gen_samples(sphere, transf_spheres)

    # Pen
    for shapeSize in [13, 14, 15, 16]:
        dimension = voxelRange * shapeSize / voxelSpaceSize
        cloud = generate_cloud_pen(size=voxelSpaceSize, dimension=dimension)
        conf_list = []
        for i in range(range_quat_cube):
            quat = Quaternion.random()
            if i == 0: quat = Quaternion()  # first quat = 1 + 0i + 0j + 0k
            if i % 100 == 0: print(strftime("%H%M%S", gmtime()), 'cube', '\t', 'shapeSize', shapeSize, '\t', 'i', i,
                                   '\t',
                                   'quat', quat)
            transformed = transform_cloud(cloud, quat)
            conf_list.append(transformed)
        gen_samples(pen, conf_list)

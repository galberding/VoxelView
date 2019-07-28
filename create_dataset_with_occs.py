from VoxelView.main import generate_cloud_pen, generate_cloud_sphere, generate_cloud_cube, transform_cloud, cloud2voxel
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from time import gmtime, strftime
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import os
from pyquaternion import Quaternion
from scipy.spatial import Delaunay
import math

voxelSpaceSize = 16
voxelRange = 2.0

# How many occ/points should be generated
point_samples = 100000

working_path = "dataset"

# Paths to datasets
qube_path = "dataset/qube/"
sphere_path = "dataset/sphere/"
pen_path = "dataset/pen/"


# range_quat_cube = 45
range_quat_cube = 5
range_quat_pen = 20 #* 10


def create_dataset_dir(path):
    '''
    Create the dataset for the specific datapoints.
    It will create directories for train and test.
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path+"train/"):
        os.makedirs(path+"train/")
    if not os.path.exists(path+"test/"):
        os.makedirs(path+"test/")


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
    if(voxel is None):
        return
    files = (len([name for name in os.listdir(path)]))
    print(files)
    safe_dir = path + str(files).zfill(5) + "/"
    if not os.path.exists(safe_dir):
        os.makedirs(safe_dir)
    np.savez(safe_dir + "sample.npz", points=points, occ=occ, voxel=voxel)


# Helper function to generate occs and points as well as returning the correct path to store the samples

def occ_generation(cloud):
    points, occs = gen_points(cloud, point_samples)
    voxel = cloud2voxel(cloud, voxelRange, size=voxelSpaceSize)
    return points, occs, voxel, working_path

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
    pool = Pool(cpu_count())
    results = pool.map(generator, config_list)
    pool.close()
    pool.join()
    for res in results:
        safe_sample(*res)

def qube_generation_process(samples):
    for shapeSize in [6, 7,8, 9]:
        dimension = voxelRange * shapeSize / voxelSpaceSize
        cloud = generate_cloud_cube(size=voxelSpaceSize, dimension=dimension)

        conf_list = []
        for i in range(samples):
            quat = Quaternion.random()
            if i == 0: quat = Quaternion()  # first quat = 1 + 0i + 0j + 0k
            if i % 100 == 0: print(strftime("%H%M%S", gmtime()), 'cube', '\t', 'shapeSize', shapeSize, '\t', 'i', i,
                                   '\t',
                                   'quat', quat)
            transformed = transform_cloud(cloud, quat)
            conf_list.append(transformed)
        gen_samples(occ_generation, conf_list)


def sphere_generation_process(samples):
    transf_spheres = []
    for k in range(samples):
        for shapeSize in [2,4,6,8]:
            dimension = voxelRange * shapeSize / voxelSpaceSize
            cloud = generate_cloud_sphere(size=voxelSpaceSize, dimension=dimension)

            quat = Quaternion.random()
            cloud_transformed = transform_cloud(cloud, quat)  #

            # voxel = cloud2voxel(cloud_transformed, voxelRange, size=voxelSpaceSize)
            transf_spheres.append(cloud_transformed)
    gen_samples(occ_generation, transf_spheres)


def pen_generation_process(samples):
    for shapeSize in [8, 9, 10, 11]:
        dimension = voxelRange * shapeSize / voxelSpaceSize
        cloud = generate_cloud_pen(size=voxelSpaceSize, dimension=dimension)
        conf_list = []
        for i in range(samples):
            quat = Quaternion.random()
            if i == 0: quat = Quaternion()  # first quat = 1 + 0i + 0j + 0k
            if i % 100 == 0: print(strftime("%H%M%S", gmtime()), 'cube', '\t', 'shapeSize', shapeSize, '\t', 'i', i,
                                   '\t',
                                   'quat', quat)
            transformed = transform_cloud(cloud, quat)
            conf_list.append(transformed)
        gen_samples(occ_generation, conf_list)


def gen_dataset(voxel_model, path, samples, voxel_space_size=16, voxel_range=2.0):
    '''
    Generate the voxel qube dataset.
    :param voxel_model (String): qube | sphere | pen
    :param path (String): note: path needs to end with "/"
    :param samples (int): amount of (training) samples to generate (actual sample size will be 4*samples)
        The test set will be 33% as big as the train set.
    :return:
    '''
    global voxelSpaceSize
    voxelSpaceSize = voxel_space_size
    global voxelRange
    voxelRange = voxel_range
    create_dataset_dir(path)
    train_path = path+"train/"
    test_path = path+"test/"
    global working_path
    working_path = train_path
    if voxel_model in "qube":
        qube_generation_process(samples)
    elif voxel_model in "sphere":
        sphere_generation_process(samples)
    elif voxel_model in "pen":
        pen_generation_process(samples)

    working_path = test_path
    if voxel_model in "qube":
        qube_generation_process(math.ceil(samples * 0.33))
    elif voxel_model in "sphere":
        sphere_generation_process(math.ceil(samples * 0.33))
    elif voxel_model in "pen":
        pen_generation_process(math.ceil(samples * 0.33))


if __name__ == '__main__':
    gen_dataset("pen", "dataset/pen/", 3)

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
range_quat_pen = 20  # * 10


def create_dataset_dir(path):
    '''
    Create the dataset for the specific datapoints.
    It will create directories for train and test.
    :param path:
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + "train/"):
        os.makedirs(path + "train/")
    if not os.path.exists(path + "test/"):
        os.makedirs(path + "test/")


def generate_occs(cloud, samples):
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


def write_samples_to_file(points, occ, voxel, path, attr):
    '''
    Stores generated voxel/pointcloud, occs and attributes in a sample.npy file.
    The file will be stored in the given path.
    A new directory will be created which contains the sample npz file.
    :param points:
    :param occ:
    :param voxel:
    :param path:
    :param attr
    '''
    if (voxel is None):
        return
    files = (len([name for name in os.listdir(path)]))
    print(files)
    safe_dir = path + str(files).zfill(5) + "/"
    if not os.path.exists(safe_dir):
        os.makedirs(safe_dir)
    np.savez(safe_dir + "sample.npz", points=points, occ=occ, voxel=voxel, size=attr[0], transl=attr[1], yaw_pitch_roll=attr[2:])


# Helper function to generate occs and points as well as returning the correct path to store the samples

def voxel_generation(cloud):
    points, occs = generate_occs(cloud[0], point_samples)
    voxel = cloud2voxel(cloud[0], voxelRange, size=voxelSpaceSize)
    return points, occs, voxel, working_path, np.array(cloud[1])


def multiprocessing_voxel_generation(generator, config_list):
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
        write_samples_to_file(*res)

def pointcloud_generation(config):
    '''
    Create transformed pointcloud according to the config it is given.
    :param config: Tuple-like:  [0]: dimension of the pointcloud
                                [1]: pointcloud generation method
                                [2]: size
    :return: Transformed pointcloud with its attributes: size, translation, yaw, pitch, roll
    '''
    mean = [0, 0, 0]
    cov = [[0.015, 0, 0], [0, 0.015, 0], [0, 0, 0.015]]
    dimension = config[0]
    cloud = config[1](size=voxelSpaceSize, dimension=dimension)
    # cloud = generate_cloud_sphere(size=voxelSpaceSize, dimension=dimension)
    transl = np.random.uniform([-0.6, -0.6, -0.6], [0.6, 0.6, 0.6], (1,3))[0]
    # transl = np.random.multivariate_normal(mean, cov, 1)[0]
    quat = Quaternion.random()
    cloud_transformed = transform_cloud(cloud, quat, cube_position=transl)  #
    # config.append((dimension, quat, transl))
    # voxel = cloud2voxel(cloud_transformed, voxelRange, size=voxelSpaceSize)
    return cloud_transformed, (config[2], transl, *quat.yaw_pitch_roll)


def multiprocessing_pointcloud_generation(generator, config_list):
    pool = Pool(cpu_count())
    results = pool.map(generator, config_list)
    pool.close()
    pool.join()
    return results


def generation_process(samples, gen_meth, size):
    '''
    Generate configs dor pointcloud and voxel generation and coordinate the execution.
    :param samples: Amount of saqmples to generate per size
    :param gen_meth: generation method for pointcloud generation
    :param size: List of sizes for pointcloud generation
    :return:
    '''
    for shapeSize in size:
        config = []
        for k in range(samples):
            dimension = voxelRange * shapeSize / voxelSpaceSize
            config.append((dimension,gen_meth, shapeSize))
        transf_spheres = multiprocessing_pointcloud_generation(pointcloud_generation, config)
        multiprocessing_voxel_generation(voxel_generation, transf_spheres)


def gen_dataset(voxel_model, path, samples, voxel_space_size=32, voxel_range=1.0, pen_sizes=[6, 7, 8, 9, 10, 11],
                sphere_sizes=[8, 9, 10, 11, 12, 13], qube_sizes=[9,9,9,9,9,9,9,9,9,9]):
    '''
    Generate the voxel qube dataset.
    :param voxel_model (String): qube | sphere | pen
    :param path (String): note: path needs to end with "/"
    :param samples (int): amount of (training) samples to generate (actual sample size will be 4*samples)
        The test set will be 33% as big as the train set.
    :param pen_sizes:
    :param sphere_sizes:
    :param qube_sizes:
    :return: writes specified datasamples to given path, divided in train/test
    '''

    mode = {
        "qube": (generate_cloud_cube, qube_sizes),
        "sphere": (generate_cloud_sphere, sphere_sizes),
        "pen": (generate_cloud_pen, pen_sizes)
    }

    if voxel_model not in mode:
        print("Unknown Mode!")
        exit(1)

    global voxelSpaceSize
    voxelSpaceSize = voxel_space_size
    global voxelRange
    voxelRange = voxel_range
    # path += voxel_model + "/"
    create_dataset_dir(path)
    train_path = os.path.join(path, "train", '')
    test_path = os.path.join(path, "test", '')
    global working_path

    working_path = train_path
    generation_process(samples, *mode[voxel_model])

    working_path = test_path
    generation_process(math.ceil(samples * 0.01), *mode[voxel_model])


if __name__ == '__main__':
    # gen_dataset("pen", "../data/dataset/", 50, pen_sizes=[6, 7, 8, 9, 10, 11, 12, 13, 14])
    # gen_dataset("sphere", "../data/dataset/", 50, pen_sizes=[6, 7, 8, 9, 10, 11, 12, 13, 14])
    gen_dataset("qube", "dataset/", 100, qube_sizes=[9], voxel_space_size=32, voxel_range=1.0)

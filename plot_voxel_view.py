import pickle, os
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import  # This import registers the 3D projection, but is otherwise unused.

file = 'assets' + os.sep + 'voxel_shapes_len343_ss16_cs10_q0_shift3px.pgz'
# file = 'assets' + os.sep + 'voxel_shapes_len36_095525.pkl'
trg_plot_type = 'quat_sequence'  # ['quat_first_rotate', 'quat_sequence']

trg_im_save = True
trg_gif = True

if file.split('.')[-1] == 'pkl':
    with open(file, 'rb') as f:
        voxel_shapes = pickle.load(f)
elif file.split('.')[-1] == 'pgz':
    import gzip
    with gzip.open(file, 'r') as f:
        voxel_shapes = pickle.load(f)
    for k in voxel_shapes.keys():
        print(k, type(voxel_shapes[k]))

print('\n', 'matplotlib  >>  draw the first voxel_shape from:  ' + file,'\n')

if trg_plot_type == 'first_rotate':
    quat = voxel_shapes['quaternions'][0]
    voxels = voxel_shapes['voxelShapes'][0]>0
    print('quat = ', quat, '\n')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=[1, 0, 0, 0.3], edgecolor='k')
    ax.view_init(0, 181)
    plt.draw()
    plt.pause(.001)

    for angle in range(181, 270):
        if angle % 10 == 0: print('view', 0, angle)
        ax.view_init(0, angle)
        plt.draw()
        plt.pause(.001)

elif trg_plot_type == 'quat_sequence':
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.gca(projection='3d')
    for i in range(len(voxel_shapes['quaternions'])):
        quat = voxel_shapes['quaternions'][i]
        voxels = voxel_shapes['voxelShapes'][i] > 0
        print('quat = ', quat, '\n')
        ax.cla()
        ax.voxels(voxels, facecolors=[1, 0, 0, 0.3], edgecolor=[0, 0, 0, 0.5])
        # angles = np.linspace(0, 2, len(voxel_shapes['quaternions']))
        # plt.title('quat = Quaternion(axis=[1, 1, 1], angle=np.pi * ' + str(round(angles[i], 2)) + '\n' + str(quat))
        # ax.view_init(0, 181)
        plt.draw()
        plt.pause(.001)
        if trg_im_save:
            im = Image.fromarray(np.array(np.array(fig.canvas.renderer._renderer), dtype=np.uint8))
            if not os.path.exists('png'):
                os.makedirs('png')
            im.save('png' + os.sep + 'im' + str(i).zfill(4) + '.png')
    if trg_gif:
        # cmd = "cd png; ffmpeg -framerate 10 -pattern_type glob -i '*0.png' -vf format=rgb24 _output.gif"
        # cmd = "cd png; convert -loop 0 -delay 10 '*0.png' _output2.gif"
        cmd = "cd png; convert -delay 20 -loop 0 -dither None -colors 32 '*0.png' -fuzz '40%' -layers OptimizeFrame '_output.gif'"
        import subprocess
        subprocess.call(cmd, shell=True)

plt.pause(60)
print('EXIT')

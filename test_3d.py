import mcfd

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = 20
    y = 20
    z = 20

    img_tensor = mcfd.generate_tensor([x, y, z], cluster_count=5, cluster_radius=2, low=0, high=255, eps=15)
    # img_tensor = 255 - img_tensor

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    grid = np.indices((x, y, z))

    x_data = grid[0].flatten()
    y_data = grid[1].flatten()
    z_data = grid[2].flatten()
    color_data = img_tensor.flatten()

    mask = np.ma.masked_where(color_data == 0, color_data)

    x_data = np.ma.array(x_data, mask=np.logical_not(mask)).compressed()
    y_data = np.ma.array(y_data, mask=np.logical_not(mask)).compressed()
    z_data = np.ma.array(z_data, mask=np.logical_not(mask)).compressed()
    color_data = np.ma.array(color_data, mask=np.logical_not(mask)).compressed()

    ax.scatter3D(x_data, y_data, z_data, c=color_data, cmap='Greys')

    plt.show()

    # ax = plt.axes(projection='3d')
    #
    # # Data for a three-dimensional line
    # zline = np.linspace(0, 15, 1000)
    # xline = np.sin(zline)
    # yline = np.cos(zline)
    # ax.plot3D(xline, yline, zline, 'gray')
    #
    # # Data for three-dimensional scattered points
    # zdata = 15 * np.random.random(100)
    # xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
    # ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
    # ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    # plt.show()

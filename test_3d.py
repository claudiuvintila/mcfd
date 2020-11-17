import mcfd

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = 20
    y = 20
    z = 20

    img_tensor, clusters = mcfd.generate_tensor([x, y, z], cluster_count=5, cluster_radius=2, low=0, high=255, eps=15)
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

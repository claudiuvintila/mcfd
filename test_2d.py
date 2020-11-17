import mcfd

from PIL import Image


if __name__ == '__main__':
    img_tensor, clusters = mcfd.generate_tensor([1024, 1024], cluster_count=25, cluster_radius=30, low=0, high=255, eps=55)
    img_tensor = 255 - img_tensor
    print(img_tensor.shape)

    # Creates PIL image
    img = Image.fromarray(img_tensor)
    img.show()

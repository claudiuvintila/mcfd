import mcfd

from PIL import Image


if __name__ == '__main__':
    img_tensor = mcfd.generate_tensor([128], cluster_count=5, cluster_radius=5, low=0, high=255, eps=55)
    img_tensor = 255 - img_tensor
    print(img_tensor.shape)

    # Creates PIL image
    img = Image.fromarray(img_tensor)
    img.show()

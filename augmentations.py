
import torch
import torchvision.transforms as T


class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    def forward(self, images):
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images

class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size, scale):
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(size=size, scale=scale)
    def forward(self, images):
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        random_resized_crop = super(DeviceAgnosticRandomResizedCrop, self).forward
        augmented_images = [random_resized_crop(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images

class CustomGaussianBlur(T.GaussianBlur):
    # for doing guassian blur on single/batch image/of images - added by AB
    def __init__(self, kernel_size=(13,13), sigma=1):
        super().__init__(kernel_size=kernel_size, sigma=sigma)
    def forward(self, images):
        gaussianblur = super(CustomGaussianBlur, self).forward
        if len(images.shape) == 4: 
            B, C, H, W = images.shape
            blur1_images = [gaussianblur(img).unsqueeze(0) for img in images]
            print("the shape of one image after blur in batch mode: ",blur1_images[0].shape)
            blur1_images = torch.cat(blur1_images)
            return blur1_images
        elif len(images.shape) == 3:
            C, H, W = images.shape
            blur1_image = gaussianblur(images)
            return blur1_image
        else:
            raise AssertionError(f"images should be a batch of images,or a single image, but it has shape {images.shape}")


if __name__ == "__main__":
    """
    You can run this script to visualize the transformations, and verify that
    the augmentations are applied individually on each image of the batch.
    """
    from PIL import Image
    # Import skimage in here, so it is not necessary to install it unless you run this script
    from skimage import data
    
    # Initialize DeviceAgnosticRandomResizedCrop
    random_crop = DeviceAgnosticRandomResizedCrop(size=[256, 256], scale=[0.5, 1])
    blur = CustomGaussianBlur()
    # Create a batch with 2 astronaut images
    pil_image = Image.fromarray(data.astronaut())
    tensor_image = T.functional.to_tensor(pil_image).unsqueeze(0)
    images_batch = torch.cat([tensor_image, tensor_image])
    # Apply augmentation (individually on each of the 2 images)
    augmented_batch = random_crop(images_batch)
    # Convert to PIL images
    augmented_image_0 = T.functional.to_pil_image(augmented_batch[0])
    augmented_image_1 = T.functional.to_pil_image(augmented_batch[1])
    # Visualize the original image, as well as the two augmented ones
    pil_image.show()
    augmented_image_0.show()
    augmented_image_1.show()



    blurs = [CustomGaussianBlur(kernel_size=(13,13), sigma=x) for x in (1,3,20)]
    scaled_images = [blurs[i](images_batch) for i in range(3)]
    T.functional.to_pil_image(scaled_images[0][0].squeeze(0)).show()
    # passing one image
    blur1_image = blurs[0](tensor_image)
    T.functional.to_pil_image(blur1_image.squeeze(0)).show()
    #passing batch of images
    blur1_images = blurs[0](images_batch)
    T.functional.to_pil_image(blur1_images[0]).show()







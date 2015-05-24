import numpy as np
import skimage
import glob
import os
import skimage.io
import skimage.transform
import multiprocessing as mp



default_augmentation_params = {
    'zoom_range': (1 / 1.1, 1.1),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
    'do_flip': False,
    'allow_stretch': False,
}

def build_augmentation_transform(zoom=(1, 1), rotation=0, shear=0, translation=(0, 0), flip=False): 
    if flip:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom[0], 1/zoom[1]), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    return tform_augment

def fast_warp(img, tf, output_shape=(50, 50), mode='constant', order=1):
    """
    This wrapper function is faster than skimage.transform.warp
    """
    m = tf.params # tf._matrix is
    # fastwarp seems to corrupt image using int8 transforms
    # TODO further research here
    return skimage.transform._warps_cy._warp_fast(img, m, output_shape=output_shape, mode=mode, order=order)


def build_center_uncenter_transforms(image_shape):
    """
    These are used to ensure that zooming and rotation happens around the center of the image.
    Use these transforms to center and uncenter the image around such a transform.
    """
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2#.0 - 0.5 # need to swap rows and cols here apparently! confusing!
    tform_uncenter = skimage.transform.SimilarityTransform(translation=-center_shift)
    tform_center = skimage.transform.SimilarityTransform(translation=center_shift)
    return tform_center, tform_uncenter

def build_rescale_transform(downscale_factor, image_shape, target_shape):
    """
    estimating the correct rescaling transform is slow, so just use the
    downscale_factor to define a transform directly. This probably isn't 
    100% correct, but it shouldn't matter much in practice.
    """
    rows, cols = image_shape
    trows, tcols = target_shape
    tform_ds = skimage.transform.AffineTransform(scale=(downscale_factor, downscale_factor))

    #1,1
    
    # centering    
    shift_x = int(cols / (2 * downscale_factor) - tcols / 2)
    shift_y = int(rows / (2 * downscale_factor) - trows / 2)
    tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))
    return tform_shift_ds + tform_ds


class Random_perturbation_transform(object):
    """ decided to put it in class to retain the randomly generated values"""
    def __init__(self, zoom_range, rotation_range, shear_range, translation_range, do_flip=False, allow_stretch=False, rng=np.random):
        self.shift_x = int(rng.uniform(*translation_range))
        self.shift_y = int(rng.uniform(*translation_range))
        self.translation = (self.shift_x, self.shift_y)
        self.rotation = int(rng.uniform(*rotation_range))
        self.shear = int(rng.uniform(*shear_range))
        if do_flip:
            self.flip = (rng.randint(2) > 0) # flip half of the time
        else:
            self.flip = False
        # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

        log_zoom_range = [np.log(z) for z in zoom_range]
        if isinstance(allow_stretch, float):
            log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
            zoom = np.exp(rng.uniform(*log_zoom_range))
            stretch = np.exp(rng.uniform(*log_stretch_range))
            zoom_x = zoom * stretch
            zoom_y = zoom / stretch
        elif allow_stretch is True: # avoid bugs, f.e. when it is an integer
            zoom_x = np.exp(rng.uniform(*log_zoom_range))
            zoom_y = np.exp(rng.uniform(*log_zoom_range))
        else:
            zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))
        # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.
        self.zoom = (zoom_x, zoom_y)

    def random_perturbation_transform(self):
        return build_augmentation_transform(self.zoom, self.rotation, self.shear, self.translation, self.flip)


#not working properly. colors are not meshing well
def perturb_multiscale_new(img, scale_factors=[1.0], augmentation_params, target_shapes=[(200,200)], rng=np.random):
    """
    scale is a DOWNSCALING factor.
    """

    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    color_channels = [b,g,r]
    output = []

    #local, but globals
    rpt = Random_perturbation_transform(rng=rng, **augmentation_params)
    tform_center, tform_uncenter = build_center_uncenter_transforms(b.shape)

    for channel in color_channels:
        #set operations using b color channel (can use any here)
        tform_augment = rpt.random_perturbation_transform()
        tform_augment = tform_uncenter + tform_augment + tform_center # shift to center, augment, shift back (for the rotation/shearing)

        for scale, target_shape in zip(scale_factors, target_shapes):
            if isinstance(scale, skimage.transform.ProjectiveTransform):
                tform_rescale = scale
            else:
                tform_rescale = build_rescale_transform(scale, channel.shape, target_shape) # also does centering
            #the reason this fucks up is due 
            output.append(fast_warp(channel, tform_rescale + tform_augment, output_shape=target_shape, mode='constant'))#.astype('float32'))
    

    out = np.zeros((patch_sizes[0][0], patch_sizes[0][1], 3),dtype='float32')
    out[:,:,0] = output[0]
    out[:,:,1] = output[1]
    out[:,:,2] = output[2]
    return out 





if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    img = np.asarray(skimage.img_as_float(Image.open(os.getcwd()+'/data/train/09c25d76fc840b3a687b59d337e585f5/09c25d76fc840b3a687b59d337e585f5.png')))#,dtype='float32')
    sfs = [1.0]
    patch_sizes = [(200,200)]
    rng_aug = np.random
    patches = perturb_multiscale_new(img, sfs, default_augmentation_params, target_shapes=patch_sizes, rng=rng_aug)
    test = skimage.transform.rotate(img, 120, order=3)
    plt.subplot(120),plt.imshow(img),plt.title('original')
    plt.subplot(121),plt.imshow(test),plt.title('simple rotation')
    plt.subplot(122),plt.imshow(patches),plt.title('sander')

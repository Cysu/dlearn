import numpy as np
import skimage.transform as sktrans
import skimage.color as skcolor
from sklearn.preprocessing import MinMaxScaler, Binarizer


def translate(image, translation):
    trans = sktrans.AffineTransform(translation=translation)
    ret = sktrans.warp(image, trans.inverse)

    if image.dtype == np.uint8:
        ret = (ret * 255).astype(np.uint8)

    return ret


def resize(image, shape, keep_ratio=False):
    """Return the resized image in specific shape.

    Parameters
    ----------
    image : numpy.ndarray
        The image to be resized. Must be either 2D(grayscale) or 3D(RGB) image.
    shape : list/tuple of int
        The shape after resizing. (height, width).
    keep_ratio : False or str
        If False, then the image will be stretched after resizing, otherwise it
        must be either ``'height'`` or ``'width'`` and the original
        height/weight ratio will be reserved. If ``'height'``, the image will be
        scaled to the desired height. Extra columns will be either truncated or
        filled with zero. If ``'width'``, the image will be scaled to the
        desired width. Extra rows will be either truncated or filled with zero.

    """

    if image.ndim == 3:
        shape += (image.shape[2],)
    elif image.ndim != 2:
        raise ValueError("Invalid image dimension")

    if keep_ratio == False:
        ret = sktrans.resize(image, shape)
    elif keep_ratio == 'height':
        scale = shape[0] * 1.0 / image.shape[0]
        image = sktrans.rescale(image, scale)
        width = image.shape[1]

        if width >= shape[1]:
            l = (width - shape[1]) // 2
            if image.ndim == 3:
                ret = image[:, l:shape[1]+l,:]
            elif image.ndim == 2:
                ret = image[:, l:shape[1] + l]
        else:
            l = (shape[1] - width) // 2
            ret = np.zeros(shape)
            if image.ndim == 3:
                ret[:, l:width+l,:] = image
            elif image.ndim == 2:
                ret[:, l:width + l] = image
    elif keep_ratio == 'width':
        scale = shape[1] * 1.0 / image.shape[1]
        image = sktrans.rescale(image, scale)
        height = image.shape[0]

        if height >= shape[0]:
            l = (height - shape[0]) // 2
            if image.ndim == 3:
                ret = image[l:shape[0]+l,:,:]
            elif image.ndim == 2:
                ret = image[l:shape[0]+l,:]
        else:
            l = (shape[0] - height) // 2
            ret = np.zeros(shape)
            if image.ndim == 3:
                ret[l:height+l,:,:] = image
            elif image.ndim == 2:
                ret[l:height+l,:] = image
    else:
        raise ValueError("Invalid argument ``keep_ratio``")

    if image.dtype == np.uint8:
        ret = (ret * 255).astype(np.uint8)

    return ret


def subtract_luminance(rgbimg, mean_luminance=None):
    labimg = skcolor.rgb2lab(rgbimg)

    if mean_luminance is None:
        mean_luminance = np.mean(labimg[:,:, 0])

    labimg[:,:, 0] -= mean_luminance

    return labimg


def scale_per_channel(img, scale_range):
    h, w, c = img.shape
    img = img.reshape(h * w, c)

    scaler = MinMaxScaler(scale_range, copy=False)
    img = scaler.fit_transform(img)

    return img.reshape(h, w, c)


def binarize(img, threshold):
    binarizer = Binarizer(threshold, copy=False)
    return binarizer.fit_transform(img)


def images2mat(images):
    return np.asarray(map(lambda x: x.ravel(), images))

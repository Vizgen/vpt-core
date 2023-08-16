from typing import List, Union, Callable

import cv2
import numpy as np
from skimage import exposure

from vpt_core.image import Filter


def normalization_clahe(image: np.ndarray, clahe_params: dict) -> np.ndarray:
    """
    Normalizes contrast using a CLAHE filter, scales dynamic range, and returns uint8 image
    """
    if image.std() < 0.1:
        return image.astype(np.uint8)
    normalized = exposure.equalize_adapthist(image, **clahe_params)
    subtract = normalized - normalized.min()
    return (
        np.array((subtract / subtract.max()) * 255, dtype=np.uint8)
        if subtract.max() != 0
        else np.zeros(image.shape, dtype=np.uint8)
    )


def normalization_clahe_3d(
    image: Union[np.ndarray, List[np.ndarray]], clahe_params: dict
) -> Union[np.ndarray, List[np.ndarray]]:
    is_3d = clahe_params["image_dimensions"] == "3D"
    cur_args = dict(clahe_params)
    cur_args.pop("image_dimensions")
    if isinstance(image, list):
        if is_3d:
            if len(cur_args["kernel_size"]) == 2:
                cur_args["kernel_size"] = [3, *cur_args["kernel_size"]]
            res = normalization_clahe(np.array(image), cur_args)
            return [res_im for res_im in res]
        else:
            return [normalization_clahe(im, cur_args) for im in image]
    return normalization_clahe(image, cur_args)


def filter_iterative(
    image: Union[np.ndarray, List[np.ndarray]], filter_callback: Callable, *args, **kwargs
) -> Union[np.ndarray, List[np.ndarray]]:
    if isinstance(image, list):
        return [filter_callback(im, *args, **kwargs) for im in image]
    return filter_callback(image, *args, **kwargs)


def normalize(image: np.ndarray) -> np.ndarray:
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def create_normalization_filter(p: dict) -> Filter:
    norm_type = p.get("type", "default")
    if norm_type == "CLAHE":
        args = {
            "clip_limit": p.get("clip_limit", 0.01),
            "kernel_size": np.array(p.get("filter_size", [100, 100])),
            "image_dimensions": p.get("image_dimensions", "2D"),
        }
        if args["kernel_size"].ndim != 1:
            raise ValueError("Invalid filter parameter: kernel size should be a one-dimensional array")
        if len(args["kernel_size"]) < 2:
            raise ValueError("Invalid filter parameter: kernel size should consist of at least two elements")
        return lambda img: normalization_clahe_3d(img, args)
    elif norm_type == "default":
        return lambda img: filter_iterative(img, normalize)
    raise TypeError(f"unsupported normalization type {norm_type}")


def create_blur_filter(p: dict) -> Filter:
    sz = p.get("size", 5)
    blur_type = p.get("type", "average")
    if blur_type == "median":
        return lambda image: filter_iterative(image, cv2.medianBlur, ksize=sz)
    elif blur_type == "gaussian":
        return lambda image: filter_iterative(image, cv2.GaussianBlur, (sz, sz), 0)
    elif blur_type == "average":
        return lambda image: filter_iterative(image, cv2.blur, (sz, sz))
    raise TypeError(f"unsupported blur type {blur_type}")


def create_downsample_filter(p: dict) -> Filter:
    scale = p.get("scale", 2)
    return lambda image: filter_iterative(image, cv2.resize, (0, 0), fx=1 / scale, fy=1 / scale)


def _filter_from_merlin(s_image: np.ndarray, thr: int) -> np.ndarray:
    ret = s_image
    if s_image.dtype == np.uint16:
        ret = cv2.convertScaleAbs(s_image, alpha=(255.0 / 65535.0))
    ret = cv2.adaptiveThreshold(ret, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, thr, -1)
    ret = ret * s_image
    return ret


def create_merlin_watershed_filter(p: dict) -> Filter:
    threshold = int(255 * p.get("threshold", 0.5))
    return lambda image: filter_iterative(image, _filter_from_merlin, threshold)


def apply_sequence(
    image: Union[np.ndarray, List[np.ndarray]], filters: List[Filter]
) -> Union[np.ndarray, List[np.ndarray]]:
    for f in filters:
        if f is not None:
            image = f(image)
    return image

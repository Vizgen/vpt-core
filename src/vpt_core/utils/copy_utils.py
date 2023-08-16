from vpt_core.io.regex_tools import parse_images_str
from vpt_core.io.vzgfs import vzg_open, retrying_attempts


def _copy_between_filesystems(input_path: str, output_path: str):
    for attempt in retrying_attempts():
        with attempt:
            with vzg_open(input_path, "rb") as i, vzg_open(output_path, "wb") as o:
                data = i.read()
                o.write(data)


def _copy_regex_images(input_regex_path, output_dir, output_separator):
    images = parse_images_str(input_regex_path)
    images = images.images
    if len(images) == 0:
        raise FileNotFoundError(f"Images are not found by regex {input_regex_path}")
    for im in images:
        stain, z, img_path = im.channel, im.z_layer, im.full_path
        _copy_between_filesystems(img_path, output_separator.join([output_dir, f"mosaic_{stain}_z{z}.tif"]))

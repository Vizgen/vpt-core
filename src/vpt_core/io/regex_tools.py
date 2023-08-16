import re
from dataclasses import dataclass
from typing import Dict, List, Set

from vpt_core import log
from vpt_core.io.vzgfs import (
    filesystem_for_protocol,
    get_rasterio_environment,
    prefix_for_protocol,
    protocol_path_split,
    rasterio_open,
)

default_stain_re = r"(?P<stain>[\w|-]+)"
default_z_re = r"(?P<z>[0-9]+)"
default_images_re = f"mosaic_{default_stain_re}_z{default_z_re}.tif"

pattern_to_re: Dict[str, str] = {"stain": default_stain_re, "z": default_z_re}


@dataclass(frozen=True)
class ImagePath:
    channel: str
    z_layer: int
    full_path: str


@dataclass(frozen=True)
class RegexInfo:
    image_width: int
    image_height: int
    images: Set[ImagePath]


def get_paths_by_regex(regex: str) -> List[str]:
    protocol, regex = protocol_path_split(regex)
    fs = filesystem_for_protocol(protocol)
    parts = regex.split(fs.sep)

    for i in range(len(parts)):
        if "?" in parts[i]:
            break

    root = fs.sep.join(parts[:i])
    parts = parts[i:]

    def walk_req(rootdir, path_regex):
        if len(path_regex) == 0:
            return [rootdir]
        res = []

        try:
            _, dirs, files = next(fs.walk(rootdir, maxdepth=1))
        except StopIteration:
            return res

        if len(path_regex) == 1:
            for filename in files:
                if re.match(path_regex[0], filename):
                    res.append(filename)
            return res

        for dirname in dirs:
            if re.match(path_regex[0], dirname):
                walk_res = walk_req(fs.sep.join([rootdir, dirname]), path_regex[1:])
                for p in walk_res:
                    res.append(fs.sep.join([dirname, p]))

        return res

    paths = [fs.sep.join([root, path]) if root else path for path in walk_req(root, parts)]
    return [path for path in paths if fs.isfile(path)]


def default_for_dir(path: str) -> str:
    fs = filesystem_for_protocol(protocol_path_split(path)[0])
    if fs.isdir(path):
        if path.endswith(fs.sep):
            return path + default_images_re
        return path + fs.sep + default_images_re
    return path


def unify_path_re_str(path: str) -> str:
    path = path.replace(r"\\", "/")
    inside = False

    def scan(a: str) -> str:
        nonlocal inside
        if not inside and a == "(":
            inside = True
        elif inside and a == ")":
            inside = False
        return "/" if not inside and a == "\\" else a

    return "".join([scan(x) for x in path])


def parse_images_str(images_str: str) -> RegexInfo:
    width, height = 0, 0
    images = set()
    images_str = unify_path_re_str(images_str)
    # apply patterns
    images_str = images_str.format(**pattern_to_re)
    # fix directory
    images_str = default_for_dir(images_str)
    paths = get_paths_by_regex(images_str)
    protocol, regex = protocol_path_split(images_str)
    fs = filesystem_for_protocol(protocol)

    if "?P<stain>" not in regex or "?P<z>" not in regex:
        raise ValueError('Bad regular expression: named group "z" or "stain" is missed')

    for path in paths:
        match = re.match(regex, path)
        if match is None:
            log.warning(f"Extracted path {path} does not match the input regular expression")
            continue

        try:
            z = int(match.group("z"))
            stain = match.group("stain")
        except IndexError:
            log.warning(
                f'Regular expression should contain groups "z" and "stain". Path {path} has not that groups '
                f"and will be skipped."
            )
            continue

        full_path = prefix_for_protocol(protocol) + fs.info(path)["name"]

        with get_rasterio_environment(full_path):
            with rasterio_open(full_path) as tif:
                im_height, im_width = tif.height, tif.width
                if width and height and (im_width != width or im_height != height):
                    raise ValueError("Images sizes are not equal")
                width = im_width
                height = im_height

        images.add(ImagePath(stain, z, full_path))

    return RegexInfo(width, height, images)

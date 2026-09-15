import glob
import math
import os

import cv2
import numpy as np
import rawpy
import colour
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import resize

import datetime


EXR_OPTIONS = [
    cv2.IMWRITE_EXR_TYPE,
    cv2.IMWRITE_EXR_TYPE_HALF,
    cv2.IMWRITE_EXR_COMPRESSION,
    cv2.IMWRITE_EXR_COMPRESSION_PIZ,
]


def tonemap(img):
    return img / (img + 1.0)


def load_ldr(ldr_file: str) -> np.ndarray:
    # load the image from the path
    if ldr_file.endswith(".exr") or ldr_file.endswith(".hdr"):
        ldr = cv2.imread(ldr_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB).clip(0, 1)
    else:
        ldr = (
            cv2.imread(
                ldr_file,
            ).astype(np.float32)
            / 255
        )
        ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
        if ldr.shape[2] == 4:
            ldr = ldr[:, :, :3]

        ldr = ldr.clip(0, 1)  # clip before linearization

        # inverse gamma correction
        ldr = np.float32(colour.models.eotf_sRGB(ldr))
        ldr /= ldr.max()
    return ldr


def load_raw(raw_file: str, random_mean=False) -> np.ndarray:
    # load the image from the path
    if raw_file.endswith(".exr") or raw_file.endswith(".hdr"):
        raw = np.float32(
            cv2.imread(raw_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        )
        hdr = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    else:
        raw = rawpy.imread(raw_file)
        hdr = np.float32(
            raw.postprocess(gamma=(1, 1), use_camera_wb=True, output_bps=16)
        )

    # normalize
    # set median value to 0.5, assume median is supposed to be properly exposed
    mval = 0.5
    if random_mean:
        mvals = [0.25, 0.5, 0.75]
        mval = np.random.choice(mvals, 1, p=[0.25, 0.5, 0.25])
    hdr_mapped = np.clip(hdr * mval / np.median(hdr), 0, None)
    return hdr_mapped


def round_32(x):
    return 32 * math.ceil(x / 32)


def to2np(t):
    return t.detach().cpu().squeeze(0).numpy().transpose(1, 2, 0)


def current_timestamp():
    return datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")


def get_raw_img_lists(dataset_root, dataset_names):
    raw_image_list = []

    for dn in dataset_names:
        if dn == "multilum":
            raw_image_path = "multilum/raw/*/*.arw"
        elif dn == "RAISE":
            raw_image_path = "RAISE/raw/*.NEF"
        elif dn == "fivek":
            raw_image_path = "fivek_dataset/raw_photos/*/photos/*.dng"
        elif dn == "LSMI-galaxy":
            raw_image_path = "LSMI/galaxy/*/*.dng"
        elif dn == "LSMI-nikon":
            raw_image_path = "LSMI/nikon/*/*.nef"
        elif dn == "LSMI-sony":
            raw_image_path = "LSMI/sony/*/*.arw"
        elif dn == "SID":
            raw_image_path = "SID/Sony/long/*.ARW"
        elif dn == "multiRAW-huawei_p30pro":
            raw_image_path = "multiRAW/huawei_p30pro/raw/*.dng"
        elif dn == "multiRAW-iphone_xsmax":
            raw_image_path = "multiRAW/iphone_xsmax/raw/*.DNG"
        elif dn == "multiRAW-oneplus_5t":
            raw_image_path = "multiRAW/oneplus_5t/raw/*.dng"
        elif dn == "ppr_cr2":
            raw_image_path = "ppr/raw/*.CR2"
        elif dn == "ppr_arw":
            raw_image_path = "ppr/raw/*.ARW"
        elif dn == "ppr_nef":
            raw_image_path = "ppr/raw/*.NEF"
        elif dn == "zoom_raw":
            raw_image_path = "zoom_raw/*/*/*.ARW"
        elif dn == "nikon_raw":
            raw_image_path = "nikon_raw/long/*.NEF"
        elif dn == "canon_raw":
            raw_image_path = "canon_raw/*/long/*.CR2"
        elif dn == "hdrplusdata":
            raw_image_path = "hdrplusdata/20171106/bursts/*/*.dng"
        elif dn == "hypersim":
            raw_image_path = "hypersim/ai_*/images/scene_cam_*_final_hdf5/*.exr"

        new_raw_image_list = get_img_lists(dataset_root, raw_image_path)

        raw_image_list += new_raw_image_list

    return raw_image_list


def get_img_lists(root, image_path):
    img_list = sorted(glob.glob(os.path.join(root, image_path)))
    return img_list


def add_chan(img):
    """Add an channel dimension to convert a gray-scale image
    to an RGB image

    params:
        img (numpy.array): numpy array image (H x W)

    returns:
        (numpy.array): the numpy image with an added channel dimension (H x W x C)
    """

    # if the provided image already has a channel dim, assume it's a grayscale
    # image and make it HxW by grabbing the first channel
    if len(img.shape) == 3:
        img = img[:, :, 0]

    return np.stack([img] * 3, -1)


# From chrislib: https://github.com/CCareaga/chrislib
def np_to_pil(img, bits=8):
    """Convert a [0-1] numpy array into a PIL image.

    params:
        img (numpy.array): the numpy array to convert to PIL Image
        bits (int) optional: desired bit-depth of output PIL Image (default 8)

    returns:
        (PIL.Image): the image converted to a PIL Image object
    """
    if bits == 8:
        int_img = (img * 255).astype(np.uint8)
    if bits == 16:
        int_img = (img * ((2**16) - 1)).astype(np.uint16)

    return Image.fromarray(int_img)


# From chrislib: https://github.com/CCareaga/chrislib
def pad_bb(bounding_box, amount=7):
    """Add padding to all elements of a PIL ImageDraw text bounding box.

    params:
        bounding_box (tuple): the bounding box to add padding to - must have four integer elements:
            left, top, right, bottom
        amount (int) optional: the amount of padding to add (default 7)

    returns:
        (tuple): bounding box tuple with padding added
    """
    return (
        bounding_box[0] - amount,
        bounding_box[1] - amount,
        bounding_box[2] + amount,
        bounding_box[3] + amount,
    )


# From chrislib: https://github.com/CCareaga/chrislib
def _tile_row(images, text, border, font, font_pos, font_color, text_box):
    """helper function to for tile_imgs and show that creates a single row
    of tiled images

    params:
        images (list of np.array): array of images to tile
        text (list of string): list of strings to draw on each image
        border (int): the size of border to add between each image
        font (TODO): font to use to render text on images
        font_pos (tuple): the (x,y) coordinates to anchor the text relative to top-left (2 integers)
        font_color (tuple): the RGB values for the desired color (3 integers)
        text_box (int): opacity of text box around text, None for no text box

    returns:
        concat (np.array): row of images as a single tiled image
    """
    assert len(images) == len(text)

    border_pix = np.ones((images[0].shape[0], border, 3))
    elements = [border_pix]

    for img, txt in zip(images, text):
        if len(img.shape) == 2:
            img = add_chan(img)

        if img.shape[-1] == 1:
            img = np.concatenate([img] * 3, -1)

        if txt != "" and txt is not None:
            pil_img = np_to_pil(img)
            draw = ImageDraw.Draw(pil_img, "RGBA")

            txtbb = draw.textbbox(font_pos, txt, font=font)

            if text_box is not None:
                draw.rectangle(pad_bb(txtbb), fill=(0, 0, 0, text_box))

            if font_color is None:
                imgbb = img[txtbb[1] : txtbb[3], txtbb[0] : txtbb[2], :]
                brightness = imgbb.mean(-1).mean()
                font_color = (0, 0, 0) if brightness > 0.5 else (255, 255, 255)

            draw.text(font_pos, txt, font_color, font=font)
            img = np.array(pil_img) / 255.0

        elements.append(img)
        elements.append(border_pix)

    concat = np.concatenate(elements, axis=1)
    return concat


# From chrislib: https://github.com/CCareaga/chrislib
def tile_imgs(
    images,
    rescale=1.0,
    text=None,
    font_size=16,
    font_file="/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    font_color=None,
    font_pos=(0, 0),
    text_box=None,
    display=False,
    save=None,
    border=20,
    cmap="viridis",
    quality=75,
):
    """Tile a set of numpy images into a grid with text and spaces in between.

    params:
        images (list of np.array): images to tile, can be 1D or 2D array (must be rectangular grid)
        rescale (float) optional: the desired amount to rescale the tiled images (default 1.0)
        text (array-like) optional: text to put on each image, must be same shape as image array (default None)
        font_size (int) optional: the desired size of the font (default 16)
        font_file (str) optional: a filename or path to a file containing a TrueType font
        font_color (tuple) optional: the RGB values for the desired color (3 integers) (default
            None)
        font_pos (tuple) optional: the (x,y) coordinates to anchor the text (2 integers) (default
            (0, 0))
        text_box (int) optional: opacity of text box around text, no text box if set to None (default None)
        display (bool) optional: whether to display the tiled images (uses matplotlib to show tiled images)
        save (str) optional: the filename or path to save the tiled images to. Use None to not save
            (default None)
        border (int) optional: the size of border to add to each image (default 20)
        cmap (str) optional: the colormap for matplotlib to use to map scalar data to colors
            (default "viridis")
        quality (int) optional: the image quality to save with. Minimum (worst) is 0, maximum
            (best) is 90 (default 75)

    returns:
        tiled (np.array): single image of the tiled image set
    """
    if not isinstance(images, list):
        print("expected list of images")
        return None

    # make 1d array in 2d to keep logic simple
    if not isinstance(images[0], list):
        images = [images]

    # if text is none make a 2d array like images to make things
    # easier, otherwise text should already be shaped like images
    if text is None:
        text = []
        for row in images:
            text.append([None] * len(row))
    else:
        if not isinstance(text[0], list):
            text = [text]

    try:
        font = ImageFont.truetype(font_file, font_size)
    except:
        font = None

    width = sum([border + x.shape[1] for x in images[0]]) + border
    border_pix = np.ones((border, width, 3))
    rows = [border_pix]

    for img_row, txt_row in zip(images, text):
        tiled_row = _tile_row(
            img_row, txt_row, border, font, font_pos, font_color, text_box
        )

        rows.append(tiled_row)
        rows.append(border_pix)

    tiled = np.concatenate(rows, axis=0)

    if rescale != 1.0:
        h, w, _ = tiled.shape
        tiled = resize(tiled, (int(h * rescale), int(w * rescale)))

    if display:
        plt.figure(figsize=(len(images[0]), len(images)))
        plt.imshow(tiled, cmap=cmap)
        plt.axis("off")

    if save:
        # TODO: save tiled image
        byte_img = (tiled * 255.0).astype(np.uint8)
        Image.fromarray(byte_img).save(save, quality=quality)
        # pass

    return tiled

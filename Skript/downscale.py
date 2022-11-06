import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.util import img_as_ubyte
import os
import sys
from tqdm import tqdm
import json

INPUTS = ("../Data/input/", "../Data/annotationen/")
OUTPUTS = ("../Data/scale4/input/", "../Data/scale4/annotationen/")


def rescale_anno(anno: dict, scale):
    new = {'size': (np.array(anno['size']) // scale).tolist(), 'tags': {}}
    for key, value in anno['tags'].items():
        new_value = []
        for polygon in value:
            new_value.append((np.array(polygon)//scale).tolist())
        new['tags'][key] = new_value

    return new


def downscale(scale=2):
    images = [f[:-4] for f in os.listdir(INPUTS[0]) if f.endswith(".tif")]

    for name in tqdm(images):
        image = io.imread(f"{INPUTS[0]}{name}.tif", as_gray=True)
        image = img_as_ubyte(image)

        with open(f'{INPUTS[1]}pc-{name}.json', 'r') as f:
            annotation = json.load(f)

        annotation = rescale_anno(annotation, scale=scale)

        image = resize(image, tuple(annotation['size']), anti_aliasing=False)

        io.imsave(f"{OUTPUTS[0]}{name}.tif", image)
        with open(f"{OUTPUTS[1]}pc-{name}.json", 'w') as outfile:
            json.dump(annotation, outfile)


if __name__ == '__main__':
    assert len(sys.argv) == 2, "function needs 1 argument."
    downscale(scale=int(sys.argv[1]))

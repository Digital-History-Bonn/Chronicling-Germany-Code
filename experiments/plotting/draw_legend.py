from src.news_seg.class_config import LABEL_NAMES
from src.news_seg.class_config import cmap_12 as cmap


import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb  # pylint: disable=no-name-in-module


# unique, counts = np.unique(img, return_counts=True)
# print(dict(zip(unique, counts)))
img = np.zeros((10,10))

values = LABEL_NAMES
for i in range(len(values)-1):
    img[-1][-(i + 1)] = i + 1
plt.imshow(label2rgb(img, bg_label=0, colors=cmap))
plt.figure(figsize=(10, 10))
plt.axis("off")
# create a patch (proxy artist) for every color
patches = [mpatches.Patch(color=cmap[i], label=f"{values[i]}") for i in range(11)]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, loc="center right", ncol=3)
plt.autoscale(tight=True)
plt.savefig("legend-img", bbox_inches=0, pad_inches=0, dpi=500)
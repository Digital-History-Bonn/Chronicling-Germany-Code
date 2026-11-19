"""Time the steps of the baseline detection stage (one process, one thread).

Runs `cgprocess.baseline_detection.predict.BaselineEngine` step by step on the
given pages and their layout PAGE XML (the text regions; existing text lines
are ignored, as in the stage itself) and writes one CSV row per step:

- ``load``: reading the image (``load_image``),
- ``prepare``: reading the layout, masking tables, halving the image,
- ``forward``: the U-Net on the GPU (CUDA synchronised), as in the stage, i.e.
  with autograd enabled (``BaselineEngine.predict`` does not use no_grad),
- ``forward_no_grad``: the same under ``torch.no_grad()`` -- not used by the
  stage, measured to see what it would save,
- ``maps``: softmax and scaling the four output maps back to full page size (CPU),
- ``region_mask``: per text region, copying the full-page maps and masking
  everything outside the region (``apply_polygon_mask(maps.clone(), roi)``,
  as in ``postprocess_per_region``),
- ``region_mask_crop``: the same result, but copying only the region's
  bounding box -- not used by the stage, measured to see what it would save;
  the script checks that both give identical tensors,
- ``region_parse``: finding the baselines in the region's maps (``parse``),
- ``region_shapely``: LineString/Polygon construction and simplification,
- ``write``: writing the PAGE XML (``add_baselines``).

Runs with the cgprocess installed in the environment (Kraken 5 env
``newspapers``); it imports nothing from the Kraken 7 OCR code.

Usage:
    python script/time_baseline_stages.py -i images/ -l layout_page/ -m baseline.pt -o out/ --pages 10
"""

import argparse
import csv
import os
import socket
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import LineString, Polygon
from skimage import draw
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from cgprocess.baseline_detection.predict import (
    BaselineEngine,
    apply_polygon_mask,
    extract_layout,
    preprocess_image,
)
from cgprocess.baseline_detection.utils import add_baselines, load_image
from cgprocess.shared.utils import enforce_image_limits


def apply_polygon_mask_crop(image: torch.Tensor, roi: np.ndarray):
    """apply_polygon_mask, but copies only the region's bounding box."""
    bounds = list(Polygon(roi).bounds)
    bounds = (
        enforce_image_limits(torch.tensor(bounds).reshape((2, 2)), (image.shape[2], image.shape[1]))
        .flatten()
        .tolist()
    )
    offset = np.array([bounds[0], bounds[1]], dtype=int)
    shape = int(bounds[3] - bounds[1]), int(bounds[2] - bounds[0])
    mask = draw.polygon2mask(shape, roi[:, ::-1] - offset[::-1])
    result = image[:, int(bounds[1]):int(bounds[3]), int(bounds[0]):int(bounds[2])].clone()
    result[:, mask == 0] = 0.0
    return result, offset


class Timer:
    def __init__(self, cuda: bool):
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-i", "--images", required=True, type=Path)
    parser.add_argument("-l", "--layout", required=True, type=Path, help="layout PAGE XML (text regions)")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-o", "--out", required=True, type=Path)
    parser.add_argument("--pages", type=int, default=10, help="first N pages (sorted by name); 0 = all")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device, -1 = CPU")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    annos = sorted(args.layout.glob("*.xml"))
    if args.pages:
        annos = annos[:args.pages]
    engine = BaselineEngine(model_name=args.model, cuda=args.cuda, thread_count=1)
    cuda = engine.device.type == "cuda"
    gpu = torch.cuda.get_device_name(engine.device) if cuda else "cpu"
    print(f"{len(annos)} pages, {gpu}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}, "
          f"torch threads={torch.get_num_threads()}")

    seconds = defaultdict(float)
    regions = lines = 0
    mismatches = 0
    to_grayscale = transforms.Grayscale(num_output_channels=3)
    with tempfile.TemporaryDirectory() as tmp:
        for anno in annos:
            image_path = args.images / f"{anno.stem}.jpg"
            with Timer(False) as t:
                image = load_image(str(image_path))
            seconds["load"] += t.seconds
            with Timer(False) as t:
                _, width, height = image.shape
                mask_regions, text_regions = extract_layout(str(anno))
                input_image = preprocess_image(image, mask_regions)
            seconds["prepare"] += t.seconds
            with Timer(cuda) as t:
                pred = engine.model(to_grayscale(input_image[None].to(engine.device))).cpu()
            seconds["forward"] += t.seconds
            del pred
            with Timer(cuda) as t, torch.no_grad():
                pred = engine.model(to_grayscale(input_image[None].to(engine.device))).cpu()
            seconds["forward_no_grad"] += t.seconds
            with Timer(False) as t:
                pred = pred.detach()
                maps = torch.stack([pred[0, 0], pred[0, 1],
                                    engine.softmax(pred[:, 2:4])[0, 1], engine.softmax(pred[:, 4:6])[0, 1]])
                maps = transforms.Resize((width, height), interpolation=InterpolationMode.NEAREST)(maps)
            seconds["maps"] += t.seconds

            baseline_lst, textline_lst = [], []
            for roi in text_regions:
                if len(roi) < 3:
                    continue
                regions += 1
                with Timer(False) as t:
                    mask_map, offset = apply_polygon_mask(maps.clone(), roi)
                seconds["region_mask"] += t.seconds
                with Timer(False) as t:
                    crop_map, crop_offset = apply_polygon_mask_crop(maps, roi)
                seconds["region_mask_crop"] += t.seconds
                if not (torch.equal(mask_map, crop_map) and np.array_equal(offset, crop_offset)):
                    mismatches += 1
                with Timer(False) as t:
                    b_list, _, t_list = engine.parse(mask_map.permute(1, 2, 0).numpy())
                seconds["region_parse"] += t.seconds
                with Timer(False) as t:
                    baseline_lst.append([LineString(line + offset).simplify(tolerance=1) for line in b_list])
                    textline_lst.append([Polygon(poly + offset).buffer(0).simplify(tolerance=1) for poly in t_list])
                seconds["region_shapely"] += t.seconds
                lines += len(b_list)
            with Timer(False) as t:
                add_baselines(layout_xml=str(anno), textlines=textline_lst, baselines=baseline_lst,
                              output_file=os.path.join(tmp, anno.name))
            seconds["write"] += t.seconds
            print(f"{anno.stem}: {len(text_regions)} regions, image {height}x{width}")

    alternatives = ("region_mask_crop", "forward_no_grad")  # measured, not part of the stage
    total = sum(v for k, v in seconds.items() if k not in alternatives)
    rows = [{"host": socket.gethostname(), "device": gpu, "step": step, "pages": len(annos),
             "regions": regions, "lines": lines, "seconds": round(s, 2),
             "s_per_page": round(s / max(len(annos), 1), 3),
             "share_pct": round(100 * s / total, 1) if step not in alternatives else "",
             "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "")}
            for step, s in seconds.items()]
    rows.append({**rows[0], "step": "total", "seconds": round(total, 2),
                 "s_per_page": round(total / max(len(annos), 1), 3), "share_pct": 100})
    for row in rows:
        print(row)
    print(f"region_mask vs region_mask_crop: {mismatches} of {regions} regions differ")
    csv_path = args.out / "baseline_times.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Written to {csv_path}")


if __name__ == "__main__":
    main()

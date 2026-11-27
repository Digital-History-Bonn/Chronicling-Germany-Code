"""Time the steps of Kraken line recognition, with Kraken 5.2.9 or 7.1.

Runs our OCR model on the text lines of the given PAGE XML files (ground-truth
or pipeline baselines) and writes one CSV row per measurement:

- ``extract``: cutting each line out of the page (``extract_polygons``, the
  new extractor, as ``predict.py`` uses it with ``no_legacy_polygons``),
- ``transform``: scaling and normalising the line image to the network input,
- ``forward_b1``: the network on one line at a time (what Kraken 5's rpred does),
- ``predict``: the full recognition call -- Kraken 5's ``rpred`` per text
  region, as ``predict.py`` on master; Kraken 7's ``RecognitionTaskModel``
  per page for every batch size / line-worker combination given.

GPU timings synchronise CUDA before reading the clock. Every ``predict`` run
also writes its text (line id -> prediction) to ``<out>/<label>.jsonl`` so the
two Kraken versions and settings can be compared and scored.

Usage:
    python script/time_ocr_stages.py -i images/ -l page/ -m ocr.mlmodel -o out/ \
        --pages 10 --device cuda:0 --batch-sizes 1 32 64 --line-workers 0 2
"""

# A timing script, not pipeline code: Kraken 5 and 7 need different (local)
# imports, the Kraken 7 forward pass is timed through its private members, and
# src/ is put on sys.path before the cgprocess imports.
# pylint: disable=import-outside-toplevel, protected-access, wrong-import-position
# pylint: disable=too-many-locals, too-many-statements, too-many-branches

import argparse
import csv
import dataclasses
import json
import os
import socket
import sys
import time
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

import torch
from bs4 import BeautifulSoup
from PIL import Image

import kraken
from kraken.containers import BaselineLine, Segmentation
from kraken.lib.dataset import ImageInputTransforms
from kraken.lib.segmentation import extract_polygons

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cgprocess.OCR.shared.utils import pad_image, pad_points  # noqa: E402
from cgprocess.shared.utils import xml_polygon_to_polygon_list  # noqa: E402

Page = Tuple[str, Image.Image, List[List[Any]]]


def kraken_version() -> str:
    """Installed Kraken version."""
    try:
        return package_version("kraken")
    except Exception:  # pylint: disable=broad-exception-caught
        return getattr(kraken, "__version__", "unknown")


def read_page(anno_path: Path) -> List[List[Any]]:
    """Lines with a baseline, per text region, as BaselineLine objects (keyword
    arguments: Kraken 5 and 7 order the positional ones differently)."""
    soup = BeautifulSoup(anno_path.read_text(encoding="utf-8"), "xml")
    regions = []
    for r_idx, region in enumerate(soup.find_all("TextRegion")):
        lines = []
        for l_idx, text_line in enumerate(region.find_all("TextLine")):
            baseline = text_line.find("Baseline")
            if not baseline:
                continue
            lines.append(BaselineLine(
                id=text_line.get("id") or f"{r_idx}_{l_idx}",
                baseline=xml_polygon_to_polygon_list(pad_points(baseline["points"])),
                boundary=xml_polygon_to_polygon_list(pad_points(text_line.find("Coords")["points"])),
            ))
        regions.append(lines)
    return regions


def segmentation(name: str, lines: List[Any]) -> Segmentation:
    """Kraken baseline segmentation of the given lines."""
    return Segmentation(type="baselines", imagename=name, text_direction="horizontal-lr",
                        script_detection=False, lines=lines, line_orders=[])


def sync(device: str) -> None:
    """Wait for the GPU, so that the clock measures finished work."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


class Timer:
    """Context manager measuring wall-clock seconds, CUDA synchronised on a GPU device."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.start = 0.0
        self.seconds = 0.0

    def __enter__(self) -> "Timer":
        sync(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        sync(self.device)
        self.seconds = time.perf_counter() - self.start


def load_kraken5(model_path: str, device: str) -> Tuple[Any, Any]:
    """Kraken 5 recognition network and its input shape."""
    from kraken.lib import models
    net = models.load_any(model_path, device=device)
    return net, net.nn.input


def load_kraken7(model_path: str, device: str, batch_size: int, line_workers: int) -> Tuple[Any, Any]:
    """Kraken 7 recognition model and its inference config."""
    from kraken.configs import RecognitionInferenceConfig
    from kraken.ketos.util import to_ptl_device
    from kraken.tasks import RecognitionTaskModel
    model = RecognitionTaskModel.load_model(model_path)
    accelerator, devices = to_ptl_device(device)
    config = RecognitionInferenceConfig(accelerator=accelerator, device=devices, batch_size=batch_size,
                                        num_line_workers=line_workers, no_legacy_polygons=True)
    return model, config


def main() -> None:
    """Measure the recognition steps and append them to <out>/times.csv."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("-i", "--images", required=True, type=Path)
    parser.add_argument("-l", "--layout", required=True, type=Path, help="PAGE XML with baselines")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-o", "--out", required=True, type=Path)
    parser.add_argument("--pages", type=int, default=10, help="first N pages (sorted by name); 0 = all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 64], help="Kraken 7 only")
    parser.add_argument("--line-workers", type=int, nargs="+", default=[0, 2], help="Kraken 7 only")
    parser.add_argument("--label", default="", help="prefix for the run labels (e.g. the env name)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    version = kraken_version()
    major = int(version.split(".")[0])
    annos = sorted(args.layout.glob("*.xml"))
    if args.pages:
        annos = annos[:args.pages]
    pages: List[Page] = []
    for anno in annos:
        candidates = (args.images / f"{anno.stem}{ext}" for ext in (".jpg", ".png", ".tif"))
        image = next((p for p in candidates if p.exists()), None)
        if image is None:
            print(f"no image for {anno.stem}, skipped", file=sys.stderr)
            continue
        pages.append((anno.stem, pad_image(Image.open(image)), read_page(anno)))
    n_lines = sum(len(r) for _, _, regions in pages for r in regions)
    print(f"Kraken {version}, {args.device}, {len(pages)} pages, {n_lines} lines, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}, torch threads={torch.get_num_threads()}")

    rows: List[Dict[str, Any]] = []
    gpu = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "cpu"

    def row(step: str, seconds: float, lines: int, batch: Any = "", workers: Any = "") -> None:
        rows.append({"kraken": version, "host": socket.gethostname(), "device": gpu, "step": step,
                     "batch_size": batch, "line_workers": workers, "pages": len(pages), "lines": lines,
                     "seconds": round(seconds, 2), "s_per_line": round(seconds / max(lines, 1), 5),
                     "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "")})
        print(rows[-1])

    if major >= 7:
        model, config = load_kraken7(args.model, args.device, 1, 0)
        model.net.prepare_for_inference(config)
        net_input = model.net.input
    else:
        net, net_input = load_kraken5(args.model, args.device)
    b, c, h, w = net_input
    transforms = ImageInputTransforms(b, h, w, c, (16, 0), False)

    # --- extract, transform, forward one line at a time --------------------
    t_ext = t_tf = t_fwd = 0.0
    done = 0
    for name, im, regions in pages:
        seg = segmentation(name, [l for r in regions for l in r])
        for line in seg.lines:
            with Timer("cpu") as t:
                try:
                    box, _ = next(extract_polygons(im, dataclasses.replace(seg, lines=[line]), legacy=False))
                except Exception:  # pylint: disable=broad-exception-caught
                    box = None
            t_ext += t.seconds
            if box is None or 0 in box.size:
                continue
            with Timer("cpu") as t:
                ts = transforms(box)
            t_tf += t.seconds
            with Timer(args.device) as t:
                with torch.inference_mode():
                    if major >= 7:
                        device = model.net._fabric.device if hasattr(model.net, "_fabric") else args.device
                        x = ts.unsqueeze(0).to(device)
                        model.net._rec_predict(x, torch.LongTensor([ts.shape[2]]))
                    else:
                        net.predict(ts.unsqueeze(0))
            t_fwd += t.seconds
            done += 1
    row("extract", t_ext, n_lines)
    row("transform", t_tf, done)
    row("forward_b1", t_fwd, done)

    # --- full recognition calls --------------------------------------------
    def run_predict(label: str, recognise: Callable[[str, Image.Image, List[List[Any]]], Iterable[Any]]
                    ) -> Tuple[float, int]:
        texts: Dict[str, str] = {}
        with Timer(args.device) as t:
            for name, im, regions in pages:
                for rec in recognise(name, im, regions):
                    texts[f"{name}/{rec.id}"] = rec.prediction
        with open(args.out / f"{args.label}{label}.jsonl", "w", encoding="utf-8") as f:
            for key, text in texts.items():
                f.write(json.dumps({"line": key, "text": text}, ensure_ascii=False) + "\n")
        return t.seconds, len(texts)

    if major >= 7:
        for workers in args.line_workers:
            for batch in args.batch_sizes:
                model, config = load_kraken7(args.model, args.device, batch, workers)

                def recognise7(name: str, im: Image.Image, regions: List[List[Any]],
                               model: Any = model, config: Any = config) -> Iterable[Any]:
                    return model.predict(im, segmentation(name, [l for r in regions for l in r]), config)

                seconds, lines = run_predict(f"k{major}_b{batch}_w{workers}", recognise7)
                row("predict", seconds, lines, batch, workers)
    else:
        from kraken import rpred

        def recognise5(name: str, im: Image.Image, regions: List[List[Any]]) -> Iterable[Any]:
            for lines in regions:
                yield from rpred.rpred(net, im, segmentation(name, lines), no_legacy_polygons=True)

        seconds, lines = run_predict(f"k{major}_rpred", recognise5)
        row("predict", seconds, lines, 1, 0)

    csv_path = args.out / "times.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            writer.writeheader()
        writer.writerows(rows)
    print(f"Written to {csv_path}")


if __name__ == "__main__":
    main()

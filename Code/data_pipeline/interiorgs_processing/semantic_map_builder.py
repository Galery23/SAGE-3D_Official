import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import label as nd_label
from shapely.geometry import Point, Polygon


def format2(value):
    return f"{float(value):.2f}"


def normalize_label(label: str) -> str:
    """Normalize labels to lowercase snake_case."""
    return label.strip().lower().replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert InteriorGS annotations into 2D semantic maps."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Root directory of the InteriorGS dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Directory to store generated semantic maps.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing semantic map files.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Process at most this many scene folders (useful for quick tests).",
    )
    return parser.parse_args()


def build_semantic_maps(
    input_root: Path, output_root: Path, overwrite: bool, max_scenes: Optional[int]
) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    scene_dirs = sorted(p for p in input_root.iterdir() if p.is_dir())
    if max_scenes is not None:
        scene_dirs = scene_dirs[:max_scenes]
    if not scene_dirs:
        print(f"[WARN] No scene directories found under {input_root}")
        return

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        out_json = output_root / f"2D_Semantic_Map_{scene_name}_Complete.json"
        out_png = output_root / f"2D_Semantic_Map_{scene_name}_Complete.png"

        if out_json.exists() and not overwrite:
            print(f"[SKIP] {out_json} already exists. Use --overwrite to regenerate.")
            continue

        occ_json_path = scene_dir / "occupancy.json"
        labels_json_path = scene_dir / "labels.json"
        occ_png_path = scene_dir / "occupancy.png"
        if not (occ_json_path.is_file() and labels_json_path.is_file() and occ_png_path.is_file()):
            print(f"[MISSING] {scene_name} lacks occupancy.json / occupancy.png / labels.json.")
            continue

        with occ_json_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        scale = meta["scale"]
        x_min, y_min = meta["min"][:2]

        occ_img = Image.open(occ_png_path).convert("L")
        occupancy = np.array(occ_img)
        h, w = occupancy.shape

        pixels, counts = np.unique(occupancy.reshape(-1), return_counts=True)
        candidate_walls = [int(p) for p in pixels if 0 < p < 250]
        if candidate_walls:
            wall_value = int(
                candidate_walls[
                    np.argmax([counts[np.where(pixels == v)[0][0]] for v in candidate_walls])
                ]
            )
        else:
            wall_value = int(pixels[0])
        print(f"[{scene_name}] wall pixel value = {wall_value}")

        with labels_json_path.open("r", encoding="utf-8") as f:
            labels = json.load(f)

        predefined_classes = [
            "door",
            "window",
            "chair",
            "table",
            "sofa",
            "bed",
            "wardrobe",
            "plant",
            "floor",
            "wall",
            "ceiling",
        ]
        label2id = {cls: idx + 1 for idx, cls in enumerate(predefined_classes)}
        cur_max_id = len(label2id) + 1
        for obj in labels:
            lbl = obj["label"]
            if lbl not in label2id:
                label2id[lbl] = cur_max_id
                cur_max_id += 1

        visual_map = np.zeros((h, w), dtype=np.int32)
        result_list = []
        item_counters = defaultdict(int)

        for obj in labels:
            if "bounding_box" not in obj:
                continue
            label = obj["label"]
            cat_id = label2id[label]
            poly3d = obj["bounding_box"]
            z_values = [v["z"] for v in poly3d]
            min_z = min(z_values)
            max_z = max(z_values)
            height = max_z - min_z

            poly2d = [[v["x"], v["y"]] for v in poly3d[:4]]
            poly = Polygon(poly2d)
            xys = np.array(poly2d)
            min_x_pixel = int(np.floor((np.min(xys[:, 0]) - x_min) / scale))
            max_x_pixel = int(np.floor((np.max(xys[:, 0]) - x_min) / scale))
            min_y_pixel = int(np.floor((np.min(xys[:, 1]) - y_min) / scale))
            max_y_pixel = int(np.floor((np.max(xys[:, 1]) - y_min) / scale))
            min_x_pixel = np.clip(min_x_pixel, 0, w - 1)
            max_x_pixel = np.clip(max_x_pixel, 0, w - 1)
            min_y_pixel = np.clip(min_y_pixel, 0, h - 1)
            max_y_pixel = np.clip(max_y_pixel, 0, h - 1)

            mask = np.zeros((h, w), dtype=bool)
            for j in range(min_x_pixel, max_x_pixel + 1):
                for i in range(min_y_pixel, max_y_pixel + 1):
                    i_flip = h - 1 - i
                    j_flip = w - 1 - j
                    cx = x_min + (j + 0.5) * scale
                    cy = y_min + (i + 0.5) * scale
                    if poly.covers(Point(cx, cy)):
                        mask[i_flip, j_flip] = True
                        visual_map[i_flip, j_flip] = cat_id

            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            xmin_pix, xmax_pix = xs.min(), xs.max()
            ymin_pix, ymax_pix = ys.min(), ys.max()
            x_left = x_min + xmin_pix * scale
            x_right = x_min + (xmax_pix + 1) * scale
            y_bottom = y_min + ymin_pix * scale
            y_top = y_min + (ymax_pix + 1) * scale
            w_box = x_right - x_left
            h_box = y_top - y_bottom
            bbox_m = [format2(x_left), format2(y_bottom), format2(x_right), format2(y_top)]
            bbox_xywh_m = [format2(x_left), format2(y_bottom), format2(w_box), format2(h_box)]
            mask_coords_m = [
                [format2(y_min + (y + 0.5) * scale), format2(x_min + (x + 0.5) * scale)]
                for y, x in zip(ys, xs)
            ]
            item_counters[label] += 1
            item_id = f"{normalize_label(label)}_{item_counters[label]}"
            result_list.append(
                {
                    "category_id": int(cat_id),
                    "category_label": label,
                    "instance_id": obj.get("ins_id", ""),
                    "item_id": item_id,
                    "bbox_m": bbox_m,
                    "bbox_xywh_m": bbox_xywh_m,
                    "area": int(mask.sum()),
                    "height_m": format2(height),
                    "min_z_m": format2(min_z),
                    "max_z_m": format2(max_z),
                    "mask_coords_m": mask_coords_m,
                }
            )

        wall_cat_id = label2id["wall"]
        wall_mask = occupancy == wall_value
        wall_mask_flip = np.flipud(wall_mask)
        visual_map[wall_mask_flip] = wall_cat_id

        wall_label_mask, wall_count = nd_label(wall_mask_flip, structure=np.ones((3, 3), dtype=np.int32))
        for idx in range(1, wall_count + 1):
            block_mask = wall_label_mask == idx
            ys, xs = np.where(block_mask)
            if xs.size == 0 or ys.size == 0:
                continue
            xmin_pix, xmax_pix = xs.min(), xs.max()
            ymin_pix, ymax_pix = ys.min(), ys.max()
            x_left = x_min + xmin_pix * scale
            x_right = x_min + (xmax_pix + 1) * scale
            y_bottom = y_min + ymin_pix * scale
            y_top = y_min + (ymax_pix + 1) * scale
            w_box = x_right - x_left
            h_box = y_top - y_bottom
            bbox_m = [format2(x_left), format2(y_bottom), format2(x_right), format2(y_top)]
            bbox_xywh_m = [format2(x_left), format2(y_bottom), format2(w_box), format2(h_box)]
            mask_coords_m = [
                [format2(y_min + (y + 0.5) * scale), format2(x_min + (x + 0.5) * scale)]
                for y, x in zip(ys, xs)
            ]
            label = "wall"
            item_counters[label] += 1
            item_id = f"{normalize_label(label)}_{item_counters[label]}"
            result_list.append(
                {
                    "category_id": int(wall_cat_id),
                    "category_label": label,
                    "instance_id": f"wall_{idx}",
                    "item_id": item_id,
                    "bbox_m": bbox_m,
                    "bbox_xywh_m": bbox_xywh_m,
                    "area": int(block_mask.sum()),
                    "height_m": format2(3.0),
                    "min_z_m": format2(0.0),
                    "max_z_m": format2(3.0),
                    "mask_coords_m": mask_coords_m,
                }
            )

        unable_mask = occupancy == 0
        unable_mask_flip = np.flipud(unable_mask)
        labeled, num = nd_label(unable_mask_flip, structure=np.ones((3, 3)))
        print(f"[{scene_name}] detected {num} unable-area clusters")
        for idx in range(1, num + 1):
            block = labeled == idx
            area = block.sum()
            if area < 5:
                continue
            ys, xs = np.where(block)
            xmin_pix, xmax_pix = xs.min(), xs.max()
            ymin_pix, ymax_pix = ys.min(), ys.max()
            x_left = x_min + xmin_pix * scale
            x_right = x_min + (xmax_pix + 1) * scale
            y_bottom = y_min + ymin_pix * scale
            y_top = y_min + (ymax_pix + 1) * scale
            w_box = x_right - x_left
            h_box = y_top - y_bottom
            mask_coords_m = [
                [format2(y_min + (y + 0.5) * scale), format2(x_min + (x + 0.5) * scale)]
                for y, x in zip(ys, xs)
            ]
            label = "Unable Area"
            item_counters[label] += 1
            item_id = f"{normalize_label(label)}_{item_counters[label]}"
            result_list.append(
                {
                    "category_id": -1,
                    "category_label": label,
                    "instance_id": f"unable_area_{idx}",
                    "item_id": item_id,
                    "bbox_m": [
                        format2(x_left),
                        format2(y_bottom),
                        format2(x_right),
                        format2(y_top),
                    ],
                    "bbox_xywh_m": [
                        format2(x_left),
                        format2(y_bottom),
                        format2(w_box),
                        format2(h_box),
                    ],
                    "area": int(area),
                    "height_m": format2(0.0),
                    "min_z_m": format2(0.0),
                    "max_z_m": format2(0.0),
                    "mask_coords_m": mask_coords_m,
                }
            )

        with out_json.open("w", encoding="utf-8") as f:
            json.dump(result_list, f, indent=2)
        print(f"[WRITE] {out_json}")

        extent = [float(x_min), float(x_min) + w * scale, float(y_min), float(y_min) + h * scale]
        plt.figure(figsize=(12, 12))

        bg_color = (31 / 255, 119 / 255, 180 / 255, 1.0)  # deep blue
        bg_img = np.zeros((h, w, 4), dtype=float)
        bg_img[:, :] = bg_color
        plt.imshow(bg_img, origin="lower", extent=extent)

        overlay = np.zeros((h, w, 4), dtype=float)
        overlay[unable_mask_flip] = [1.0, 128 / 255, 128 / 255, 1.0]  # #FF8080
        overlay[wall_mask_flip] = [158 / 255, 218 / 255, 229 / 255, 0.8]  # light blue
        plt.imshow(overlay, origin="lower", extent=extent)

        plt.axis("off")
        plt.savefig(out_png, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"[WRITE] {out_png}")

    print("Semantic map batch generation finished.")


def main():
    args = parse_args()
    build_semantic_maps(
        args.input_root.expanduser(),
        args.output_root.expanduser(),
        args.overwrite,
        args.max_scenes,
    )


if __name__ == "__main__":
    main()
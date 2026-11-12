"""So sánh một ZIP export CVAT với gold label: metric tổng (giống scoring/score.py) + từng ảnh.

Usage:
  python3 scripts/compare.py <task> <submission.zip> [--gt DIR] [--overlay OUTDIR] [--json out.json]

  task      : easy_semantic | medium_instance | hard_panoptic | cp1_holes | ... (xem data/manifest.json)
  --gt      : thư mục gold chứa <task>/groundtruth/ (vd. ../tiers). Mặc định: data/<group>/<task>/groundtruth
  --overlay : (semantic) ghi ảnh lỗi từng ảnh: xanh = đúng, đỏ = sai class, vàng = bỏ sót, tím = tô lên vùng void

Cần numpy, pillow, pycocotools (requirements.txt). Gold label không được commit vào fork.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scoring"))
import lab_utils as L  # noqa: E402
from score import load_registry, resolve_task  # noqa: E402


def only(coco, stem):
    """COCO dict restricted to one image."""
    imgs = [im for im in coco["images"] if Path(im["file_name"]).stem == stem]
    ids = {im["id"] for im in imgs}
    return {**coco, "images": imgs, "annotations": [a for a in coco["annotations"] if a["image_id"] in ids]}


def semantic(task_dir, gt_dir, sub_zip, overlay_dir):
    classes = json.load(open(task_dir / "classes.json"))
    name2id = classes["trainid"]
    id2name = {v: k for k, v in name2id.items()}
    sub = L.load_semantic_submission(sub_zip, name2id)
    gt = L.load_semantic_gt(gt_dir, set(name2id.values()))
    check_overlap(sub, gt)
    overall = L.score_semantic(sub, gt, id2name)
    rows = []
    for stem in sorted(gt):
        r = L.score_semantic({stem: sub[stem]} if stem in sub else {}, {stem: gt[stem]}, id2name)
        conf = {}
        if stem in sub:
            g, s = gt[stem], sub[stem]
            for i, n in id2name.items():
                fn = (g == i) & (s != i)            # GT class i, predicted something else
                if fn.any():
                    conf[n] = {("unlabeled" if k == 255 else id2name.get(int(k), str(k))): int(c)
                               for k, c in zip(*np.unique(s[fn], return_counts=True))}
            extra = int(((s != 255) & (g == 255)).sum())
            if overlay_dir:
                write_overlay(task_dir, stem, g, s, overlay_dir)
        else:
            extra = 0
        rows.append({"image": stem, "mIoU": r["value"], "coverage": r["coverage"],
                     "per_class_iou": r["per_class_iou"], "missed_as": conf, "pixels_on_void": extra})
    return overall, rows


def write_overlay(task_dir, stem, g, s, out):
    img = next((task_dir / "images").glob(stem + ".*"))
    im = np.array(Image.open(img).convert("RGB").resize(g.shape[::-1])).astype(float) * 0.5
    colors = [((g == s) & (g != 255), (0, 200, 0)), ((g != s) & (g != 255) & (s != 255), (230, 0, 0)),
              ((g != 255) & (s == 255), (255, 220, 0)), ((g == 255) & (s != 255), (170, 0, 255))]
    for m, c in colors:
        im[m] = im[m] * 0.4 + np.array(c) * 0.6
    Path(out).mkdir(parents=True, exist_ok=True)
    Image.fromarray(im.astype(np.uint8)).save(Path(out) / f"{stem}_errors.png")


def instance(task_dir, gt_dir, sub_zip):
    names = set(json.load(open(task_dir / "classes.json"))["classes"])
    sub = L._find_coco_json(L._read_members(sub_zip))
    gt = json.load(open(gt_dir / "instances.json"))
    stems = lambda c: {Path(im["file_name"]).stem for im in c["images"]}  # noqa: E731
    check_overlap(stems(sub), stems(gt))
    overall = L.score_instance(sub, gt, names)
    rows = []
    for stem in sorted(stems(gt)):
        s1, g1 = only(sub, stem), only(gt, stem)
        r = L.score_instance(s1, g1, names)
        per_cls = {}
        for c in sorted(names):
            rc = L.score_instance(s1, g1, {c})
            if rc["gt_instances"] or rc["submitted_instances"]:
                per_cls[c] = {k: rc[k] for k in ("tp", "fp", "fn", "gt_instances", "submitted_instances")}
                per_cls[c]["mean_iou"] = rc["mean_matched_iou"]
        rows.append({"image": stem, "metric": r["value"], "mean_matched_iou": r["mean_matched_iou"],
                     "tp": r["tp"], "fp": r["fp"], "fn": r["fn"], "per_class": per_cls})
    return overall, rows


def panoptic(task_dir, gt_dir, sub_zip):
    classes = json.load(open(task_dir / "classes.json"))
    names, stuff = set(classes["classes"]), set(classes["stuff"])
    catid2name = {int(k): v for k, v in classes["catid2name"].items()}
    pred = L.build_pred_panoptic(L._find_coco_json(L._read_members(sub_zip)), names, stuff)
    gt = L.load_panoptic_gt(gt_dir / "panoptic.json", gt_dir / "png", catid2name, names)
    check_overlap(pred, gt)
    overall = L.score_panoptic(pred, gt, names)
    rows = []
    for stem in sorted(gt):
        r = L.score_panoptic({stem: pred[stem]} if stem in pred else {}, {stem: gt[stem]}, names)
        rows.append({"image": stem, "PQ": r["value"], "SQ": r["SQ"], "RQ": r["RQ"], "per_class": r["per_class"]})
    return overall, rows


def check_overlap(sub, gt):
    if not set(sub) & set(gt):
        raise SystemExit(f"ảnh trong ZIP không khớp gold.\n  ZIP:  {sorted(sub)}\n  gold: {sorted(gt)}")


def fmt(v):
    return "  n/a" if v is None else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("submission")
    ap.add_argument("--gt", type=Path, help="thư mục chứa <task>/groundtruth/")
    ap.add_argument("--overlay", type=Path, help="(semantic) thư mục ghi ảnh lỗi")
    ap.add_argument("--json", type=Path, dest="json_out")
    a = ap.parse_args()

    info, task_dir, _ = resolve_task(load_registry(), a.task)
    gt_dir = (a.gt / a.task / "groundtruth") if a.gt else task_dir / "groundtruth"
    if not gt_dir.exists():
        raise SystemExit(f"không thấy gold: {gt_dir}")
    t = info["type"]
    if t == "semantic":
        overall, rows = semantic(task_dir, gt_dir, a.submission, a.overlay)
        cap, floor = L.HUMAN_CAP, L.FLOOR
    elif t == "instance":
        overall, rows = instance(task_dir, gt_dir, a.submission)
        cap, floor = L.HUMAN_CAP, L.FLOOR
    else:
        overall, rows = panoptic(task_dir, gt_dir, a.submission)
        cap, floor = L.PQ_CAP, L.PQ_FLOOR
    points = L.metric_to_points(overall["value"], info.get("weight", 0), cap=cap, floor=floor)

    print(f"\n=== {a.task} ({t}) — {overall['metric']}: {overall['value']:.3f}   points {points} / {info.get('weight', 0)}")
    if t == "semantic":
        print(f"coverage {overall['coverage']*100:.1f}%   per-class IoU: "
              + "  ".join(f"{k} {fmt(v)}" for k, v in overall["per_class_iou"].items()))
        for r in rows:
            print(f"\n## {r['image']}   mIoU {r['mIoU']:.3f}   coverage {r['coverage']*100:.1f}%   "
                  f"pixels tô lên void {r['pixels_on_void']}")
            for c, v in r["per_class_iou"].items():
                if v is None:
                    continue
                miss = ", ".join(f"{k} {n}" for k, n in sorted(r["missed_as"].get(c, {}).items(), key=lambda x: -x[1]) if n >= 100)
                print(f"  {c:12s} IoU {v:.3f}" + (f"   GT {c} bị gán thành: {miss}" if miss else ""))
    elif t == "instance":
        print(f"mean matched IoU {overall['mean_matched_iou']:.3f}  P@0.5 {overall['precision@0.5']:.2f}  "
              f"R@0.5 {overall['recall@0.5']:.2f}  TP {overall['tp']} FP {overall['fp']} FN {overall['fn']}")
        for r in rows:
            print(f"\n## {r['image']}   metric {r['metric']:.3f}  mean IoU {r['mean_matched_iou']:.3f}  "
                  f"TP {r['tp']} FP {r['fp']} FN {r['fn']}")
            for c, d in r["per_class"].items():
                print(f"  {c:12s} GT {d['gt_instances']:2d}  nộp {d['submitted_instances']:2d}  "
                      f"TP {d['tp']} FP {d['fp']} FN {d['fn']}  IoU {d['mean_iou']:.3f}")
    else:
        print(f"SQ {overall['SQ']:.3f}  RQ {overall['RQ']:.3f}")
        for r in rows:
            print(f"\n## {r['image']}   PQ {r['PQ']:.3f}  SQ {r['SQ']:.3f}  RQ {r['RQ']:.3f}")
            for c, d in r["per_class"].items():
                print(f"  {c:14s} PQ {d['pq']:.3f}  SQ {d['sq']:.3f}  TP {d['tp']} FP {d['fp']} FN {d['fn']}")
    if a.overlay and t == "semantic":
        print(f"\nảnh lỗi: {a.overlay}/<ảnh>_errors.png")
    if a.json_out:
        out = {"task": a.task, "type": t, "points": points,
               "overall": {k: v for k, v in overall.items() if not k.startswith("_")}, "images": rows}
        a.json_out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"wrote {a.json_out}")


if __name__ == "__main__":
    main()

"""
╔══════════════════════════════════════════════════════════════════════╗
║  YOLOv8 CONE DETECTION — COMPLETE PRESENTATION PIPELINE             ║
║  Optimized for small datasets  (114 train / 22 val / 16 test)       ║
║  GPU-accelerated | Full metrics | Presentation-ready output         ║
╚══════════════════════════════════════════════════════════════════════╝
OUTPUT STRUCTURE
────────────────
<Results>/run_NNN/
  01_Weights/           best.pt, last.pt
  02_Training_Curves/   loss curves, metric curves, overview grid
  03_Validation/        confusion matrix, PR curve, F1 curve, bar chart
  04_Test_Predictions/  annotated images, prediction grid, detections.csv
  05_Report/            report.html  (open in browser for presentation)
  training_log.csv      raw epoch-by-epoch numbers
  results_summary.json  all metrics in one JSON
  metrics_summary.csv   val + test metrics side by side
"""
# ── standard library ──────────────────────────────────────────────────
import os, sys, json, shutil, warnings, textwrap
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")
# ── third-party ───────────────────────────────────────────────────────
import yaml
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")          # non-interactive — works headless / Windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
from ultralytics import YOLO
# ══════════════════════════════════════════════════════════════════════
#  ❶  CONFIGURATION  —  edit these two paths if your layout changes
# ══════════════════════════════════════════════════════════════════════
DATA_YAML   = Path(r"C:\Users\gbhan\Desktop\YOLO DETCTION\My First Project.yolov8\data.yaml")
OUTPUT_BASE = Path(r"C:\Users\gbhan\Desktop\YOLO DETCTION\Presentation_Results")
CONF_THRESH = 0.25   # detection confidence for inference
IOU_THRESH  = 0.50   # IoU threshold (NMS + evaluation)
# ── small-dataset training recipe ────────────────────────────────────
TRAIN_CFG = dict(
    epochs          = 150,
    patience        = 30,        # early-stop if no improvement for 30 epochs
    batch           = 8,
    imgsz           = 640,
    optimizer       = "AdamW",
    lr0             = 0.001,
    lrf             = 0.01,
    weight_decay    = 0.0005,
    warmup_epochs   = 5,
    close_mosaic    = 15,
    workers         = 0,         # required on Windows
    verbose         = True,
    # ── augmentation (critical for 114-image dataset) ──────────────
    hsv_h           = 0.02,
    hsv_s           = 0.90,
    hsv_v           = 0.50,
    degrees         = 180.0,
    translate       = 0.30,
    scale           = 0.60,
    shear           = 2.0,
    perspective     = 0.0005,
    flipud          = 0.50,
    fliplr          = 0.50,
    mosaic          = 1.0,
    mixup           = 0.15,
    copy_paste      = 0.30,
    label_smoothing = 0.10,
    dropout         = 0.10,
)
CLASS_PALETTE = [          # BGR colours for bounding boxes
    (0,   200, 83),        # class 0
    (255, 160,  0),        # class 1
    (33,  150, 243),       # class 2
    (233,  30,  99),       # class 3
    (156,  39, 176),       # class 4
    (0,   188, 212),       # class 5
]
# ══════════════════════════════════════════════════════════════════════
class Pipeline:
# ══════════════════════════════════════════════════════════════════════
    # ── initialisation ────────────────────────────────────────────────
    def __init__(self):
        self.ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.device  = self._detect_device()
        self._load_dataset_info()
        self.run_dir = None   # assigned in _make_dirs()
    def _detect_device(self):
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n🖥  GPU: {name}  ({mem:.1f} GB VRAM)")
            return 0
        print("\n⚠  No GPU found — falling back to CPU (training will be slow)")
        return "cpu"
    def _load_dataset_info(self):
        if not DATA_YAML.exists():
            sys.exit(f"❌  data.yaml not found:\n   {DATA_YAML}")
        with open(DATA_YAML) as f:
            cfg = yaml.safe_load(f)
        self.class_names = cfg.get("names", [])
        self.nc          = len(self.class_names)
        self.train_path  = cfg.get("train", "")
        self.val_path    = cfg.get("val",   "")
        self.test_path   = cfg.get("test",  "")
        print(f"\n📂 Dataset  →  {self.nc} classes: {', '.join(self.class_names)}")
        print(f"   train={self.train_path}")
        print(f"   val  ={self.val_path}")
        print(f"   test ={self.test_path}")
    def _make_dirs(self, tag: str):
        self.run_dir      = OUTPUT_BASE / tag
        self.weights_dir  = self.run_dir / "01_Weights"
        self.curves_dir   = self.run_dir / "02_Training_Curves"
        self.val_dir      = self.run_dir / "03_Validation"
        self.preds_dir    = self.run_dir / "04_Test_Predictions"
        self.report_dir   = self.run_dir / "05_Report"
        for d in [self.weights_dir, self.curves_dir,
                  self.val_dir, self.preds_dir, self.report_dir]:
            d.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Output folder: {self.run_dir.absolute()}")
    # ── user interaction ──────────────────────────────────────────────
    def ask_mode(self) -> str:
        print("\n" + "═"*62)
        print("  YOLOv8 CONE DETECTION — PIPELINE")
        print("═"*62)
        print("  [1]  TRAIN  — fresh training with YOLOv8n pretrained weights")
        print("  [2]  TEST   — evaluate / predict with an existing model")
        print("  [3]  EXIT")
        print("═"*62)
        while True:
            c = input("\n  Choice (1/2/3): ").strip()
            if c in ("1","2","3"):
                return c
            print("  ❌ Please enter 1, 2, or 3.")
    def _find_weights(self):
        """Return all best.pt files under OUTPUT_BASE, newest first."""
        return sorted(OUTPUT_BASE.rglob("best.pt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    # ── ❷  TRAINING ───────────────────────────────────────────────────
    def run_training(self):
        self._make_dirs(f"train_{self.ts}")
        print("\n" + "═"*62)
        print("  🚀  TRAINING  (YOLOv8n · epochs=150 · batch=8 · GPU)")
        print("═"*62 + "\n")
        model = YOLO("yolov8n.pt")
        model.train(
            data    = str(DATA_YAML),
            project = str(self.run_dir),
            name    = "train_output",
            device  = self.device,
            exist_ok= True,
            **TRAIN_CFG,
        )
        train_out = self.run_dir / "train_output"
        best_src  = train_out / "weights" / "best.pt"
        last_src  = train_out / "weights" / "last.pt"
        if not best_src.exists():
            sys.exit("❌  Training failed — best.pt not produced.")
        shutil.copy(best_src, self.weights_dir / "best.pt")
        if last_src.exists():
            shutil.copy(last_src, self.weights_dir / "last.pt")
        # save path reference for future TEST runs
        (OUTPUT_BASE / "LATEST_WEIGHTS.txt").write_text(
            str(self.weights_dir / "best.pt"))
        print(f"\n✅  best.pt  →  {self.weights_dir / 'best.pt'}")
        # post-training artefacts
        results_csv = train_out / "results.csv"
        if results_csv.exists():
            shutil.copy(results_csv, self.run_dir / "training_log.csv")
            self._plot_training_curves(results_csv)
        for f in train_out.glob("*.png"):    # YOLO's built-in plots
            shutil.copy(f, self.curves_dir / f.name)
        return YOLO(str(self.weights_dir / "best.pt"))
    # ── ❸  LOAD EXISTING MODEL ────────────────────────────────────────
    def load_existing_model(self):
        candidates = self._find_weights()
        if candidates:
            print("\n🔍  Found trained weights (newest first):")
            for i, p in enumerate(candidates[:6], 1):
                mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  [{i}] {p}  ({mtime})")
            sel = input("\n  Select number (Enter = [1]): ").strip()
            idx = (int(sel) - 1) if sel.isdigit() else 0
            chosen = candidates[min(idx, len(candidates)-1)]
        else:
            print("\n  ⚠  No previous weights found in output folder.")
            chosen = Path(input("  Enter full path to .pt file: ").strip())
        if not chosen.exists():
            sys.exit(f"❌  File not found: {chosen}")
        self._make_dirs(f"test_{self.ts}")
        shutil.copy(chosen, self.weights_dir / "best.pt")
        print(f"\n✅  Loaded: {chosen}")
        return YOLO(str(chosen))
    # ── ❹  VALIDATION ─────────────────────────────────────────────────
    def run_validation(self, model, split: str = "val") -> dict:
        print(f"\n{'─'*62}")
        print(f"  📊  VALIDATION  (split = {split})")
        print(f"{'─'*62}")
        res = model.val(
            data    = str(DATA_YAML),
            split   = split,
            conf    = CONF_THRESH,
            iou     = IOU_THRESH,
            device  = self.device,
            workers = 0,
            plots   = True,
            project = str(self.val_dir),
            name    = f"val_{split}",
        )
        d      = res.results_dict
        prec   = d.get("metrics/precision(B)", 0)
        recall = d.get("metrics/recall(B)",    0)
        map50  = d.get("metrics/mAP50(B)",     0)
        map5095= d.get("metrics/mAP50-95(B)",  0)
        f1     = (2*prec*recall/(prec+recall)) if (prec+recall) > 0 else 0
        metrics = dict(precision=prec, recall=recall,
                       map50=map50, map5095=map5095, f1=f1)
        print(f"\n  {'Metric':<22} {'Value':>8}")
        print(f"  {'─'*32}")
        labels = dict(precision="Precision", recall="Recall",
                      map50="mAP@0.5", map5095="mAP@0.5:0.95", f1="F1 Score")
        for k, v in metrics.items():
            print(f"  {labels[k]:<22} {v:>8.4f}")
        # copy YOLO's auto-generated plots into val_dir
        yolo_out = self.val_dir / f"val_{split}"
        if yolo_out.exists():
            for f in yolo_out.glob("*.png"):
                shutil.copy(f, self.val_dir / f"{split}_{f.name}")
        return metrics
    # ── ❺  TRAINING CURVES ────────────────────────────────────────────
    def _plot_training_curves(self, csv_path: Path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        col0 = df.columns[0]
        if col0 != "epoch":
            df = df.rename(columns={col0: "epoch"})
        ep = df["epoch"]
        # ── A: loss curves ─────────────────────────────────────────
        pairs = [("train/box_loss","val/box_loss","Box Loss"),
                 ("train/cls_loss","val/cls_loss","Class Loss"),
                 ("train/dfl_loss","val/dfl_loss","DFL Loss")]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Training vs Validation Loss", fontsize=15, fontweight="bold")
        for ax, (tc, vc, title) in zip(axes, pairs):
            if tc in df and vc in df:
                ax.plot(ep, df[tc], label="Train", color="#3B82F6", lw=2)
                ax.plot(ep, df[vc], label="Val",   color="#EF4444", lw=2, ls="--")
                ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
                ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.curves_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        # ── B: metric curves ────────────────────────────────────────
        mcols = [("metrics/precision(B)","Precision","#10B981"),
                 ("metrics/recall(B)",   "Recall",   "#F59E0B"),
                 ("metrics/mAP50(B)",    "mAP@0.5",  "#8B5CF6"),
                 ("metrics/mAP50-95(B)","mAP@0.5:0.95","#EC4899")]
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        fig.suptitle("Validation Metrics vs Epoch", fontsize=15, fontweight="bold")
        for ax, (col, title, color) in zip(axes, mcols):
            if col in df:
                ax.plot(ep, df[col], color=color, lw=2.5)
                ax.fill_between(ep, df[col], alpha=0.15, color=color)
                ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylim(0, 1.05)
                ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.curves_dir / "metric_curves.png", dpi=150, bbox_inches="tight")
        plt.close()
        # ── C: 2×5 overview grid ────────────────────────────────────
        all_plots = [
            ("train/box_loss","Train Box Loss","#3B82F6"),
            ("train/cls_loss","Train Cls Loss","#6366F1"),
            ("train/dfl_loss","Train DFL Loss","#A78BFA"),
            ("val/box_loss",  "Val Box Loss",  "#EF4444"),
            ("val/cls_loss",  "Val Cls Loss",  "#F97316"),
            ("metrics/precision(B)", "Precision","#10B981"),
            ("metrics/recall(B)",    "Recall",   "#F59E0B"),
            ("metrics/mAP50(B)",     "mAP@0.5",  "#8B5CF6"),
            ("metrics/mAP50-95(B)", "mAP@0.5:0.95","#EC4899"),
            ("val/dfl_loss",  "Val DFL Loss",  "#FB7185"),
        ]
        fig = plt.figure(figsize=(25, 10))
        gs  = gridspec.GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle("Complete Training Overview", fontsize=17, fontweight="bold")
        for i, (col, title, color) in enumerate(all_plots):
            r, c = divmod(i, 5)
            ax = fig.add_subplot(gs[r, c])
            if col in df:
                ax.plot(ep, df[col], color=color, lw=2)
                ax.fill_between(ep, df[col], alpha=0.12, color=color)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Epoch", fontsize=8)
            ax.grid(alpha=0.3)
        fig.savefig(self.curves_dir / "training_overview.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  📈 Training curves ➜ {self.curves_dir}")
    # ── ❻  METRICS BAR CHART ──────────────────────────────────────────
    def _plot_metrics_bar(self, val_m: dict, test_m: dict | None):
        labels  = ["Precision","Recall","mAP@0.5","mAP@0.5:0.95","F1 Score"]
        keys    = ["precision","recall","map50","map5095","f1"]
        val_v   = [val_m.get(k, 0) for k in keys]
        test_v  = [test_m.get(k, 0) for k in keys] if test_m else None
        x = np.arange(len(labels)); w = 0.35
        fig, ax = plt.subplots(figsize=(13, 6))
        b1 = ax.bar(x - w/2 if test_v else x, val_v, w,
                    label="Validation", color="#3B82F6", alpha=0.88, edgecolor="white", linewidth=1.2)
        if test_v:
            b2 = ax.bar(x + w/2, test_v, w,
                        label="Test", color="#10B981", alpha=0.88, edgecolor="white", linewidth=1.2)
        def _label(bars):
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2, h+0.012,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        _label(b1)
        if test_v: _label(b2)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylim(0, 1.18)
        ax.set_title("Detection Metrics — Validation vs Test", fontsize=14, fontweight="bold")
        ax.set_ylabel("Score"); ax.legend(fontsize=12)
        ax.axhline(0.9, color="gray", ls=":", lw=1.5, alpha=0.7)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = self.val_dir / "metrics_bar_chart.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  📊 Metrics bar chart ➜ {out}")
    # ── ❼  TEST PREDICTIONS ───────────────────────────────────────────
    def run_test_predictions(self, model) -> list:
        print(f"\n{'─'*62}")
        print("  🔍  TEST SET PREDICTIONS")
        print(f"{'─'*62}")
        if not self.test_path or not Path(self.test_path).exists():
            print(f"  ⚠  Test path not found: {self.test_path}"); return []
        results_gen = model.predict(
            source  = self.test_path,
            conf    = CONF_THRESH,
            iou     = IOU_THRESH,
            device  = self.device,
            workers = 0,
            stream  = True,
            verbose = False,
        )
        detections = []
        saved      = 0
        for res in results_gen:
            img      = res.orig_img.copy()
            img_name = Path(res.path).name
            if res.boxes and len(res.boxes):
                for box in res.boxes:
                    cls_id   = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_name = res.names[cls_id]
                    color    = CLASS_PALETTE[cls_id % len(CLASS_PALETTE)]
                    detections.append(dict(
                        image=img_name, cls=cls_name,
                        confidence=round(conf_val,4),
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        w_px=x2-x1, h_px=y2-y1,
                    ))
                    # thick box
                    cv2.rectangle(img, (x1,y1), (x2,y2), color, 3)
                    label = f"{cls_name} {conf_val:.2f}"
                    (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img, (x1, y1-th-10), (x1+tw+6, y1), color, -1)
                    cv2.putText(img, label, (x1+3, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            else:
                cv2.putText(img, "No Detection", (20,45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,220), 3)
            cv2.imwrite(str(self.preds_dir / img_name), img)
            saved += 1
        if detections:
            pd.DataFrame(detections).to_csv(
                self.preds_dir / "detections.csv", index=False)
        print(f"\n  ✅ {saved} images annotated  |  {len(detections)} total detections")
        print(f"     ➜ {self.preds_dir}")
        return detections
    # ── prediction grid ───────────────────────────────────────────────
    def _make_prediction_grid(self):
        imgs = sorted([p for p in self.preds_dir.glob("*.*")
                       if p.suffix.lower() in {".jpg",".jpeg",".png"}
                       and "prediction_grid" not in p.name])
        if not imgs: return
        n = len(imgs); ncols = min(4, n); nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols*5, nrows*4.2),
                                 squeeze=False)
        fig.suptitle("Test Set Predictions", fontsize=16, fontweight="bold")
        idx = 0
        for r in range(nrows):
            for c in range(ncols):
                ax = axes[r][c]
                if idx < n:
                    bgr = cv2.imread(str(imgs[idx]))
                    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    ax.set_title(imgs[idx].name, fontsize=9)
                ax.axis("off"); idx += 1
        fig.tight_layout()
        out = self.preds_dir / "prediction_grid.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  🖼  Prediction grid   ➜ {out}")
    # ── ❽  HTML REPORT ────────────────────────────────────────────────
    def _generate_html_report(self, val_m: dict, test_m: dict | None, mode: str):
        """Self-contained HTML file — open in any browser for presentation."""
        def metric_rows(m: dict | None, label: str) -> str:
            if not m: return ""
            nice = dict(precision="Precision", recall="Recall",
                        map50="mAP@0.5", map5095="mAP@0.5:0.95", f1="F1 Score")
            rows = "".join(
                f"<tr><td>{nice[k]}</td><td>{m.get(k,0)*100:.2f}%</td></tr>"
                for k in nice)
            return f"<h3>{label}</h3><table class='mt'>{rows}</table>"
        def img_tag(path: Path, caption: str = "", width: str = "100%") -> str:
            rel = os.path.relpath(path, self.report_dir).replace("\\", "/")
            cap = f"<p class='caption'>{caption}</p>" if caption else ""
            return f"<div class='img-wrap'><img src='{rel}' style='width:{width}'>{cap}</div>"
        # gather images
        curves   = sorted(self.curves_dir.glob("*.png"))
        val_imgs = sorted(self.val_dir.glob("*.png"))
        pred_img = self.preds_dir / "prediction_grid.png"
        curves_html = "\n".join(img_tag(p, p.stem.replace("_"," ").title()) for p in curves)
        val_html    = "\n".join(img_tag(p, p.stem.replace("_"," ").title()) for p in val_imgs)
        pred_html   = img_tag(pred_img, "All Test Predictions") if pred_img.exists() else ""
        html = f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<title>YOLOv8 Cone Detection — Results</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e2e8f0}}
  header{{background:linear-gradient(135deg,#1d4ed8,#7c3aed);padding:36px 48px;text-align:center}}
  header h1{{font-size:2.2rem;letter-spacing:1px}}
  header p{{opacity:.8;margin-top:6px}}
  .badge{{display:inline-block;background:#ffffff22;border-radius:999px;
          padding:4px 14px;font-size:.85rem;margin:4px 2px}}
  section{{max-width:1400px;margin:40px auto;padding:0 32px}}
  h2{{font-size:1.5rem;color:#93c5fd;border-bottom:2px solid #1e3a5f;
      padding-bottom:8px;margin-bottom:24px}}
  h3{{font-size:1.1rem;color:#c4b5fd;margin:18px 0 8px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}}
  .card{{background:#1e293b;border-radius:12px;padding:20px;
         box-shadow:0 4px 20px #0005}}
  .img-wrap{{text-align:center;margin-bottom:12px}}
  img{{border-radius:8px;max-width:100%}}
  .caption{{font-size:.8rem;color:#94a3b8;margin-top:6px}}
  table.mt{{width:100%;border-collapse:collapse;font-size:.95rem}}
  .mt th,.mt td{{padding:9px 14px;border:1px solid #334155;text-align:left}}
  .mt tr:nth-child(even){{background:#0f172a}}
  .mt td:last-child{{font-weight:700;color:#4ade80;text-align:right}}
  footer{{text-align:center;padding:30px;color:#475569;font-size:.85rem}}
</style>
</head>
<body>
<header>
  <h1>🎯 YOLOv8 Cone Detection</h1>
  <p>Object Detection Results — Class Presentation</p>
  <br>
  <span class='badge'>Mode: {mode.upper()}</span>
  <span class='badge'>Dataset: 114 train · 22 val · 16 test</span>
  <span class='badge'>Model: YOLOv8n</span>
  <span class='badge'>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
</header>
<section>
  <h2>📊 Performance Metrics</h2>
  <div class='grid'>
    <div class='card'>
      {metric_rows(val_m, "Validation Set")}
    </div>
    <div class='card'>
      {metric_rows(test_m, "Test Set") if test_m else "<p style='color:#94a3b8'>Test metrics not available.</p>"}
    </div>
    <div class='card'>
      {img_tag(self.val_dir / "metrics_bar_chart.png", "Metrics Comparison") if (self.val_dir / "metrics_bar_chart.png").exists() else ""}
    </div>
  </div>
</section>
<section>
  <h2>📈 Training Curves</h2>
  <div class='grid'>
    {curves_html}
  </div>
</section>
<section>
  <h2>📉 Validation Plots</h2>
  <div class='grid'>
    {val_html}
  </div>
</section>
<section>
  <h2>🔍 Test Predictions</h2>
  {pred_html}
</section>
<footer>
  YOLOv8 Pipeline · {self.ts} · Results in {self.run_dir.name}
</footer>
</body></html>"""
        out = self.report_dir / "report.html"
        out.write_text(html, encoding="utf-8")
        print(f"  🌐 HTML report        ➜ {out}")
    # ── ❾  SAVE SUMMARY FILES ─────────────────────────────────────────
    def _save_summary(self, val_m, test_m, mode, train_params=None):
        summary = dict(
            timestamp        = self.ts,
            mode             = mode,
            device           = str(self.device),
            dataset          = dict(classes=self.class_names, nc=self.nc,
                                    yaml=str(DATA_YAML)),
            training_params  = train_params,
            validation_metrics = val_m,
            test_metrics       = test_m,
        )
        (self.run_dir / "results_summary.json").write_text(
            json.dumps(summary, indent=2))
        rows = []
        for split, m in [("Validation", val_m), ("Test", test_m)]:
            if m:
                rows.append({"Split": split, **m})
        if rows:
            pd.DataFrame(rows).to_csv(
                self.run_dir / "metrics_summary.csv", index=False)
        print(f"  💾 Summary JSON/CSV   ➜ {self.run_dir}")
    # ── ❿  FINAL PRINT ────────────────────────────────────────────────
    def _print_final(self, val_m, test_m):
        labels = [("precision","Precision"),("recall","Recall"),
                  ("map50","mAP@0.5"),("map5095","mAP@0.5:0.95"),("f1","F1 Score")]
        print("\n" + "═"*62)
        print("  🎉  PIPELINE COMPLETE")
        print("═"*62)
        print(f"\n  {'Metric':<20} {'Validation':>12} {'Test':>12}")
        print(f"  {'─'*46}")
        for key, label in labels:
            v = f"{val_m.get(key,0):.4f}"  if val_m  else "–"
            t = f"{test_m.get(key,0):.4f}" if test_m else "N/A"
            print(f"  {label:<20} {v:>12} {t:>12}")
        print(f"\n  📁 All results:       {self.run_dir.absolute()}")
        print(f"  🏆 Best weights:      {self.weights_dir / 'best.pt'}")
        print(f"  📈 Curves:            {self.curves_dir}")
        print(f"  📊 Validation:        {self.val_dir}")
        print(f"  🖼  Predictions:       {self.preds_dir}")
        print(f"  🌐 Open in browser:   {self.report_dir / 'report.html'}")
        print("═"*62)
    # ── MAIN ──────────────────────────────────────────────────────────
    def run(self):
        choice = self.ask_mode()
        if choice == "3":
            print("\n👋  Bye!"); return
        val_m = test_m = model = None
        mode  = "train" if choice == "1" else "test"
        if choice == "1":
            model = self.run_training()
            train_params = TRAIN_CFG
        else:
            model = self.load_existing_model()
            train_params = None
        # validation
        val_m = self.run_validation(model, split="val")
        # test-split evaluation (only if ground-truth labels exist)
        if self.test_path and Path(self.test_path).exists():
            lbl_dir = Path(self.test_path).parent / "labels"
            if lbl_dir.exists() and any(lbl_dir.glob("*.txt")):
                test_m = self.run_validation(model, split="test")
        # metrics bar chart
        self._plot_metrics_bar(val_m, test_m)
        # test predictions + grid
        self.run_test_predictions(model)
        self._make_prediction_grid()
        # summary files + HTML
        self._save_summary(val_m, test_m, mode, train_params)
        self._generate_html_report(val_m, test_m, mode)
        # final print
        self._print_final(val_m, test_m)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    Pipeline().run()

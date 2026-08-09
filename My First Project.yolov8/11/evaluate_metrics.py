# evaluate_metrics.py
import pandas as pd
import os

CSV_PATH = r"runs/detect/train/results.csv"

if not os.path.exists(CSV_PATH):
    print(f"Results CSV not found at {CSV_PATH}. Train the model first.")
    exit()

df = pd.read_csv(CSV_PATH)
# The last row contains final validation metrics
last_row = df.iloc[-1]

# Extract relevant columns (names may vary slightly)
# Typical columns: 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
precision = last_row.get('metrics/precision(B)', None)
recall = last_row.get('metrics/recall(B)', None)
mAP50 = last_row.get('metrics/mAP50(B)', None)
mAP5095 = last_row.get('metrics/mAP50-95(B)', None)

# Calculate F1 score = 2 * (precision * recall) / (precision + recall)
if precision is not None and recall is not None and (precision + recall) > 0:
    f1 = 2 * (precision * recall) / (precision + recall)
else:
    f1 = None

print("\n========== VALIDATION METRICS ==========")
print(f"Precision (box): {precision:.4f}" if precision is not None else "Precision: N/A")
print(f"Recall (box):    {recall:.4f}" if recall is not None else "Recall: N/A")
print(f"mAP@0.5:         {mAP50:.4f}" if mAP50 is not None else "mAP50: N/A")
print(f"mAP@0.5:0.95:    {mAP5095:.4f}" if mAP5095 is not None else "mAP50-95: N/A")
print(f"F1 Score:        {f1:.4f}" if f1 is not None else "F1 Score: N/A")
print("========================================")
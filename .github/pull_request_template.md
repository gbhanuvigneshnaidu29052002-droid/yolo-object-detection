## Description
A concise summary of the architectural changes, tracking logic, or bug fixes introduced.

## Related Issue(s)
Fixes #(issue number) or Addresses #(issue number)

## Type of Change
- [ ] Bug fix (non-breaking fix in model forward pass, inference, or parser)
- [ ] New feature (adding new detector variant, tracker, or export pipeline)
- [ ] Performance improvement (training speedup, TensorRT optimization, latency reduction)
- [ ] Documentation update (improving guides, docstrings, or benchmarks)
- [ ] Automated testing (adding unit tests or coverage)

## Verification Checklist
- [ ] Dependencies install cleanly via `pip install -r requirements.txt`
- [ ] Automated unit test suite passes:
  ```bash
  python3 -m unittest discover -s tests -v
  ```
- [ ] Detection & counting logic validated on test imagery
- [ ] Device-agnostic execution verified (CPU and CUDA)
- [ ] Evaluator metrics (mAP@0.5, Precision, Recall, MAE) preserved

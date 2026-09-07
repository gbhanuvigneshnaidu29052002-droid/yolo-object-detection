# Security Policy

## Supported Versions

We actively maintain and provide security patches for the current release of **Real-Time Traffic Cone Object Detection & Instance Counting via YOLOv8**.

| Version | Supported |
| :--- | :---: |
| 1.0.x | :white_check_mark: |
| < 1.0.0 | :x: |

---

## Threat Model & Considerations

Please observe the following security considerations when deploying this object detection framework in laboratory, cloud, or autonomous field environments:

1. **Model Checkpoint Deserialization**: PyTorch checkpoint loading (`.pt` files via `torch.load` / `pickle`) can execute arbitrary code if weights originate from untrusted sources. Only load checkpoint weights from verified, cryptographically validated sources and prefer safe tensor serialization (`safetensors` or `torch.load(..., weights_only=True)`).
2. **Input Stream & Image Decoding**: Camera frame ingestion and image processing via OpenCV (`cv2.imread`) or PIL should guard against malformed headers, buffer overflows, and image decompression bomb vulnerabilities.
3. **Adversarial Attacks on Autonomous Systems**: In safety-critical robotic navigation and autonomous driving, physical adversarial patch attacks (adversarially painted patterns on cones) could cause miss-detections or false positives. Deploy defensive sanity checks such as temporal track verification and sensor fusion (LiDAR/Radar + Camera).
4. **Supply Chain & Dependency Security**: Ultralytics, PyTorch, and CUDA dependencies should be regularly updated to patched upstream versions.

---

## Reporting a Vulnerability

If you identify a security issue or vulnerability within this project, please follow responsible disclosure:

1. **Do not create a public issue**: Refrain from submitting publicly accessible bug reports for potential security exploits.
2. **Contact Maintainer**: Reach out to the project maintainer via their GitHub profile at [gbhanuvigneshnaidu29052002-droid](https://github.com/gbhanuvigneshnaidu29052002-droid) or submit a private security advisory through the GitHub repository's **Security** tab.
3. **Provide Detailed Information**:
   - Description of the vulnerability and its potential exploit vectors
   - Affected modules (`src/detector.py`, `src/trainer.py`, dependencies)
   - Step-by-step reproduction instructions or Proof-of-Concept (PoC) script

We will acknowledge reports within 48 hours and coordinate a timely resolution.

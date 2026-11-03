# MedSeg U-Net

A reproducible PyTorch U-Net foundation for binary medical-image segmentation research.

> **Evidence policy:** This first milestone uses generated geometric masks in CI to verify pairing, splits, architecture, training, and artifact handling. It contains no clinical data, pretrained weights, or real medical-image result. Synthetic Dice or IoU values are software smoke-test outputs—not medical performance evidence.

## Engineering focus

- Real 2D U-Net encoder/decoder with skip connections
- Filename-stem image/mask pairing, never position-based matching
- ID-level train/validation/test splits that keep a pair together
- Dice/IoU evaluation and reproducible checkpoint manifests
- Offline tests using generated shapes only

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e . --no-deps
medseg-unet synth-demo --out runs/synthetic-demo
```

The generated manifest is labeled `synthetic`; it must not be cited as medical benchmark evidence.

## Working with real data

Contributors should normalize a licensed, de-identified source into:

```text
images/<case_id>.pt
masks/<case_id>.pt
```

Every case ID must have exactly one image and mask. Check this before training:

```bash
medseg-unet validate-pairs --data path/to/data
```

It lists image-only and mask-only case IDs and exits non-zero if any pair is missing. Keep all augmented crops and slices from one case in the same split. Do not commit patient data, datasets, or unverified weights.

Any real result must ship with dataset/version/license, split manifest, preprocessing details, seed, command, metrics, and known limitations. A bare Dice score is not sufficient.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), [ROADMAP.md](ROADMAP.md), and [docs/extending.md](docs/extending.md). The roadmap includes well-scoped adapters, augmentations, multiclass support, and inference/export work.

## Scope limits

This is a research pipeline, not a clinical tool. It does not perform diagnosis, validate generalization across sites/modalities, or make any clinical claim.

## License

MIT — see [LICENSE](LICENSE).
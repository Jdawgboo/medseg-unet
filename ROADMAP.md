# Roadmap

## Milestone 1 — current

- [x] Pairing/split invariants
- [x] Offline U-Net smoke pipeline
- [x] Provenance-aware artifacts

## Contribution-ready extensions

1. **Good first issue:** Add a case-ID validator for real-data folder adapters.
2. **Good first issue:** Add deterministic flips/rotations and tests proving masks transform with images.
3. **Help wanted:** Add multiclass Dice and class-aware loss support.
4. **Help wanted:** Add an inference CLI and ONNX export path.
5. **Extension:** Add a licensed public-dataset adapter with conversion notes, no data committed.

Every extension must preserve case-level splits and synthetic-versus-real evidence labels.

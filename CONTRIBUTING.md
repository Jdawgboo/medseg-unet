# Contributing

## Setup

Install CPU PyTorch and the package, then run:

```bash
python -m unittest discover -s tests -v
```

## Rules

- Keep synthetic fixtures clearly labeled; never present them as clinical data.
- Preserve image/mask pairing by case ID and split by case, not by augmented slice.
- Add tests for new data adapters, losses, or architectures.
- Do not commit patient data, private keys, unlicensed datasets, or unverified performance claims.

## Good first contributions

Start with one of the scoped roadmap issues. Keep changes focused and include an explanation of data provenance whenever a data adapter changes.

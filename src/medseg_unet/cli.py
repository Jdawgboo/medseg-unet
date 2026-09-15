from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence
def validate_pairs_command(data: str | Path) -> int:
    from .pipeline import list_case_ids, pair_case_ids
    root = Path(data); image_ids, mask_ids = list_case_ids(root)
    try: paired = pair_case_ids(image_ids, mask_ids)
    except ValueError:
        image_only = sorted(image_ids - mask_ids); mask_only = sorted(mask_ids - image_ids)
        print(f"Case ID mismatch in {root}:")
        print(f" image-only ({len(image_only)}): {', '.join(image_only) if image_only else 'none'}")
        print(f" mask-only ({len(mask_only)}): {', '.join(mask_only) if mask_only else 'none'}")
        print("Fix: add missing images/<case_id>.pt or masks/<case_id>.pt files so every case ID has exactly one image and mask.")
        return 1
    print(f"Validated {len(paired)} paired cases in {root}."); return 0
def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); commands=parser.add_subparsers(dest="command",required=True); demo=commands.add_parser("synth-demo"); demo.add_argument("--out",required=True); demo.add_argument("--epochs",type=int,default=1); validator=commands.add_parser("validate-pairs",help="Validate paired case IDs before training"); validator.add_argument("--data",required=True,help="Real-data folder containing images/<case_id>.pt and masks/<case_id>.pt"); args=parser.parse_args(argv)
    if args.command == "validate-pairs": return validate_pairs_command(args.data)
    from .pipeline import run_synthetic_demo, write_artifacts
    model,result=run_synthetic_demo(epochs=args.epochs); write_artifacts(model,result,args.out); print("Synthetic segmentation smoke run completed; no medical performance claim is implied."); return 0

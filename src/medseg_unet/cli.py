from __future__ import annotations
import argparse
import sys
from typing import Sequence
from .pipeline import run_synthetic_demo, validate_case_dir, write_artifacts
def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); commands=parser.add_subparsers(dest="command",required=True); demo=commands.add_parser("synth-demo"); demo.add_argument("--out",required=True); demo.add_argument("--epochs",type=int,default=1)
    check=commands.add_parser("validate-pairs",help="check that images/<case_id>.pt and masks/<case_id>.pt match"); check.add_argument("--data",required=True,help="folder containing images/ and masks/"); args=parser.parse_args(argv)
    if args.command == "validate-pairs": return _validate_pairs(args.data)
    model,result=run_synthetic_demo(epochs=args.epochs); write_artifacts(model,result,args.out); print("Synthetic segmentation smoke run completed; no medical performance claim is implied."); return 0

def _validate_pairs(data: str) -> int:
    try: paired, image_only, mask_only = validate_case_dir(data)
    except FileNotFoundError as error: print(f"error: {error}", file=sys.stderr); return 2
    if image_only or mask_only:
        if image_only: print(f"images without masks ({len(image_only)}): {', '.join(image_only)}", file=sys.stderr)
        if mask_only: print(f"masks without images ({len(mask_only)}): {', '.join(mask_only)}", file=sys.stderr)
        return 1
    print(f"OK: {len(paired)} paired cases"); return 0

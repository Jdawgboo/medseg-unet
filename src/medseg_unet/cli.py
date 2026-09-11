from __future__ import annotations
import argparse
from typing import Sequence
from .pipeline import run_synthetic_demo, write_artifacts
def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); commands=parser.add_subparsers(dest="command",required=True); demo=commands.add_parser("synth-demo"); demo.add_argument("--out",required=True); demo.add_argument("--epochs",type=int,default=1); args=parser.parse_args(argv)
    model,result=run_synthetic_demo(epochs=args.epochs); write_artifacts(model,result,args.out); print("Synthetic segmentation smoke run completed; no medical performance claim is implied."); return 0

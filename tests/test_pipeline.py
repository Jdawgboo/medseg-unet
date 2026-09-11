from pathlib import Path
import tempfile, unittest
import torch
from medseg_unet.cli import main
from medseg_unet.model import UNet
from medseg_unet.pipeline import pair_case_ids, run_synthetic_demo, split_case_ids, write_artifacts
class MedSegTests(unittest.TestCase):
 def test_unet_shapes(self):
  self.assertEqual(tuple(UNet()(torch.randn(2,1,32,32)).shape),(2,1,32,32))
 def test_pairing_and_splits(self):
  ids=pair_case_ids({"a","b","c","d","e"},{"a","b","c","d","e"}); splits=split_case_ids(ids); self.assertFalse(set(splits["train"])&set(splits["test"]))
  with self.assertRaises(ValueError): pair_case_ids({"a"},{"b"})
 def test_smoke_and_cli(self):
  model,result=run_synthetic_demo(epochs=1); self.assertTrue(0<=result["metrics"]["test_dice"]<=1)
  with tempfile.TemporaryDirectory() as target:
   write_artifacts(model,result,target); self.assertTrue((Path(target)/"run_manifest.json").exists()); self.assertEqual(main(["synth-demo","--out",target]),0)
if __name__=="__main__": unittest.main()

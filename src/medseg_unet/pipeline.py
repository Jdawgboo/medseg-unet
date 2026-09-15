from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from .model import UNet

def pair_case_ids(image_ids: set[str], mask_ids: set[str]) -> list[str]:
    if image_ids != mask_ids: raise ValueError(f"image/mask case IDs differ: images_only={sorted(image_ids-mask_ids)}, masks_only={sorted(mask_ids-image_ids)}")
    return sorted(image_ids)

def list_case_ids(data_dir: str | Path) -> tuple[set[str], set[str]]:
    root = Path(data_dir); image_dir = root / "images"; mask_dir = root / "masks"
    image_ids = {p.stem for p in image_dir.glob("*.pt")} if image_dir.is_dir() else set()
    mask_ids = {p.stem for p in mask_dir.glob("*.pt")} if mask_dir.is_dir() else set()
    return image_ids, mask_ids

def validate_data_dir(data_dir: str | Path) -> list[str]:
    image_ids, mask_ids = list_case_ids(data_dir)
    return pair_case_ids(image_ids, mask_ids)

def split_case_ids(case_ids: list[str], seed: int = 17) -> dict[str, list[str]]:
    if len(case_ids) < 5: raise ValueError("at least five paired cases are needed")
    ordered = sorted(case_ids); random.Random(seed).shuffle(ordered)
    return {"test": ordered[:1], "validation": ordered[1:2], "train": ordered[2:]}

def synthetic_cases(seed: int = 17, cases: int = 8, size: int = 32) -> tuple[Tensor, Tensor, list[str]]:
    generator = torch.Generator().manual_seed(seed); axis = torch.arange(size).float(); yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    images=[]; masks=[]; ids=[]
    for index in range(cases):
        cx, cy = torch.randint(8, size-8, (2,), generator=generator); radius = torch.randint(4, 8, (1,), generator=generator)
        mask = (((xx-cx)**2 + (yy-cy)**2) < radius.float()**2).float()
        image = (mask * 0.7 + 0.12 * torch.randn(size, size, generator=generator)).clamp(0, 1)
        images.append(image.unsqueeze(0)); masks.append(mask.unsqueeze(0)); ids.append(f"synthetic-case-{index:03d}")
    return torch.stack(images), torch.stack(masks), ids

def dice(logits: Tensor, masks: Tensor) -> Tensor:
    predicted = (torch.sigmoid(logits) > .5).float(); return ((2*(predicted*masks).sum((1,2,3)) + 1e-6) / (predicted.sum((1,2,3)) + masks.sum((1,2,3)) + 1e-6)).mean()

def run_synthetic_demo(seed: int = 17, epochs: int = 1) -> tuple[UNet, dict[str, Any]]:
    torch.manual_seed(seed); images, masks, ids = synthetic_cases(seed); splits = split_case_ids(pair_case_ids(set(ids), set(ids)), seed)
    positions = {case: index for index, case in enumerate(ids)}; train_ids, val_ids, test_ids = (splits[key] for key in ("train", "validation", "test"))
    loader = DataLoader(TensorDataset(images[[positions[x] for x in train_ids]], masks[[positions[x] for x in train_ids]]), batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model=UNet(); optimizer=torch.optim.Adam(model.parameters(), lr=.001); loss_fn=nn.BCEWithLogitsLoss(); history=[]
    for epoch in range(epochs):
        model.train(); total=0.
        for image, mask in loader:
            optimizer.zero_grad(); loss=loss_fn(model(image), mask); loss.backward(); optimizer.step(); total += loss.item()
        model.eval()
        with torch.no_grad(): validation_dice=float(dice(model(images[[positions[x] for x in val_ids]]), masks[[positions[x] for x in val_ids]]))
        history.append({"epoch":epoch+1,"train_loss":total/max(len(loader),1),"validation_dice":validation_dice})
    with torch.no_grad(): test_logits=model(images[[positions[x] for x in test_ids]]); test_dice=float(dice(test_logits,masks[[positions[x] for x in test_ids]]))
    return model, {"metrics":{"history":history,"test_dice":test_dice},"manifest":{"data_source":"synthetic","warning":"Synthetic geometric masks only; not medical-image performance.","seed":seed,"split_case_counts":{key:len(value) for key,value in splits.items()}}}

def write_artifacts(model: nn.Module, result: dict[str, Any], output: str | Path) -> None:
    target=Path(output); target.mkdir(parents=True,exist_ok=True); (target/"metrics.json").write_text(json.dumps(result["metrics"],indent=2)+"\n"); (target/"run_manifest.json").write_text(json.dumps(result["manifest"],indent=2)+"\n"); torch.save(model.state_dict(),target/"model.pt")

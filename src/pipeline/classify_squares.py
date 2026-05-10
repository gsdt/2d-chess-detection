"""Classify the 64 squares of a board crop using the trained piece classifier."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ..train.train_classifier import build_model


# Same case-safe folder mapping used during training (alphabetical order).
CLASS_DIRS = sorted(["empty"] + [f"{c}{p}" for c in "wb" for p in "PNBRQK"])


def dirname_to_symbol(name: str) -> str:
    if name == "empty":
        return "."
    color, piece = name[0], name[1]
    return piece if color == "w" else piece.lower()


class PieceClassifier:
    def __init__(self, ckpt: Path, device: str | None = None):
        self.device = torch.device(device) if device else torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        state = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.classes: list[str] = state["classes"]
        self.img_size: int = int(state.get("img_size", 96))
        self.model = build_model(num_classes=len(self.classes), pretrained=False).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.tf = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def classify_squares(self, board_img: Image.Image) -> tuple[list[str], list[list[float]]]:
        """Return (symbols, probs) for the 64 squares in display order (rank8→rank1, a→h)."""
        w, h = board_img.size
        sw, sh = w / 8, h / 8
        crops = []
        for r in range(8):
            for c in range(8):
                crops.append(board_img.crop((int(c * sw), int(r * sh), int((c + 1) * sw), int((r + 1) * sh))))
        batch = torch.stack([self.tf(c) for c in crops]).to(self.device)
        logits = self.model(batch)
        probs = F.softmax(logits, dim=1).cpu().tolist()
        symbols = [dirname_to_symbol(self.classes[max(range(len(p)), key=lambda i: p[i])]) for p in probs]
        return symbols, probs

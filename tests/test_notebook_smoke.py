"""Fast structural smoke checks for the teaching notebook, without training."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import Dataset


def notebook_components():
    notebook = json.loads((Path(__file__).resolve().parents[1] / "Untitled7.ipynb").read_text())
    namespace = {"torch": torch, "nn": nn, "F": functional, "math": math, "Dataset": Dataset}
    prefixes = ("class MultiHeadAttention", "class FeedForward", "class TransformerBlock",
                "class MiniGPT", "class TextDataset", "def load_data")
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and source.startswith(prefixes):
            exec(compile(source, "Untitled7.ipynb", "exec"), namespace)
    return namespace


class NotebookSmokeTests(unittest.TestCase):
    def test_part_one_artifact_loader_and_model_forward(self):
        components = notebook_components()
        artifact = {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.pt"
            torch.save(artifact, path)
            tokens, vocabulary = components["load_data"](str(path), max_tokens=5)
        self.assertEqual(tokens, [1, 2, 3, 4, 5])
        self.assertEqual(vocabulary, 6)
        model = components["MiniGPT"](vocab_size=vocabulary, d_model=16,
                                       num_layers=1, num_heads=4, d_ff=32,
                                       max_seq_len=4, dropout=0)
        logits = model(torch.tensor([[1, 2, 3, 4]]))
        self.assertEqual(tuple(logits.shape), (1, 4, vocabulary))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()

# Mini-GPT training experiment (coursework, Part 2)

This is the training half of a two-part learning project. [Part 1](https://github.com/jiabeiliu/Assignment-1) prepares text into GPT-2 token-ID blocks; this repository contains an exploratory decoder-only Transformer notebook with causal self-attention, training/validation loops, perplexity, and checkpoints. It is a **small educational experiment**, not a general-purpose foundation model.

## Connect Part 1 to Part 2

Use Python 3.11. Clone both repositories side by side, prepare a corpus you are allowed to use, and run Part 1 as documented there. Then, from this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p data
cp ../Assignment-1/data/processed/tokenized_blocks.pt data/tokenized_blocks.pt
jupyter notebook Untitled7.ipynb
```

Run notebook cells from the top. The loader now consumes the actual Part 1 dictionary (`input_ids` plus `attention_mask`), removes padding positions, and keeps the original token IDs. It no longer maps distinct tokens to the same value with `% 1000`. For a first CPU experiment, reduce `max_samples`, `batch_size`, and `num_epochs` in the notebook config. Check that both the training and validation splits contain more than `seq_len` tokens before training.

Run the fast loader/forward-pass check with `python -m unittest discover -s tests -v`. In a local smoke run, Part 1 produced 19 short blocks from a temporary text input and this revised loader passed 400 non-padding tokens through a model forward pass. A full training run was not performed.

## What to measure

Record corpus provenance and size, split size, model configuration, train/validation loss and perplexity, runtime, and hardware. Compare against a simple baseline before making any performance claim. The notebook's saved outputs are from an earlier exploratory run; **the revised Part 1 → Part 2 pipeline has not been retrained or benchmarked in this repository**. Do not present those old outputs as results from the current code.

## Known limitations

- Concatenating non-padding tokens from separate documents can create artificial transitions at document boundaries; a stronger data loader would preserve boundaries or insert explicit end-of-document tokens.
- The notebook contains exploratory duplicate class definitions and is not yet a clean training package or automated CI benchmark.
- No raw training corpus or model checkpoint is committed. Reproducing a meaningful run requires an appropriately licensed corpus and sufficient CPU/GPU memory.
- The attached PDF is the original coursework report, not verification of a fresh run with the revised loader.

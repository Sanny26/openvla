## OPENVLA Worklog

- Use `pyproject.toml` to set up OpenVLA (PyTorch versions are defined here).
- Install FlashAttention and BitsAndBytes versions compatible with the `pyproject.toml`.

## Action Comparison Pipeline

### 1) Download the Bridge-v2 dataset

- Change directory to your base datasets folder:

```bash
cd <PATH TO BASE DATASETS DIR>
```

- Download the full dataset (approximately 124 GB):

```bash
wget -r -nH --cut-dirs=4 --reject="index.html*" https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/
```

- Rename the dataset to `bridge_orig` (required; omitting this may lead to runtime errors later):

```bash
mv bridge_dataset bridge_orig
```

### 2) Generate action-comparison JSONs

Run the following script:

```bash
python scripts/action-compare-pipeline/bridgev2_eval.py \
  --num_action_samples 8 \
  --sample_temperature 0.8
```

## Notes
- Using quantized model variants works better with an RTX 4090.
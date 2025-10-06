import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import tqdm
from PIL import Image
from typing import Iterable, Optional, Tuple, Union
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig


def safe_builder_from_directory(dir_path: str) -> tfds.core.DatasetBuilder:
    """Wrap builder_from_directory with minimal metadata checks."""
    features_path = os.path.join(dir_path, "features.json")
    info_path = os.path.join(dir_path, "dataset_info.json")
    if not (os.path.exists(features_path) and os.path.exists(info_path)):
        raise ValueError(f"Directory {dir_path} is missing TFDS metadata files.")
    return tfds.builder_from_directory(dir_path)


def _get_expected_shard_paths(builder: tfds.core.DatasetBuilder, split: str) -> Tuple[str, list[str]]:
    if split == "all":
        splits = list(builder.info.splits.keys())
        if not splits:
            raise ValueError("Dataset has no registered splits; cannot determine shards.")
    else:
        splits = [split]

    dataset_name = builder.info.name
    data_dir = builder.data_path

    pattern_prefixes = [f"{dataset_name}-{s}.tfrecord" for s in splits]
    return data_dir, pattern_prefixes


def _fallback_dataset_from_available_shards(
    builder: tfds.core.DatasetBuilder,
    split: str,
    read_config: tfds.ReadConfig,
    shuffle_files: bool = False,
) -> tf.data.Dataset:
    """Create a dataset directly from whichever TFRecord shards are present on disk."""

    # Infer underlying split name when users request "all"
    data_dir, pattern_prefixes = _get_expected_shard_paths(builder, split)
    shard_files = []
    for pattern_prefix in pattern_prefixes:
        for fname in tf.io.gfile.listdir(data_dir):
            if fname.startswith(pattern_prefix):
                shard_files.append(os.path.join(data_dir, fname))

    if not shard_files:
        raise tf.errors.NotFoundError(
            node_def=None,
            op=None,
            message=f"No TFRecord shards found for requested split '{split}' under {data_dir}",
        )

    shard_files = sorted(set(shard_files))
    num_parallel_reads = tf.data.AUTOTUNE if shuffle_files else 1
    raw_ds = tf.data.TFRecordDataset(shard_files, num_parallel_reads=num_parallel_reads)

    features = builder.info.features

    def _deserialize(serialized_example):
        example = features.deserialize_example(serialized_example)
        return example

    dataset = raw_ds.map(_deserialize, num_parallel_calls=tf.data.AUTOTUNE)

    return dataset


def safe_load_dataset(builder: tfds.core.DatasetBuilder, split: str = "all") -> tf.data.Dataset:
    """Load TFDS dataset while relaxing strict cardinality checks and tolerating missing shards."""
    read_config = tfds.ReadConfig(assert_cardinality=False)

    # Proactively detect missing shards to avoid runtime NotFound errors during iteration.
    data_dir, pattern_prefixes = _get_expected_shard_paths(builder, split)
    expected_shards = 0
    split_lookup = {split.name: split for split in builder.info.splits.values()}
    for prefix in pattern_prefixes:
        try:
            split_name = prefix.split(f"{builder.info.name}-", 1)[1].split(".tfrecord", 1)[0]
        except IndexError:
            continue
        split_info = split_lookup.get(split_name)
        if split_info is not None and split_info.num_shards is not None:
            expected_shards += split_info.num_shards

    actual_shards = 0
    try:
        listing = tf.io.gfile.listdir(data_dir)
    except tf.errors.NotFoundError:
        listing = []
    for prefix in pattern_prefixes:
        actual_shards += sum(1 for fname in listing if fname.startswith(prefix))

    should_fallback = expected_shards and actual_shards and actual_shards < expected_shards

    try:
        if should_fallback:
            raise tf.errors.NotFoundError(None, None, "Incomplete shard set detected")
        return builder.as_dataset(split=split, shuffle_files=False, read_config=read_config)
    except tf.errors.NotFoundError as err:
        print(f"Warning: encountered missing TFRecord shards ({err}); falling back to partial shard loader.")
        return _fallback_dataset_from_available_shards(builder, split, read_config, shuffle_files=False)


def load_bridge_dataset(rlds_dir: str, split: str = "all") -> Tuple[tfds.core.DatasetBuilder, tf.data.Dataset]:
    """Build the BridgeV2 TFDS dataset using the same guards as read_bridgev2.py."""
    builder = safe_builder_from_directory(rlds_dir)
    ds_tf = safe_load_dataset(builder, split=split)
    return builder, tfds.as_numpy(ds_tf)


def materialize_steps_structure(steps_obj, max_steps: Optional[int] = None):
    """Convert a potentially nested TFDS steps object into dicts / numpy arrays."""
    if steps_obj is None:
        return {}
    if isinstance(steps_obj, dict):
        return steps_obj

    if isinstance(steps_obj, (list, tuple)):
        steps_iter: Iterable = steps_obj if max_steps is None else steps_obj[:max_steps]
    else:
        steps_iter = steps_obj
        if max_steps is not None and hasattr(steps_iter, "take"):
            steps_iter = steps_iter.take(max_steps)
        try:
            steps_iter = tfds.as_numpy(steps_iter)
        except TypeError:
            try:
                steps_iter = steps_iter.as_numpy_iterator()
            except AttributeError:
                try:
                    steps_iter = iter(steps_iter)
                except TypeError:
                    return {}

    steps_list = []
    for idx, step in enumerate(steps_iter):
        steps_list.append(step)
        if max_steps is not None and (idx + 1) >= max_steps:
            break

    aggregated = {}

    def _accumulate(target, source):
        if not isinstance(source, dict):
            return
        for key, value in source.items():
            if isinstance(value, dict):
                child = target.setdefault(key, {})
                _accumulate(child, value)
            else:
                target.setdefault(key, []).append(value)

    for step in steps_list:
        _accumulate(aggregated, step)

    def _finalize(node):
        if isinstance(node, dict):
            return {key: _finalize(value) for key, value in node.items()}
        if isinstance(node, list):
            if not node:
                return np.array([])
            try:
                return np.stack(node)
            except Exception:
                try:
                    return np.array(node)
                except Exception:
                    return node
        return node

    return _finalize(aggregated)


def _to_device(device: Union[torch.device, str]) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def load_openvla_quantized(
    model_name: str = "openvla/openvla-7b",
    device: Union[torch.device, str] = "cuda:0",
    quant_bits: int = 4,
):
    """
    Try to load OpenVLA in quantized mode (4-bit or 8-bit) using bitsandbytes / Transformers.
    If not supported, fallback to bfloat16.
    """
    # Load processor (same regardless of quantization)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    device = _to_device(device)
    device_map = None
    if device.type == "cuda":
        suffix = f":{device.index}" if device.index is not None else ""
        device_map = f"cuda{suffix}" if quant_bits not in (4, 8) else "auto"

    common_kwargs = dict(trust_remote_code=True, low_cpu_mem_usage=True)

    if quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            **common_kwargs,
        )
    elif quant_bits == 8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            **common_kwargs,
        )
    else:
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            device_map=device_map,
            **common_kwargs,
        )

    if device_map is None or quant_bits not in (4, 8):
        model = model.to(device)

    return processor, model


def determine_input_dtype(model, quant_bits: int, device: torch.device) -> torch.dtype:
    if quant_bits in (4, 8):
        # bitsandbytes quantized models expect fp16 activations on GPU
        return torch.float16 if device.type == "cuda" else torch.float32

    for param in model.parameters():
        return param.dtype
    for buffer in model.buffers():
        return buffer.dtype

    if device.type == "cuda":
        return torch.float16
    return torch.float32

def make_prompt(instruction: str, model_name: str):
    """
    Create textual prompt given instruction. The README uses:
       "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
    Some variants may include system/user messages depending on model version.
    """
    # If prompt style differs (v01 vs newer), you might adapt here.
    return f"In: What action should the robot take to {instruction}?\nOut:"

def run_inference_on_episode(
    processor,
    vla,
    device,
    instruction: str,
    image_np: np.ndarray,
    input_dtype: torch.dtype,
    num_samples: int = 1,
    do_sample: Optional[bool] = None,
    sample_kwargs: Optional[dict] = None,
):
    """Generate one or more action samples for a single frame."""
    prompt = make_prompt(instruction, model_name=None)
    pil_img = Image.fromarray(image_np.astype(np.uint8))

    effective_samples = max(1, int(num_samples))
    if do_sample is None:
        do_sample = effective_samples > 1
    sample_kwargs = sample_kwargs or {}

    actions = []
    for _ in range(effective_samples):
        encoded = processor(prompt, pil_img)
        inputs = encoded.to(device, dtype=input_dtype)
        with torch.inference_mode():
            # returns numpy action arrays
            action = vla.predict_action(
                **inputs,
                unnorm_key="bridge_orig",
                do_sample=do_sample,
                **sample_kwargs,
            )
        actions.append(action)

    if effective_samples == 1:
        return actions[0]
    return np.stack(actions, axis=0)


def _decode_first_string(value) -> Optional[str]:
    """Recursively extract and decode the first non-empty string/bytes entry."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", errors="ignore").strip()
        return text or None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, np.ndarray):
        # flatten handles ragged/object arrays as well
        for item in value.reshape(-1):
            text = _decode_first_string(item)
            if text:
                return text
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _decode_first_string(item)
            if text:
                return text
    return None


def extract_instruction(example, materialized_steps=None) -> Optional[str]:
    """Probe common fields for language instructions within a Bridge RLDS episode."""
    steps = materialized_steps
    if steps is None and isinstance(example, dict):
        raw_steps = example.get("steps")
        if isinstance(raw_steps, dict):
            steps = raw_steps
        else:
            steps = materialize_steps_structure(raw_steps)

    if isinstance(steps, dict):
        if "language_instruction" in steps:
            text = _decode_first_string(steps["language_instruction"])
            if text:
                return text
        observation = steps.get("observation")
        if isinstance(observation, dict):
            for key in ("language_instruction", "instructions", "natural_language_instruction"):
                if key in observation:
                    text = _decode_first_string(observation[key])
                    if text:
                        return text

    episode_meta = example.get("episode_metadata") if isinstance(example, dict) else None
    if isinstance(episode_meta, dict):
        for key in ("language_instruction", "natural_language_instruction", "instruction"):
            if key in episode_meta:
                text = _decode_first_string(episode_meta[key])
                if text:
                    return text

    return None


def extract_image_sequence(steps) -> Tuple[Optional[str], Optional[np.ndarray]]:
    """Return the full image tensor sequence from steps["observation"]."""
    if not isinstance(steps, dict):
        return None, None
    observation = steps.get("observation")
    if not isinstance(observation, dict):
        return None, None

    candidate_keys = [key for key, value in observation.items() if isinstance(value, np.ndarray)]
    for key in sorted(candidate_keys):
        value = observation[key]
        if value.size == 0:
            continue
        # Expect shape (T, H, W, C) or (H, W, C)
        if value.ndim == 3:
            frames = value[np.newaxis, ...]
        elif value.ndim >= 4:
            frames = value
        else:
            continue
        if frames is None or frames.size == 0:
            continue
        return key, frames

    return None, None

def main(
    rlds_dir,
    num_action_samples: int = 1,
    sample_temperature: Optional[float] = None,
    sample_top_p: Optional[float] = None,
    force_do_sample: bool = False,
):
    # Load dataset (partial / robust mode) using guarded helpers from read_bridgev2
    builder, ds = load_bridge_dataset(rlds_dir, split="all")

    try:
        total_examples = builder.info.splits["all"].num_examples
    except Exception:
        total_examples = None

    # Load OpenVLA model & processor
    quant_bits = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor, vla = load_openvla_quantized(model_name="openvla/openvla-7b", device=device, quant_bits=quant_bits)
    input_dtype = determine_input_dtype(vla, quant_bits, device)

    num_action_samples = max(1, int(num_action_samples))
    sample_kwargs = {}
    if sample_temperature is not None:
        sample_kwargs["temperature"] = float(sample_temperature)
    if sample_top_p is not None and sample_top_p < 1.0:
        sample_kwargs["top_p"] = float(sample_top_p)
    do_sample_flag = bool(force_do_sample or num_action_samples > 1)


    for idx, example in enumerate(tqdm.tqdm(ds, total=total_examples)):
        # Extract instruction
        # OpenVLA expects a natural language instruction.
        steps = example.get("steps") if isinstance(example, dict) else None
        if steps is None:
            print(f"Episode {idx} missing 'steps' field; skipping")
            continue

        steps_struct = materialize_steps_structure(steps)
        if not isinstance(steps_struct, dict) or not steps_struct:
            print(f"Episode {idx} steps could not be materialized; type: {type(steps)}")
            continue

        instruction = extract_instruction(example, steps_struct)
        if instruction is None:
            print(f"Episode {idx} missing instruction; skipping")
            continue

        image_key, image_sequence = extract_image_sequence(steps_struct)
        if image_sequence is None:
            obs_debug = steps_struct.get("observation")
            obs_keys = list(obs_debug.keys()) if isinstance(obs_debug, dict) else "n/a"
            print(f"Episode {idx} missing image observation; keys: {obs_keys}")
            continue

        image_sequence = np.asarray(image_sequence)
        if image_sequence.dtype != np.uint8:
            image_sequence = np.clip(image_sequence, 0, 255).astype(np.uint8)

        # Ground truth
        actions = steps_struct.get("action")
        gt_actions = np.asarray(actions) if actions is not None else None

        # if gt_actions is not None:
        #     rollout_limit = min(len(gt_actions) + 50, image_sequence.shape[0])
        # else:
        rollout_limit = image_sequence.shape[0]

        predicted_actions = []
        for step_idx in range(rollout_limit):
            frame = image_sequence[step_idx]
            try:
                pred_action = run_inference_on_episode(
                    processor,
                    vla,
                    device,
                    instruction,
                    frame,
                    input_dtype,
                    num_samples=num_action_samples,
                    do_sample=do_sample_flag,
                    sample_kwargs=sample_kwargs,
                )
            except Exception as e:
                print(f"Error running OpenVLA on episode {idx}, step {step_idx}: {e}")
                break
            pred_action = np.asarray(pred_action)
            if pred_action.ndim == 1:
                pred_action = pred_action[np.newaxis, :]
            predicted_actions.append(pred_action)

        if not predicted_actions:
            continue

        predicted_actions = np.stack(predicted_actions, axis=0)
        pred_mean = predicted_actions.mean(axis=1)

        print(f"Episode {idx}")
        print("  Instruction:", instruction)
        print(f"  Image key used: {image_key}")
        print(f"  Pred sample tensor shape: {predicted_actions.shape}")

        if gt_actions is not None and gt_actions.size > 0:
            aligned = min(pred_mean.shape[0], len(gt_actions))
            if aligned > 0:
                l2_errors = np.linalg.norm(pred_mean[:aligned] - gt_actions[:aligned], axis=1)
                print(f"  Ground truth shape: {gt_actions.shape}")
                print(f"  Mean L2 error over {aligned} steps: {float(l2_errors.mean()):.6f}")
        else:
            print(
                f"  Predicted {predicted_actions.shape[0]} steps with {predicted_actions.shape[1]} samples each (no ground truth available)."
            )

        # Optionally, compare, log error, or feed into simulator / robot

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds_dir", type=str,
                        default="/data/user/san/datasets/bridge_orig/1.0.0")
    parser.add_argument("--num_action_samples", type=int, default=1,
                        help="Number of action samples to draw per observation frame.")
    parser.add_argument("--sample_temperature", type=float, default=None,
                        help="Optional sampling temperature passed to predict_action when sampling.")
    parser.add_argument("--sample_top_p", type=float, default=None,
                        help="Optional nucleus sampling top_p passed to predict_action when sampling.")
    parser.add_argument("--do_sample", action="store_true",
                        help="Force stochastic sampling even when only one sample is requested.")
    args = parser.parse_args()
    main(
        args.rlds_dir,
        num_action_samples=args.num_action_samples,
        sample_temperature=args.sample_temperature,
        sample_top_p=args.sample_top_p,
        force_do_sample=args.do_sample,
    )
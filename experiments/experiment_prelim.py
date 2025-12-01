
"""
model target : LLM
subtask vector 순차 주입하여 compositional 성능 확인 실험

"""


"""
Experiment for Sequential Patch Testing of Subtask Vectors
=========================================================

This script implements a two‑step activation patching experiment based on
the *Internal Chain-of-Thought: Empirical Evidence for Layer‑wise Subtask
Scheduling in LLMs* paper.  The goal is to extract task vectors for two
subtasks independently and then patch these vectors sequentially into a
language model at different layers to evaluate whether the model can
reconstruct the composite task behaviour without in‑context examples.

Overview
--------

Given a composite task ``t = s1 ∘ s2`` comprised of two subtasks ``s1`` and
``s2``, we perform the following steps:

1. **Subtask Vector Extraction:**  We sample prompts from the respective
   subtask datasets and compute the average residual stream activation at
   every layer of the model when conditioned on in‑context examples.  This
   produces two matrices ``θ_s1`` and ``θ_s2`` of shape ``[n_layers, d_model]``.

2. **Baseline Evaluation:**  For the composite task we build a dataset of
   in‑context prompts and corresponding zero‑shot prompts.  We evaluate
   the model on these prompts to obtain the ICL baseline (with examples)
   and the zero‑shot baseline (no examples).  These baselines provide the
   upper and lower bounds for accuracy.

3. **Sequential Patching:**  For each pair of layers ``(l1, l2)`` we
   sequentially patch the activation vectors ``θ_s1[l1]`` and
   ``θ_s2[l2]`` into the residual stream at layers ``l1`` and ``l2``
   respectively, while running the model on the zero‑shot composite
   prompts.  We record the resulting accuracy on the composite task.

4. **Strength Calculation:**  We normalise the patched accuracy by
   subtracting the zero‑shot baseline and dividing by the difference
   between the ICL baseline and the zero‑shot baseline.  This yields a
   score in ``[0, 1]`` indicating how much of the composite behaviour is
   recovered by sequentially applying the subtask vectors.

This script follows the implementation patterns of ``experiments/experiment_patching.py`` in
the repository but extends it to handle two separate task vectors and
two patch layers.  It uses the utilities defined in ``src/data``,
``src/patching``, ``src/hook`` and ``utils/tools``.

Usage
-----

Run this script from the repository root.  The script will iterate over
all task pairs defined in ``data/list.json`` and across a range of seeds
(default 5).  Intermediate and final results are written to
``outputs/{model_name}/{experiment_name}/{task_name}_<timestamp>/``.

Example command:

.. code-block:: bash

   python experiments/experiment_prelim.py \
       --batch_size 16 \
       --train_data_num 100 \
       --data_num 200 \
       --n_prepended 5 \
       --replace True \
       --patching_mode resid_post \
       --model_name meta-llama/Llama-3.1-8B \
       --device cuda

Note that this experiment can be computationally intensive: it sweeps
over all layer pairs ``(l1, l2)`` in the model, leading to
``n_layers²`` runs per task and seed.  Reduce ``data_num`` or the
number of layers (via ``patching_layers`` arguments) to make it more
tractable on limited hardware.
"""

import os
import sys
from pathlib import Path
import argparse
import json
from functools import partial
from tqdm import tqdm

import torch as t
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

# Adjust the Python path so that we can import from src and utils when
# executing from within the experiments directory.
dir_name = "experiments"
main_dir = Path(f"{os.getcwd().split(dir_name)[0]}").resolve()
if str(main_dir) not in sys.path:
    sys.path.append(str(main_dir))

from src.data.dataset import ICLDataset
from src.data.split_data import split_and_generate
from src.patching import save_activation, base_run
from src.hook import act_patching_hook
from utils.tools import save_parameter, eval_subtask


def compute_task_vector(
    model: HookedTransformer,
    prompts: list[str],
    saving_mode: str,
    pos_ids: int,
    device: t.device,
) -> t.Tensor:
    """Compute the averaged residual activation for a list of prompts.

    This helper wraps ``save_activation`` over a list of prompts and
    averages the resulting activations along the batch dimension.  It
    returns a tensor of shape ``[n_layers, d_model]`` on the specified
    device.

    Args:
        model: The hooked transformer model.
        prompts: A list of prompt strings (ICL prompts) used to
            accumulate activations.
        saving_mode: One of ``'resid_post'``, ``'attn_out'``, or
            ``'mlp_out'``; determines which hook to extract.
        pos_ids: Position index to read from; ``-1`` corresponds to the
            last token.
        device: Device on which to allocate the result.

    Returns:
        A tensor ``[n_layers, d_model]`` representing the average
        activation per layer.
    """
    # ``save_activation`` returns activations aggregated across the
    # provided prompts.  We sum them and divide by the number of
    # prompts to obtain the mean.
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    acc = t.zeros((n_layers, d_model), device=device)
    for prompt in prompts:
        acc += save_activation(model, [prompt], saving_mode, pos_ids)
    acc = acc / max(1, len(prompts))
    return acc


def main(args: argparse.Namespace, model: HookedTransformer, device: t.device) -> None:
    """Run the sequential patching experiment for one composite task.

    For the current ``args.task_name`` (implied by ``args.subtask1_name`` and
    ``args.subtask2_name``), this function computes the subtask vectors,
    evaluates baselines, performs a sweep over layer pairs for sequential
    patching, normalises the results and writes them to disk.

    Args:
        args: Parsed command line arguments controlling the experiment.
        model: Preloaded ``HookedTransformer`` model.
        device: Torch device to use for tensor operations.
    """
    # Log experiment parameters and obtain a directory for outputs.
    log_path = save_parameter(args)

    # Split the base datasets into subtask validation sets and composite
    # training data.  ``s1_data`` and ``s2_data`` are validation lists
    # [(x, y, t1, t2)].  ``composite_data`` contains 4‑tuples of
    # (input, final_output, subtask1_output, subtask2_output).
    s1_data, s2_data, composite_data = split_and_generate(
        args.subtask1_name, args.subtask2_name, args.seed
    )

    # ------------------------------------------------------------------
    # (1) Subtask vector extraction
    # ------------------------------------------------------------------
    # Build datasets for vector extraction.  We use ICL prompts from
    # the respective subtask validation sets.  ``n_prepended`` controls
    # how many examples are prepended for each prompt.  The size of the
    # dataset (``train_data_num``) determines how many prompts we use to
    # accumulate activations.
    dataset_s1 = ICLDataset(
        s1_data,
        size=args.train_data_num,
        n_prepended=args.n_prepended,
        seed=args.seed,
    )
    dataset_s2 = ICLDataset(
        s2_data,
        size=args.train_data_num,
        n_prepended=args.n_prepended,
        seed=args.seed,
    )

    # Use DataLoader for batching prompts.  Batching reduces overhead
    # when calling ``save_activation``.
    loader_s1 = DataLoader(dataset_s1, batch_size=args.batch_size, shuffle=True)
    loader_s2 = DataLoader(dataset_s2, batch_size=args.batch_size, shuffle=True)

    # Initialise accumulators for activations.  Each has shape
    # (n_layers, d_model).
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    s1_act = t.zeros((n_layers, d_model), device=device)
    s2_act = t.zeros((n_layers, d_model), device=device)

    # Accumulate activations for subtask 1
    for batch in tqdm(loader_s1, desc=f"Extracting θ_s1 for {args.task_name}"):
        prompts = batch['prompt']  # list[str]
        s1_act += save_activation(model, prompts, args.patching_mode, args.pos_ids)
    s1_act = s1_act / max(1, args.train_data_num)

    # Accumulate activations for subtask 2
    for batch in tqdm(loader_s2, desc=f"Extracting θ_s2 for {args.task_name}"):
        prompts = batch['prompt']
        s2_act += save_activation(model, prompts, args.patching_mode, args.pos_ids)
    s2_act = s2_act / max(1, args.train_data_num)

    # ------------------------------------------------------------------
    # (2) Baseline evaluation on composite task
    # ------------------------------------------------------------------
    # Build an evaluation dataset from the composite training data.  We
    # use ``generate_zero=True`` to obtain zero‑shot prompts as well as
    # full ICL prompts.
    eval_dataset = ICLDataset(
        composite_data,
        size=args.data_num,
        n_prepended=args.n_prepended,
        seed=args.seed,
        generate_zero=True,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)

    base_result = 0.0  # ICL baseline accuracy accumulator
    corrupted_result = 0.0  # zero‑shot baseline accuracy accumulator

    # Precompute baselines over the evaluation dataset.
    for batch in tqdm(eval_loader, desc=f"Evaluating baselines for {args.task_name}"):
        # Full ICL prompt baseline
        base_prob = base_run(model, batch['prompt'])
        base_result += eval_subtask(model, base_prob, batch, layer=-1, metric='accuracy')[0].item()

        # Zero‑shot baseline (no context examples)
        corrupted_prob = base_run(model, batch['zero_prompt'])
        corrupted_result += eval_subtask(model, corrupted_prob, batch, layer=-1, metric='accuracy')[0].item()

    # Normalise baselines by total number of evaluation samples
    total_samples = len(eval_dataset)
    base_result = base_result / max(1, total_samples)
    corrupted_result = corrupted_result / max(1, total_samples)

    # Avoid division by zero when computing strengths
    denom = max(1e-8, base_result - corrupted_result)

    # ------------------------------------------------------------------
    # (3) Sequential patching sweep
    # ------------------------------------------------------------------
    # Prepare result matrices.  We will collect the raw patched
    # accuracy for each layer pair and then normalise at the end.
    patching_raw = t.zeros((n_layers, n_layers), device=device)

    # For each batch in the evaluation loader we patch activations
    # sequentially.  To avoid re‑registering hooks for every example,
    # we loop over layer pairs inside the batch loop.  Because hooks are
    # cumulative within a model instance, we reset and register them
    # inside the loops.
    for batch in tqdm(eval_loader, desc=f"Sequential patching for {args.task_name}"):
        zero_prompts = batch['zero_prompt']
        # Loop over layer pairs (l1, l2).  We allow l1 ≤ l2 but record
        # all combinations.  Users may later ignore invalid orders if
        # desired.
        for l1 in range(n_layers):
            for l2 in range(n_layers):
                # Reset hooks before adding new ones
                model.reset_hooks()
                # Register the first subtask patch on layer l1
                model.add_hook(
                    f'blocks.{l1}.hook_{args.patching_mode}',
                    partial(act_patching_hook, tar_act=s1_act[l1], replace=args.replace, pos_ids=args.pos_ids),
                )
                # Register the second subtask patch on layer l2
                model.add_hook(
                    f'blocks.{l2}.hook_{args.patching_mode}',
                    partial(act_patching_hook, tar_act=s2_act[l2], replace=args.replace, pos_ids=args.pos_ids),
                )
                # Run the model on the zero‑shot prompts with both
                # patches active.  We use only the final logits for the
                # last token.
                logits = model(zero_prompts)[:, -1, :].softmax(dim=-1)
                # Evaluate only the composite (target) accuracy; index 0
                # corresponds to the final answer.
                patching_raw[l1, l2] += eval_subtask(
                    model,
                    logits,
                    batch,
                    layer=max(l1, l2),
                    metric='accuracy',
                )[0].item()
                # Remove hooks for next iteration
                model.reset_hooks()

    # Normalise patched results by the number of samples
    patching_raw = patching_raw / max(1, total_samples)

    # Compute normalised strengths.  Strength is defined as
    # (patched_acc - zero_shot_acc) / (icls_acc - zero_shot_acc).  It
    # measures how much of the composite behaviour is recovered by
    # sequential patching.
    patching_strength = (patching_raw - corrupted_result) / denom

    # ------------------------------------------------------------------
    # (4) Persist results to disk
    # ------------------------------------------------------------------
    # Write raw accuracies and normalised strengths to JSON files.  The
    # matrices are converted to nested lists for serialisation.
    results = {
        'task': args.task_name,
        'subtask1': args.subtask1_name,
        'subtask2': args.subtask2_name,
        'seed': args.seed,
        'model': args.model_name,
        'baselines': {
            'icls_accuracy': base_result,
            'zero_shot_accuracy': corrupted_result,
        },
        'raw_patching_accuracy': patching_raw.tolist(),
        'patching_strength': patching_strength.tolist(),
    }
    with open(f"{log_path}/sequential_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for DataLoader.')
    parser.add_argument('--train_data_num', default=100, type=int, help='Number of prompts used to extract task vectors.')
    parser.add_argument('--data_num', default=500, type=int, help='Number of prompts used for evaluation.')
    parser.add_argument('--n_prepended', default=5, type=int, help='Number of ICL examples prepended in each prompt.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--replace', default=True, type=bool, help='If True, replace activations; otherwise add.')
    parser.add_argument('--pos_ids', default=-1, type=int, help='Position index to patch; -1 selects last token.')
    parser.add_argument('--patching_mode', default='resid_post', type=str, help='Hook name to patch (resid_post, attn_out, mlp_out).')
    parser.add_argument('--task_name', default='', type=str, help='Composite task name (e.g., antonym_uppercase).')
    parser.add_argument('--subtask1_name', default='', type=str, help='Name of the first subtask.')
    parser.add_argument('--subtask2_name', default='', type=str, help='Name of the second subtask.')
    parser.add_argument('--experiment_name', default='experiment_prelim', type=str, help='Name of this experiment.')
    parser.add_argument('--model_name', default='meta-llama/Llama-3.1-8B', type=str, help='Hugging Face model identifier.')
    parser.add_argument('--device', default='cuda' if t.cuda.is_available() else 'cpu', type=str, help='Computation device.')

    args = parser.parse_args()

    # Load the model once and reuse across tasks.  Disable gradient
    # computation to reduce overhead.
    t.set_grad_enabled(False)
    device = t.device(args.device)
    test_model: HookedTransformer = HookedTransformer.from_pretrained(
        model_name=args.model_name,
        device=device,
        default_padding_side='left',
    )

    # If a single task is specified on the command line, run only that
    # task; otherwise iterate over all tasks defined in ``data/list.json``.
    if args.subtask1_name and args.subtask2_name:
        args.task_name = f"{args.subtask1_name}-{args.subtask2_name}"
        main(args, test_model, device)
    else:
        # Load task pairs from data/list.json
        with open("data/list.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            args.task_name = f"{d['task1']}-{d['task2']}"
            args.subtask1_name = d['task1']
            args.subtask2_name = d['task2']
            for seed in range(5):
                args.seed = seed
                main(args, test_model, device)
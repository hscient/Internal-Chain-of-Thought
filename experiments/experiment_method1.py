"""
Experiment: Method 1 — LogitLens analysis for image–question tasks
------------------------------------------------------------

This script implements a simple LogitLens experiment for compositional
image–question tasks (e.g., CLEVR).  The goal is to probe **where** in
the network a large multimodal model (LMM) first recognises an image
attribute (the "recognition" subtask) and **when** it produces the final
answer (the "answer" subtask).  In other words, we ask: does the model
first decode the concept from the picture (such as the colour or
material) and only later answer the question (e.g., count, compare or
apply some rule)?  This corresponds to the observational evidence
described in the ICoT paper, where hidden states are projected into
token space via a LogitLens and the relative rank or probability of
intermediate vs. final answers is tracked across layers【406210521020237†L900-L930】.

Method overview:

* Each example consists of an image, a question (with a simple
  attribute query such as ``"color: blue"``), and a numerical answer.
* We extract the *recognition concept* from the question by taking the
  string after the colon (e.g. ``"blue"``) and treat the *answer
  concept* as the answer itself (e.g. ``"3"``).
* A multimodal model is loaded via the Hugging Face `transformers` API.
  The processor takes care of converting both the image and text into
  input tensors.  The model is run once per example with
  ``output_hidden_states=True`` so that all intermediate residual
  representations are returned.
* For each transformer layer we project the hidden state at the final
  token position to logits using the model’s language head (LM head).
  A softmax over the vocabulary yields a probability distribution.
* To measure how prominently the recognition concept and the final
  answer appear in the distribution we compute their **mean
  reciprocal rank** (MRR).  The MRR is the inverse of the rank of the
  target token in the sorted logits: 1 for rank 1, 1/2 for rank 2,
  etc.  Averaging MRRs across examples produces a layerwise profile
  similar to Figure 4 in the paper【406210521020237†L900-L930】.
* The script identifies the *crossover layer*, defined as the first
  layer at which the average MRR for the answer exceeds that of the
  recognition concept.  This crossover is indicative of a handoff from
  image recognition to task execution【406210521020237†L900-L930】.

This implementation is intentionally generic: it should work with any
vision–language model that inherits from `AutoModelForCausalLM` and
whose processor can tokenize both images and text.  By default it
loads the "llava-v1.5-7b" checkpoint, but you can supply any other
model via the command line.  If your dataset uses a different
question format or answer representation, adjust the
``extract_concept`` function accordingly.

Usage example:

```bash
python experiments/experiment_method1.py \
  --data_dir data/clevr \             # root containing query.json and images
  --model_name liuhaotian/llava-v1.5-7b \  # any vision-language model
  --device cuda \                     # or "cpu" if GPU is unavailable
  --num_examples 200 \               # number of query examples to process
  --output_file outputs/method1_results.json
```

The resulting JSON file will contain the average MRR per layer for
recognition and answer concepts as well as the inferred crossover
layer.

Note: Running this script on a full CLEVR dataset with a large model
requires a GPU with sufficient memory.  For quick experimentation
consider limiting ``--num_examples`` or using a smaller checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoProcessor


def load_query_examples(data_dir: str, limit: int | None = None) -> List[Dict]:
    """Load query examples from a CLEVR-style dataset directory.

    The directory is expected to contain a ``query.json`` file with
    entries of the form ``{"id": ..., "image": [<path>], "question": ...,
    "answer": <int>}``.  Only the query split is used for method 1
    because in-context demonstrations are not required.

    Args:
        data_dir: Root directory containing ``query.json`` and an
            ``image`` subdirectory.
        limit: Optional maximum number of examples to load.  Useful
            for quick tests.

    Returns:
        A list of dictionaries describing each example.
    """
    json_path = Path(data_dir) / "query.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data


def extract_concept(question: str) -> str:
    """Extract the recognition concept from a question.

    CLEVR questions in this setup take the form ``"attribute: value"``.
    The value after the colon corresponds to the concept to be
    recognised from the image (e.g. colour, material).  If the colon
    pattern is not found, the entire question is returned.  Feel free
    to modify this function to suit your dataset.
    """
    if ":" in question:
        return question.split(":", 1)[1].strip()
    return question.strip()


def compute_mrr(logits: torch.Tensor, token_ids: List[int]) -> float:
    """Compute the mean reciprocal rank (MRR) for a set of token ids.

    Args:
        logits: Unnormalised scores over the vocabulary (1D tensor).
        token_ids: List of target token IDs (e.g. for multi-token
            concepts).  The MRR is computed as the inverse of the
            best (lowest) rank among these tokens.  If none of the
            target tokens are found, returns 0.

    Returns:
        A float between 0 and 1.  Larger values indicate the target
        appears closer to the top of the distribution.
    """
    # Sort descending and compute ranks once
    sorted_indices = torch.argsort(logits, descending=True)
    best_rank = logits.shape[0] + 1  # initialise with worst rank
    for tid in token_ids:
        # Find rank of tid (1-indexed)
        positions = (sorted_indices == tid).nonzero(as_tuple=True)[0]
        if positions.numel() > 0:
            rank = positions.item() + 1
            if rank < best_rank:
                best_rank = rank
    if best_rank <= logits.shape[0]:
        return 1.0 / best_rank
    return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="LogitLens experiment for image–question tasks")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root containing query.json and images")
    parser.add_argument("--model_name", type=str, default="liuhaotian/llava-v1.5-7b", help="HuggingFace model identifier for a vision-language model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device: cuda or cpu")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of query examples to process")
    parser.add_argument("--output_file", type=str, default="method1_results.json", help="Path to write results JSON")
    args = parser.parse_args()

    # Load data
    examples = load_query_examples(args.data_dir, limit=args.num_examples)
    if len(examples) == 0:
        raise RuntimeError(f"No examples found in {args.data_dir}")

    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()
    # Ensure that hidden states are returned
    if not getattr(model.config, "output_hidden_states", False):
        model.config.output_hidden_states = True

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        # Fallback for models with different config naming
        num_layers = getattr(model.config, "num_layers", None)
    if num_layers is None:
        raise ValueError("Unable to determine number of layers from model configuration")

    # Prepare accumulators for MRR per layer
    concept_mrr: List[List[float]] = [[] for _ in range(num_layers)]
    answer_mrr: List[List[float]] = [[] for _ in range(num_layers)]

    for ex in tqdm(examples, desc="Processing examples"):
        # Resolve image path and load image
        image_path = Path(args.data_dir) / ex["image"][0]
        image = Image.open(image_path).convert("RGB")
        question = ex["question"]
        answer = str(ex["answer"])
        concept = extract_concept(question)

        # Tokenise concept and answer once
        concept_ids = processor.tokenizer(concept, add_special_tokens=False).input_ids
        answer_ids = processor.tokenizer(answer, add_special_tokens=False).input_ids

        # Prepare input; note that processor expects a list of images and a list of texts of equal length
        inputs = processor(images=[image], text=[question], return_tensors="pt").to(args.device)

        with torch.no_grad():
            outputs = model(**inputs)
        hidden_states = outputs.hidden_states  # tuple of (embedding_output, layer1_output, ..., layerL_output)
        # We ignore the embedding output at index 0
        for layer_idx in range(1, len(hidden_states)):
            # hidden state shape: (batch_size=1, seq_len, hidden_size)
            hs = hidden_states[layer_idx][0]  # remove batch dimension
            # Use the final token position (last generated token) for decoding
            # The processor pads text and image tokens such that the question is last
            last_pos = hs.shape[0] - 1
            residual = hs[last_pos]
            # Project via LM head
            logits = model.lm_head(residual)
            # Compute MRR for concept and answer
            mrr_c = compute_mrr(logits, concept_ids)
            mrr_a = compute_mrr(logits, answer_ids)
            concept_mrr[layer_idx - 1].append(float(mrr_c))
            answer_mrr[layer_idx - 1].append(float(mrr_a))

    # Aggregate results
    avg_concept_mrr = [float(np.mean(mrs)) if mrs else 0.0 for mrs in concept_mrr]
    avg_answer_mrr = [float(np.mean(mrs)) if mrs else 0.0 for mrs in answer_mrr]

    # Determine crossover layer: first layer where answer MRR exceeds concept MRR
    crossover_layer = None
    for idx, (cmrr, amrr) in enumerate(zip(avg_concept_mrr, avg_answer_mrr)):
        if amrr > cmrr:
            crossover_layer = idx
            break

    results = {
        "model": args.model_name,
        "num_examples": len(examples),
        "avg_concept_mrr": avg_concept_mrr,
        "avg_answer_mrr": avg_answer_mrr,
        "crossover_layer": crossover_layer,
    }

    # Write results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved LogitLens results to {output_path}")


if __name__ == "__main__":
    main()

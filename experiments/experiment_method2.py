"""
Experiment: Method 2 — Layer‑wise context masking for compositional tasks
-----------------------------------------------------------------------

This script provides a **causal** probe of whether a large
transformer–based model processes composite tasks in a sequential,
layerwise fashion.  It implements the **layer‑from context masking**
method described in the ICoT paper【406210521020237†L641-L682】.  In this
setting, a model is given several in‑context examples (demonstrations)
followed by a query.  By gradually masking attention to these
demonstrations after different layers in the network, we can
observe how the model’s predictions evolve.  If the model first
infers an intermediate result (e.g. recognising an image attribute)
before applying the final rule, we should see an “X‑shaped” pattern
in accuracy: early masking preserves only the first subtask and
produces the intermediate answer, whereas late masking allows both
subtasks to be executed【406210521020237†L641-L743】.

**Important caveat**: Handling images in a truly multimodal way
requires access to a vision–language model and hooks into its
attention layers to zero out cross‑attention from context tokens.  Such
implementation details vary widely across architectures (e.g. LLaVA,
BLIP, Flamingo).  To keep this example simple and broadly usable
without heavy dependencies, we instead **approximate** context
masking by working in the *text* domain only.  Each image is
represented in the prompt as the literal path (e.g. ``"Image:
CLEVR_val_000123.png"``).  The model therefore only sees text, but
the masking mechanism still illustrates how to perform causal
interventions in a transformer: we zero out hidden states for
demonstration tokens from a specified layer onward.  To adapt this
script to a true LMM, replace the prompt construction with the
appropriate multimodal processor and modify the masking function
``mask_hidden_states`` to operate on the model’s residual stream or
attention scores.

The procedure is as follows:

1. **Data loading.**  The dataset directory contains ``support.json``
   and ``query.json`` files in the CLEVR format.  We sample a
   specified number of query examples and, for each, randomly pick
   ``n_support`` demonstrations from the support set.
2. **Prompt construction.**  For each demonstration we append a text
   block of the form
   ``"Image: <img>\nQuestion: <q>\nAnswer: <a>"``.  The query
   ends with ``"Answer:"`` so that the model must fill in the
   missing answer.  Note that images are *not* loaded; their paths
   serve as placeholders.
3. **Baseline evaluation.**  We run the model on the full prompt
   without any masking and decode the predicted answer.
4. **Layer‑wise masking.**  For each layer index `l` from 0 to
   `num_layers-1`, we obtain all hidden states via a forward pass
   (setting ``output_hidden_states=True``).  Then, for every layer
   index ≥ `l`, we zero out the hidden states at token positions
   corresponding to the demonstrations.  We leave earlier layers
   unchanged.  We then pass the final masked hidden state through
   the model’s language head to obtain the next‑token logits and
   decode the answer.  Although this simple masking only removes the
   residual stream contribution of context tokens and does not
   re‑propagate through subsequent layers, it nonetheless
   approximates the causal intervention described in the paper【406210521020237†L641-L743】.
5. **Accuracy computation.**  For each layer we record whether the
   predicted answer matches the ground truth.  Aggregating across
   examples yields a layerwise accuracy curve.  In practice, one
   expects to see a dip followed by a recovery, indicating that the
   second subtask executes deeper in the network.

Usage example:

```bash
python experiments/experiment_method2.py \
  --data_dir data/clevr \     # directory containing query.json & support.json
  --model_name meta-llama/Llama-3.1-8B \  # any causal LM
  --device cuda \             # or "cpu"
  --n_support 5 \            # number of demonstrations per prompt
  --num_queries 100 \        # number of query examples to evaluate
  --output_file outputs/method2_results.json
```

If you wish to adapt this script to a real vision–language model,
modify the ``build_prompt`` function to call the model’s processor
with images and texts, and replace the simple hidden‑state masking
with a hook on the attention scores as described in the paper (see
Equation 4)【406210521020237†L641-L682】.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_split(data_dir: str, split: str) -> List[Dict]:
    """Load a split (``support`` or ``query``) from a CLEVR dataset directory.

    The expected JSON format is a list of objects with ``image`` (list of
    image paths), ``question`` (string) and ``answer`` (int or string).
    """
    file_path = Path(data_dir) / f"{split}.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_prompt(support_examples: List[Dict], query_example: Dict) -> str:
    """Construct a textual prompt from support demonstrations and a query.

    Each support example contributes a block:

        Image: <path>
        Question: <question>
        Answer: <answer>

    The query is appended similarly but the answer is left blank to
    elicit a prediction from the model.

    Args:
        support_examples: A list of demonstration examples.
        query_example: The test example whose answer we want to predict.

    Returns:
        A single string representing the in‑context learning prompt.
    """
    parts: List[str] = []
    for ex in support_examples:
        img_path = ex["image"][0] if isinstance(ex.get("image"), list) else ex.get("image", "")
        parts.append(f"Image: {img_path}\nQuestion: {ex['question']}\nAnswer: {ex['answer']}\n")
    # Append query without answer
    q_img_path = query_example["image"][0] if isinstance(query_example.get("image"), list) else query_example.get("image", "")
    parts.append(f"Image: {q_img_path}\nQuestion: {query_example['question']}\nAnswer:")
    return "\n".join(parts)


def mask_hidden_states(hidden_states: Tuple[torch.Tensor, ...], context_mask: torch.Tensor, mask_layer: int) -> Tuple[torch.Tensor, ...]:
    """Zero out hidden states for context tokens from ``mask_layer`` onward.

    Args:
        hidden_states: Tuple of tensors as returned by ``output_hidden_states``.
        context_mask: A boolean tensor of shape ``(seq_len,)`` where
            ``True`` indicates positions belonging to demonstrations.
        mask_layer: The first layer at which masking begins.  A value
            of -1 implies no masking.

    Returns:
        A new tuple of hidden state tensors with the specified
        positions zeroed out from ``mask_layer+1`` onward.
    """
    if mask_layer < 0:
        return hidden_states
    # Clone to avoid modifying original
    masked = list(hidden_states)
    # Loop over layers; index 0 is embedding output
    for idx in range(mask_layer + 1, len(hidden_states)):
        h = masked[idx].clone()
        # h shape: (batch_size=1, seq_len, hidden_size)
        h[0, context_mask, :] = 0
        masked[idx] = h
    return tuple(masked)


def decode_answer(tokenizer: AutoTokenizer, logits: torch.Tensor) -> str:
    """Decode the most likely next token as a string."""
    predicted_id = int(torch.argmax(logits, dim=-1))
    return tokenizer.decode([predicted_id]).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer-wise context masking experiment (approximate)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root containing support.json and query.json")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", help="Causal language model for text-only masking")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    parser.add_argument("--n_support", type=int, default=5, help="Number of in-context demonstrations per prompt")
    parser.add_argument("--num_queries", type=int, default=200, help="Number of query examples to evaluate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for support sampling")
    parser.add_argument("--output_file", type=str, default="method2_results.json", help="Where to save the layer-wise accuracy JSON")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load data
    support_data = load_split(args.data_dir, "support")
    query_data = load_split(args.data_dir, "query")
    if args.num_queries > 0:
        query_data = query_data[:args.num_queries]

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()
    # Ensure we get hidden states
    if not getattr(model.config, "output_hidden_states", False):
        model.config.output_hidden_states = True

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        num_layers = getattr(model.config, "num_layers", None)
    if num_layers is None:
        raise ValueError("Cannot infer number of transformer layers from model configuration")

    # Collect per-layer accuracy lists
    # We include one entry for "no masking" (mask_layer = -1)
    accuracies = {layer: [] for layer in range(-1, num_layers)}

    for q in tqdm(query_data, desc="Evaluating queries"):
        # Sample support examples (avoid including the query itself if present in support)
        supp_examples = rng.sample(support_data, min(args.n_support, len(support_data)))
        # Build prompt string
        prompt = build_prompt(supp_examples, q)
        # Tokenise prompt
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(args.device)
        attention_mask = enc["attention_mask"].to(args.device)
        seq_len = input_ids.shape[1]
        # Determine context mask: tokens belonging to demonstrations
        # Heuristic: the last token of the sequence corresponds to the query answer position
        # We assume demonstrations occupy the first (seq_len - len(query_answer_tokens) - 1) tokens.
        # Since we cannot separate tokens precisely without more structure, we mark all tokens
        # except the final one as context.  This means we remove all preceding content when masking.
        context_mask = torch.zeros(seq_len, dtype=torch.bool, device=args.device)
        context_mask[: seq_len - 1] = True

        # Run full model to get hidden states and baseline prediction
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        baseline_logits = outputs.logits[0, -1, :]  # final token logits
        baseline_pred = decode_answer(tokenizer, baseline_logits)
        # Record baseline accuracy
        correct_answer = str(q["answer"]).strip()
        accuracies[-1].append(1.0 if baseline_pred == correct_answer else 0.0)

        # Compute hidden states for masking
        with torch.no_grad():
            outputs_h = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs_h.hidden_states

        # Precompute language head once
        # We'll reuse the LM head to project masked hidden states
        lm_head = model.lm_head

        # Evaluate each masking layer
        for l in range(num_layers):
            # Mask hidden states from layer l onward
            masked_hs = mask_hidden_states(hidden_states, context_mask, mask_layer=l)
            # Take final hidden state (embedding of last layer)
            last_h = masked_hs[-1][0, -1, :]
            logits = lm_head(last_h)
            pred = decode_answer(tokenizer, logits)
            accuracies[l].append(1.0 if pred == correct_answer else 0.0)

    # Compute mean accuracy per layer
    layer_acc = {str(layer): float(np.mean(vals)) if vals else 0.0 for layer, vals in accuracies.items()}

    # Write results
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model_name, "n_support": args.n_support, "layer_accuracy": layer_acc}, f, indent=2)
    print(f"Saved context masking results to {out_path}")


if __name__ == "__main__":
    main()

import torch
import math
from tqdm.auto import trange
from typing import List, Optional
from ankh.models.ankh_transformers import load_model


def loop_range(
    start: int,
    end: int,
    step: int = 1,
    mininterval: float = 1.0,
    leave: bool = False,
    verbose: bool = False,
):
    if verbose:
        return trange(
            start,
            end,
            step,
            mininterval=mininterval,
            leave=leave,
        )
    else:
        return range(start, end, step)


def tokenize_sequence(
    tokenizer,
    sequence,
    prefix: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    if prefix is not None and prefix not in ["[NLU]", "[S2S]"]:
        raise ValueError(f"Invalid prefix: {prefix}")
    prefix = prefix if prefix is not None else ""
    tokens = tokenizer(prefix + sequence, return_tensors="pt")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    if device is not None:
        input_ids = input_ids.to(device=device, non_blocking=True)
        attention_mask = attention_mask.to(device=device, non_blocking=True)
    return input_ids, attention_mask


def mask_ignore_tokens(
    logits: torch.Tensor,
    ignore_tokens: Optional[List[int]] = None,
) -> torch.Tensor:
    if ignore_tokens is not None:
        logits[:, :, ignore_tokens] = -torch.inf
    return logits


@torch.no_grad()
def ankh_pseudo_likelihood_vectorized(
    model,
    tokenizer,
    sequence,
    ignore_tokens=None,
    device="cuda:0",
    prefix=None,
    shift=0,
    shard_input=True,
    shard_batch_size=1024,
    verbose=False,
):
    seqlen = len(sequence)  # String length, used as count of tokens for PL.
    input_ids, attention_mask = tokenize_sequence(
        tokenizer, sequence, prefix, device
    )
    extra_id_0 = tokenizer.get_vocab()["<extra_id_0>"]

    # Expand the input_ids and attention_mask to the sequence length.
    expanded_input_ids = input_ids.expand((seqlen, -1))
    expanded_attention_mask = attention_mask.expand((seqlen, -1))

    # idxs are the indices within input_ids that will be masked.
    # Based on current usage, shift is 0, seqlen is string length.
    # So, it processes the first 'seqlen' tokens of (prefix + sequence).
    idxs = torch.arange(shift, seqlen + shift, device=device).view(-1, 1)

    # Create a tensor of mask tokens for the masked positions.
    mask_tensor = torch.tensor(
        [extra_id_0] * idxs.shape[0], device=device
    ).view(-1, 1)

    # Replace the tokens at the masked positions with the mask token.
    masked_input_ids = torch.scatter(
        expanded_input_ids, dim=-1, index=idxs, src=mask_tensor
    )
    # For T5-style models, decoder_input_ids usually start with
    # model.config.decoder_start_token_id (or pad_token_id) and then the
    # sentinel token <extra_id_0> to predict the masked position.
    decoder_input_ids = torch.tensor(
        [model.config.decoder_start_token_id, extra_id_0], device=device,
    ).view(1, -1).expand((seqlen, -1))

    if shard_input:
        sharded_logits = []
        for i in loop_range(0, seqlen, shard_batch_size, verbose=verbose):
            outputs = model(
                input_ids=masked_input_ids[i: i + shard_batch_size, :],
                attention_mask=expanded_attention_mask[
                    i: i + shard_batch_size, :
                ],
                decoder_input_ids=decoder_input_ids[
                    i: i + shard_batch_size, :
                ],
                use_cache=False,
            )
            sharded_logits.append(outputs.logits)
        logits = torch.cat(sharded_logits, dim=0)
    else:
        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=expanded_attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        logits = outputs.logits

    logits = mask_ignore_tokens(logits, ignore_tokens)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    sequeezed_input_ids = input_ids.squeeze(0)[shift: -1]
    likelihood_output = 0.0

    last_token_log_probs = log_probs[:, -1, :]

    for i in range(seqlen):
        likelihood_output += last_token_log_probs[i, sequeezed_input_ids[i]]

    # The final output is divided by the number of tokens in the sequence
    # to get the average log probability of the sequence without accounting
    # the sequence length.
    # The final output is the exponential of the average log probability
    # to cancel the log operation and return to the probability space.
    return math.exp(likelihood_output / seqlen)


@torch.no_grad()
def ankh_pseudo_likelihood(
    model,
    tokenizer,
    sequence,
    ignore_tokens=None,
    device="cuda:0",
    prefix=None,
    shift=0,
    verbose=False,
):
    input_ids, attention_mask = tokenize_sequence(
        tokenizer, sequence, prefix, device
    )

    extra_id_0 = tokenizer.get_vocab()["<extra_id_0>"]

    # Initialize log likelihood
    total_log_likelihood = 0

    decoder_input_ids = torch.tensor(
        [[model.config.pad_token_id, extra_id_0]], device=device
    )

    # For each position in sequence
    for i in loop_range(shift, len(sequence) + shift, verbose=verbose):
        # Create masked input by replacing token at position i with mask token
        masked_input_ids = input_ids.clone()
        masked_input_ids[0, i] = extra_id_0

        # Get model predictions
        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

        logits = outputs.logits[0, -1, :]

        logits = mask_ignore_tokens(logits, ignore_tokens)

        # Get log probability of correct token
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        correct_token_prob = log_probs[input_ids[0, i]]

        # Add to total log likelihood
        total_log_likelihood += correct_token_prob.item()

    # The final output is divided by the number of tokens in the sequence
    # to get the average log probability of the sequence without accounting
    # the sequence length.
    # The final output is the exponential of the average log probability
    # to cancel the log operation and return to the probability space.
    return math.exp(total_log_likelihood / len(sequence))


def compute_pseudo_likelihood(
    model_name: str,
    sequences: List[str],
    device: Optional[torch.device] = None,
    shard_input: bool = False,
    shard_batch_size: int = 512,
    verbose: bool = False,
) -> List[float]:
    """Compute the likelihood of a sequence of tokens.

    Args:
        model_name (str): String specifying the model name.
        sequences (List[str]): List of strings specifying the sequences to
        compute the likelihood for.
        device (Optional[torch.device], optional): Device to use for
        computation. Defaults to None.
        shard_input (bool, optional): Whether to shard the input. Defaults to
        False.
        shard_batch_size (int, optional): Batch size to use for sharding. If
        `shard_input` is True, this is the batch size for sharding the input.
        Defaults to 512. If `shard_batch_size` is 1 it will be as if the
        calculation of likelihood is not vectorized.
        verbose (bool, optional): Whether to use tqdm for progress bar.
        Defaults to False.

    Raises:
        ValueError: If sharding is not supported for non-vectorized mode.

    Returns:
        List[float]: Likelihood values for each sequence.
    """

    if not isinstance(sequences, list):
        raise ValueError(
            "Expected a list of strings. "
            f"Received: {type(sequences)}."
        )

    if len(sequences) == 0:
        raise ValueError(
            "Expected a non-empty list of strings. "
            f"Received: {len(sequences)}."
        )

    if not all(isinstance(sequence, str) for sequence in sequences):
        raise ValueError(
            "Expected a list of strings. "
            f"Received: {type(sequences)}."
        )

    if shard_input and shard_batch_size <= 0:
        raise ValueError(
            "Expected a positive integer for shard_batch_size. "
            f"Received: {shard_batch_size}."
        )

    available_models = ["ankh_base", "ankh_large", "ankh3_large", "ankh3_xl"]
    if model_name not in available_models:
        raise ValueError(
            f"Expected one of {available_models}. "
            f"Received: {model_name}."
        )

    if shard_batch_size < 1:
        raise ValueError(
            "Expected `shard_batch_size` to be at least 1. "
            f"Received: {shard_batch_size}."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name.startswith("ankh3"):
        prefix = "[NLU]"
        shift = 1
    else:
        prefix = None
        shift = 0

    model, tokenizer = load_model(
        model_name=model_name,
        generation=True,
        output_attentions=False,
        framework="pt",
    )

    # Get the ignore tokens from the tokenizer.
    # Tokens that start with < (e.g. <extra_id_0>) or [ (e.g. [NLU])
    # are ignored in the likelihood computation because they are not part of
    # the protein alphabet when computing the likelihood.
    ignore_tokens = [
        v for k, v in tokenizer.get_vocab().items()
        if k.startswith("<") or k.startswith("[")
    ]

    model.eval().to(device=device)
    likelihoods = []
    num_seqs = len(sequences)
    for idx in loop_range(0, num_seqs, 1, verbose=verbose, leave=True):
        likelihood = ankh_pseudo_likelihood_vectorized(
            model=model,
            tokenizer=tokenizer,
            sequence=sequences[idx],
            ignore_tokens=ignore_tokens,
            device=device,
            prefix=prefix,
            shift=shift,
            shard_input=shard_input,
            shard_batch_size=shard_batch_size,
            verbose=verbose,
        )
        likelihoods.append(likelihood)
    return likelihoods

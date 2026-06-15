import os
import json
import logging
from typing import List, Optional, Dict, Set, Tuple, Any

import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd

import esm
from safetensors.torch import load_file

from .tokenizer import PhyloTokenizer
from .model import ESMFeatureExtractor, TaxonomyTransformerDecoder, apply_lora_to_model

logger = logging.getLogger(__name__)

# ------------------------- Defaults (overridden by config.json via init) -------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ESM_MODEL_NAME = 'esm2_t33_650M_UR50D'
TAXONOMY_SEQ_LENGTH = 118
VOCAB_SIZE = 15000
DECODER_LAYERS = 4
DECODER_HEADS = 8
DROPOUT = 0.1
LORA_RANK = 8
LORA_ALPHA = 16
LORA_TARGET_MODULES = ['query', 'key', 'value']
MAX_SEQ_LEN_ESM = 1022
BOS_TOKEN_ID = 1  # CONT id; verified correct for greedy decoding

# ------------------------- Model state (set by init) -------------------------
_esmmodel = None
_alphabet = None
_feature_extractor = None
_decoder = None
_tokenizer: PhyloTokenizer
_parent_to_children: Optional[Dict[str, set]] = None


def _load_parent_to_children(parent_child_csv: str):
    global _parent_to_children
    if not parent_child_csv or not os.path.exists(parent_child_csv):
        _parent_to_children = None
        return
    df = pd.read_csv(parent_child_csv)
    mapping = {v['Parent']: [s.strip() for s in str(v['Children']).split(',')]
               for _, v in df.iterrows()}
    _parent_to_children = {p: set(children) for p, children in mapping.items()}


def init(model_dir: str, device: Optional[str] = None):
    """Load model + tokenizer + taxonomy maps from a directory containing
    config.json, model.safetensors, phylo2_mapping.json, parent_to_child_mapping.csv.
    Base ESM-2 weights are downloaded via the `esm` package and the trained delta
    (LoRA + pooling + decoder) is loaded from model.safetensors."""
    global DEVICE, ESM_MODEL_NAME, TAXONOMY_SEQ_LENGTH, VOCAB_SIZE, DECODER_LAYERS
    global DECODER_HEADS, DROPOUT, LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES
    global MAX_SEQ_LEN_ESM, BOS_TOKEN_ID
    global _esmmodel, _alphabet, _feature_extractor, _decoder, _tokenizer

    if device:
        DEVICE = device
    with open(os.path.join(model_dir, 'config.json')) as f:
        cfg = json.load(f)
    ESM_MODEL_NAME = cfg.get('esm_model_name', ESM_MODEL_NAME)
    TAXONOMY_SEQ_LENGTH = cfg.get('taxonomy_seq_length', TAXONOMY_SEQ_LENGTH)
    VOCAB_SIZE = cfg.get('vocab_size', VOCAB_SIZE)
    DECODER_LAYERS = cfg.get('decoder_layers', DECODER_LAYERS)
    DECODER_HEADS = cfg.get('decoder_heads', DECODER_HEADS)
    DROPOUT = cfg.get('dropout', DROPOUT)
    LORA_RANK = cfg.get('lora_rank', LORA_RANK)
    LORA_ALPHA = cfg.get('lora_alpha', LORA_ALPHA)
    LORA_TARGET_MODULES = cfg.get('lora_target_modules', LORA_TARGET_MODULES)
    MAX_SEQ_LEN_ESM = cfg.get('max_seq_len_esm', MAX_SEQ_LEN_ESM)
    BOS_TOKEN_ID = cfg.get('bos_token_id', BOS_TOKEN_ID)

    _load_parent_to_children(os.path.join(model_dir, 'parent_to_child_mapping.csv'))

    _tokenizer = PhyloTokenizer(mapping_file=os.path.join(model_dir, 'phylo2_mapping.json'))

    _esmmodel, _alphabet = esm.pretrained.load_model_and_alphabet(ESM_MODEL_NAME)
    _esmmodel.eval()
    for p in _esmmodel.parameters():
        p.requires_grad = False
    if LORA_RANK > 0:
        _esmmodel = apply_lora_to_model(_esmmodel, LORA_RANK, LORA_ALPHA, LORA_TARGET_MODULES)

    _feature_extractor = ESMFeatureExtractor(_esmmodel)
    _decoder = TaxonomyTransformerDecoder(
        embed_dim=_esmmodel.embed_dim, vocab_size=VOCAB_SIZE,
        num_layers=DECODER_LAYERS, num_heads=DECODER_HEADS, dropout=DROPOUT,
        seq_length=TAXONOMY_SEQ_LENGTH, level_positions=_tokenizer.level_positions)
    _feature_extractor.to(DEVICE)
    _decoder.to(DEVICE)
    if DEVICE == 'cuda' and torch.cuda.is_bf16_supported():
        _feature_extractor.bfloat16()
        _decoder.bfloat16()

    # load the trained delta (LoRA + pooling + decoder); base ESM stays as package weights
    _load_checkpoint_into_models(os.path.join(model_dir, 'model.safetensors'),
                                 _feature_extractor, _decoder)
    _feature_extractor.eval()
    _decoder.eval()
    _init_special_ids()


def _load_checkpoint_into_models(ckpt_base_path: str, feature_extractor: torch.nn.Module, decoder: torch.nn.Module):
    def _try_load_safetensors(path: str):
        try:
            return load_file(path)
        except Exception:
            return None

    def _try_load_torch(path: str):
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            return ckpt.get('state_dict', ckpt)
        except Exception:
            return None

    candidates = []
    if os.path.exists(ckpt_base_path):
        candidates.append(ckpt_base_path)
    for ext in ['.safetensors', '.ckpt']:
        p = ckpt_base_path + ext
        if os.path.exists(p):
            candidates.append(p)

    state_dict = None
    for p in candidates:
        if p.endswith('.safetensors') or ('.safetensors' in p):
            state_dict = _try_load_safetensors(p)
            if state_dict is not None:
                break
        if state_dict is None and os.path.isfile(p):
            st = _try_load_safetensors(p)
            if st is not None:
                state_dict = st
                break
        if state_dict is None:
            st = _try_load_torch(p)
            if st is not None:
                state_dict = st
                break

    if state_dict is None:
        raise FileNotFoundError(
            f"Failed to load checkpoint from any of: {candidates}")

    fe_prefix = 'feature_extractor.'
    dec_prefix = 'decoder.'
    fe_state = {k[len(fe_prefix):]: v for k,
                v in state_dict.items() if k.startswith(fe_prefix)}
    dec_state = {k[len(dec_prefix):]: v for k,
                 v in state_dict.items() if k.startswith(dec_prefix)}

    feature_extractor.load_state_dict(fe_state, strict=False)
    decoder.load_state_dict(dec_state, strict=False)


# ------------------------- Helpers -------------------------
SPECIAL_IDS = {}
SLOT_ALLOWED_IDS_CACHE: Optional[Dict[int, List[int]]] = None
SLOT_ALLOWED_IDX_TENSOR_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def _init_special_ids():
    global SPECIAL_IDS
    SPECIAL_IDS = {
        'PAD': _tokenizer.PAD_ID,
        'CONT': _tokenizer.CONT_TOKEN_ID,
        'ABSENT': _tokenizer.ABSENT_ID,
    }


def _truncate_sequence(seq: str, max_len: int) -> str:
    return seq[:max_len]


def _esm_tokenize_single(sequence: str, alphabet):
    batch_converter = alphabet.get_batch_converter()
    _, _, toks = batch_converter([("protein", sequence)])
    return toks.to(DEVICE)


def _esm_tokenize_batch(sequences: List[str], alphabet) -> torch.Tensor:
    """Tokenize multiple sequences for ESM in a single batch."""
    batch_converter = alphabet.get_batch_converter()
    batch_data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
    _, _, batch_tokens = batch_converter(batch_data)
    return batch_tokens.to(DEVICE)


@torch.no_grad()
def predict_taxonomy_batch_true(sequences: List[str], max_batch_size: int = 32) -> List[Dict[str, Any]]:
    """
    True batched inference: process multiple sequences simultaneously on GPU.

    Args:
        sequences: List of protein sequences
        max_batch_size: Maximum batch size for GPU memory management

    Returns:
        List of dicts with keys: 'decoded_taxonomy', 'decoder_embeddings', 'success', 'error'
    """
    if not sequences:
        return []

    # Truncate sequences
    truncated_sequences = [_truncate_sequence(
        seq, MAX_SEQ_LEN_ESM) for seq in sequences]

    # Process in chunks to manage GPU memory
    all_results = []

    for batch_start in range(0, len(truncated_sequences), max_batch_size):
        batch_end = min(batch_start + max_batch_size, len(truncated_sequences))
        batch_sequences = truncated_sequences[batch_start:batch_end]

        try:
            # Tokenize batch
            batch_tokens = _esm_tokenize_batch(batch_sequences, _alphabet)
            batch_size, seq_len = batch_tokens.shape

            # Extract features for entire batch
            batch_features = _feature_extractor(
                batch_tokens)  # (batch_size, embed_dim)

            # Ensure proper dtype for bfloat16 inference (matching training)
            decoder_dtype = next(_decoder.parameters()).dtype
            if decoder_dtype == torch.bfloat16:
                batch_features = batch_features.bfloat16()
            elif decoder_dtype == torch.float16:
                batch_features = batch_features.half()

            # Decode each sequence (decoder is currently single-sequence)
            batch_results = []
            for i in range(batch_size):
                try:
                    # Get features for this sequence
                    features = batch_features[i:i+1]  # Keep batch dimension

                    # Generate taxonomy with embeddings
                    memory_expanded = features.unsqueeze(1)
                    generated_ids = torch.full(
                        (1, TAXONOMY_SEQ_LENGTH), -1, dtype=torch.long, device=features.device)
                    decoder_embeddings_list = []

                    # Start with BOS token
                    current_input_sequence = torch.full(
                        (1, 1), _tokenizer.CONT_TOKEN_ID, dtype=torch.long, device=features.device)

                    # Greedy decoding with embedding extraction
                    for step in range(TAXONOMY_SEQ_LENGTH):
                        current_len = current_input_sequence.size(1)

                        if current_len > _decoder.positional_embedding.num_embeddings:
                            break

                        positions = torch.arange(
                            0, current_len, device=features.device).unsqueeze(0)
                        tgt_emb = _decoder.token_embedding(
                            current_input_sequence) + _decoder.positional_embedding(positions)

                        causal_mask = nn.Transformer.generate_square_subsequent_mask(
                            current_len, device=features.device)

                        # Get decoder output (embeddings)
                        decoder_output = _decoder.decoder(
                            tgt=tgt_emb,
                            memory=memory_expanded,
                            tgt_mask=causal_mask
                        )

                        # Store embedding
                        decoder_embeddings_list.append(
                            decoder_output[:, -1, :])

                        # Get next token
                        logits = _decoder.output_head(decoder_output[:, -1, :])
                        next_token = logits.argmax(dim=-1)

                        generated_ids[:, step] = next_token
                        current_input_sequence = torch.cat(
                            (current_input_sequence, next_token.unsqueeze(1)), dim=1)

                    # Stack embeddings and decode
                    decoder_embeddings = torch.stack(decoder_embeddings_list, dim=1)[
                        0]  # (seq_length, embed_dim)
                    token_ids = generated_ids[0].tolist()
                    taxonomy_terms = _tokenizer.decode(token_ids)

                    batch_results.append({
                        'decoded_taxonomy': taxonomy_terms,
                        'decoder_embeddings': decoder_embeddings,
                        'success': True,
                        'error': None
                    })

                except Exception as e:
                    batch_results.append({
                        'decoded_taxonomy': [],
                        'decoder_embeddings': torch.zeros(TAXONOMY_SEQ_LENGTH, _decoder.embed_dim),
                        'success': False,
                        'error': str(e)
                    })

            all_results.extend(batch_results)

        except Exception as e:
            # If batch processing fails, return failed results for this batch
            for _ in range(len(batch_sequences)):
                all_results.append({
                    'decoded_taxonomy': [],
                    'decoder_embeddings': torch.zeros(TAXONOMY_SEQ_LENGTH, _decoder.embed_dim),
                    'success': False,
                    'error': str(e)
                })

    return all_results


@torch.no_grad()
def predict_taxonomy_for_sequence(sequence: str) -> List[str]:
    sequence = _truncate_sequence(sequence, MAX_SEQ_LEN_ESM)
    toks = _esm_tokenize_single(sequence, _alphabet)
    features = _feature_extractor(toks)
    generated_ids = _decoder(features, target_tokens=None,
                             bos_token_id=_tokenizer.CONT_TOKEN_ID)
    token_ids = generated_ids[0].tolist()
    taxonomy_terms = _tokenizer.decode(token_ids)
    return taxonomy_terms


def _build_bins_per_level(level_positions: List[int]) -> Dict[int, List[int]]:
    bins_per_level: Dict[int, List[int]] = {}
    for pos, lvl in enumerate(level_positions):
        bins_per_level.setdefault(lvl, []).append(pos)
    return bins_per_level


def _build_slot_allowed_ids_cache() -> Dict[int, List[int]]:
    cache: Dict[int, List[int]] = {}
    for (slot_idx, tok_id), _name in _tokenizer.slot_token_map.items():
        cache.setdefault(slot_idx, []).append(tok_id)
    return cache


def _allowed_token_ids_for_slot(slot_idx: int) -> List[int]:
    global SLOT_ALLOWED_IDS_CACHE
    if SLOT_ALLOWED_IDS_CACHE is None:
        SLOT_ALLOWED_IDS_CACHE = _build_slot_allowed_ids_cache()
    return SLOT_ALLOWED_IDS_CACHE.get(slot_idx, [])


def _allowed_idx_tensor(slot_idx: int, device: torch.device) -> torch.Tensor:
    key = (slot_idx, str(device))
    if key in SLOT_ALLOWED_IDX_TENSOR_CACHE:
        return SLOT_ALLOWED_IDX_TENSOR_CACHE[key]
    ids = _allowed_token_ids_for_slot(slot_idx)
    t = torch.tensor(ids, dtype=torch.long, device=device) if ids else torch.empty(
        0, dtype=torch.long, device=device)
    SLOT_ALLOWED_IDX_TENSOR_CACHE[key] = t
    return t


@torch.no_grad()
def _next_step_logprobs(features: torch.Tensor, partial_ids: List[int]) -> torch.Tensor:
    cur_len = len(partial_ids)
    if cur_len == 0:
        tgt = torch.tensor([[_tokenizer.PAD_ID]],
                           dtype=torch.long, device=features.device)
    else:
        tgt_ids = partial_ids + [_tokenizer.PAD_ID]
        tgt = torch.tensor([tgt_ids], dtype=torch.long, device=features.device)
    logits = _decoder(features, target_tokens=tgt,
                      bos_token_id=_tokenizer.CONT_TOKEN_ID)
    step_idx = tgt.size(1) - 1
    step_logits = logits[0, step_idx, :]
    return torch.log_softmax(step_logits, dim=-1)


def is_valid_edge(parent_name: Optional[str], child_name: str) -> bool:
    if _parent_to_children is None:
        return True
    if parent_name is None:
        return True
    children = _parent_to_children.get(parent_name)
    if children is None:
        return True
    return child_name in children


# -------------- Minimal-change repair (single best) --------------
_NAME_TO_SLOT_TOK: Optional[Dict[Tuple[int, str], Tuple[int, int]]] = None


def _build_name_to_slot_tok() -> Dict[Tuple[int, str], Tuple[int, int]]:
    global _NAME_TO_SLOT_TOK
    if _NAME_TO_SLOT_TOK is not None:
        return _NAME_TO_SLOT_TOK
    mapping: Dict[Tuple[int, str], Tuple[int, int]] = {}
    for (slot_idx, tok_id), nm in _tokenizer.slot_token_map.items():
        lvl = _tokenizer.level_positions[slot_idx]
        mapping[(lvl, nm)] = (slot_idx, tok_id)
    _NAME_TO_SLOT_TOK = mapping
    return mapping


@torch.no_grad()
def _simulate_level_candidates_under_prefix(features: torch.Tensor, prefix_ids: List[int], level_bins: List[int], parent_name: Optional[str]) -> Tuple[List[dict], float]:
    candidates: List[dict] = []
    cont_prefix = 0.0
    sim_ids = list(prefix_ids)
    step_logp0 = _next_step_logprobs(features, sim_ids)
    absent_lp = step_logp0[SPECIAL_IDS['ABSENT']].item()

    for j, slot_idx in enumerate(level_bins):
        if j > 0:
            step_logp_prev = _next_step_logprobs(features, sim_ids)
            cont_lp_prev = step_logp_prev[SPECIAL_IDS['CONT']].item()
            cont_prefix += cont_lp_prev
            sim_ids.append(SPECIAL_IDS['CONT'])
        step_logp = _next_step_logprobs(features, sim_ids)
        allowed_idx = _allowed_idx_tensor(slot_idx, step_logp.device)
        if allowed_idx.numel() == 0:
            continue
        mask_keep = (allowed_idx != SPECIAL_IDS['PAD']) & (
            allowed_idx != SPECIAL_IDS['CONT']) & (allowed_idx != SPECIAL_IDS['ABSENT'])
        allowed_idx2 = allowed_idx[mask_keep]
        if allowed_idx2.numel() == 0:
            continue
        vals = step_logp.index_select(0, allowed_idx2)
        for lp, tid in zip(vals.tolist(), allowed_idx2.tolist()):
            nm = _tokenizer.slot_token_map.get((slot_idx, tid))
            if nm is None:
                continue
            if not is_valid_edge(parent_name, nm):
                continue
            candidates.append({
                'name': nm,
                'tok_id': tid,
                'score': cont_prefix + lp,
                'slot_idx': slot_idx,
                'bin_index': j,
            })
    return candidates, absent_lp


@torch.no_grad()
def predict_taxonomy_for_sequence_validated(sequence: str) -> List[str]:
    """
    Build on top of predict_taxonomy_for_sequence and make minimal corrections to ensure validity.

    This function:
    1. Gets the excellent autoregressive result from predict_taxonomy_for_sequence
    2. Validates each level and replaces only invalid nodes with valid alternatives
    3. Preserves the original length and overall structure
    4. Stops at first unrecoverable invalid node to maintain quality

    Returns a list of taxonomy names (same format as predict_taxonomy_for_sequence)
    """
    # Get the high-quality baseline result
    original_taxonomy = predict_taxonomy_for_sequence(sequence)

    if not original_taxonomy:
        return []

    # If no validation mapping available, return original
    if _parent_to_children is None:
        return original_taxonomy

    # Validate and repair each level
    validated_taxonomy = []

    for i, current_name in enumerate(original_taxonomy):
        parent_name = validated_taxonomy[-1] if validated_taxonomy else None

        # Check if current node is valid given its parent
        if is_valid_edge(parent_name, current_name):
            # Keep the original - it's valid
            validated_taxonomy.append(current_name)
        else:
            # Find a valid replacement
            replacement = _find_best_replacement(
                parent_name, current_name, original_taxonomy[i:])
            if replacement:
                validated_taxonomy.append(replacement)
            else:
                # If no valid replacement found, stop here to preserve quality
                logger.info(f"No valid replacement found for '{current_name}' under parent '{parent_name}'. Stopping at level {i+1} to preserve quality.")
                break

    return validated_taxonomy


@torch.no_grad()
def predict_taxonomy_for_sequence_to_organism(sequence: str) -> List[str]:
    """
    Build on top of predict_taxonomy_for_sequence_validated and extend to reach an actual organism.

    This function:
    1. Gets the validated taxonomy (guaranteed to be valid)
    2. Extends it further down the tree to reach a leaf node (organism with no children)
    3. Makes intelligent choices to find the most likely organism while preserving the path

    Returns a list of taxonomy names ending with an actual organism
    """
    # Start with the validated taxonomy
    validated_taxonomy = predict_taxonomy_for_sequence_validated(sequence)

    if not validated_taxonomy:
        return []

    # If no validation mapping available, return what we have
    if _parent_to_children is None:
        return validated_taxonomy

    # Extend to reach an organism (leaf node)
    extended_taxonomy = list(validated_taxonomy)
    current_parent = extended_taxonomy[-1]

    # Keep extending until we reach a leaf node (organism with no children)
    max_extensions = 20  # Prevent infinite loops
    extensions = 0

    while extensions < max_extensions:
        # Check if current node has children
        children = _parent_to_children.get(current_parent, [])

        if not children:
            # We've reached a leaf node (organism)!
            break

        # Find the best child to continue with
        best_child = _find_best_organism_path(current_parent, children)

        if best_child:
            extended_taxonomy.append(best_child)
            current_parent = best_child
            extensions += 1
        else:
            # No good child found, stop here
            break

    logger.info(f"Extended taxonomy from {len(validated_taxonomy)} to {len(extended_taxonomy)} levels, reaching organism: {extended_taxonomy[-1] if extended_taxonomy else 'None'}")

    return extended_taxonomy


@torch.no_grad()
def predict_top_k_organisms(sequence: str, k: int = 5) -> List[dict]:
    """
    Build on top of predict_taxonomy_for_sequence_validated and find top K valid trees ending with organisms.

    This function:
    1. Gets the validated taxonomy (guaranteed to be valid)  
    2. From the last valid node, explores multiple paths to find different organisms
    3. Returns K different valid trees, each ending with a specific organism

    Returns a list of dicts with 'names' and 'score' keys
    """
    # Start with the validated taxonomy
    base_taxonomy = predict_taxonomy_for_sequence_validated(sequence)

    if not base_taxonomy:
        return []

    # If no validation mapping available, return single result
    if _parent_to_children is None:
        return [{'names': base_taxonomy, 'score': 0.0}]

    # Find all possible organism endings from the base taxonomy
    organism_paths = []

    # Start from the last node in the validated taxonomy
    current_parent = base_taxonomy[-1]
    base_path = list(base_taxonomy)

    # Use BFS to explore all paths to organisms
    # (path, current_node, depth)
    paths_to_explore = [(base_path, current_parent, 0)]
    max_depth = 15  # Prevent going too deep

    # Get extra candidates
    while paths_to_explore and len(organism_paths) < k * 3:
        current_path, current_node, depth = paths_to_explore.pop(0)

        if depth > max_depth:
            continue

        children = _parent_to_children.get(current_node, [])

        if not children:
            # This is a leaf node (organism)
            score = _score_organism_path(current_path)
            organism_paths.append({
                'names': current_path,
                'score': score,
                'organism': current_path[-1]
            })
            continue

        # Score and sort children to explore best ones first
        scored_children = []
        for child in children:
            child_score = _score_organism_preference(child)
            # Bonus for being closer to organisms
            grandchildren = _parent_to_children.get(child, [])
            if not grandchildren:
                child_score += 20  # Big bonus for direct organisms
            elif len(grandchildren) < 5:
                child_score += 10  # Bonus for small clades

            scored_children.append((child_score, child))

        # Sort by score (best first) and take top candidates
        scored_children.sort(reverse=True)
        # Explore top 3 children per node
        top_children = scored_children[:min(3, len(scored_children))]

        for child_score, child in top_children:
            new_path = current_path + [child]
            paths_to_explore.append((new_path, child, depth + 1))

    # Sort organism paths by score and return top K unique ones
    organism_paths.sort(key=lambda x: x['score'], reverse=True)

    # Remove duplicates and ensure diversity
    unique_paths = []
    seen_organisms = set()

    for path_info in organism_paths:
        organism = path_info['organism']
        if organism not in seen_organisms:
            unique_paths.append({
                'names': path_info['names'],
                'score': path_info['score']
            })
            seen_organisms.add(organism)

            if len(unique_paths) >= k:
                break

    # If we don't have enough unique organisms, fill with best available
    while len(unique_paths) < k and len(unique_paths) < len(organism_paths):
        for path_info in organism_paths:
            if len(unique_paths) >= k:
                break
            # Add paths we haven't added yet
            path_dict = {
                'names': path_info['names'], 'score': path_info['score']}
            if path_dict not in unique_paths:
                unique_paths.append(path_dict)

    logger.info(f"Found {len(unique_paths)} unique organism paths from base taxonomy of length {len(base_taxonomy)}")

    return unique_paths


def _score_organism_path(taxonomy_path: List[str]) -> float:
    """
    Score a complete taxonomy path ending with an organism.
    Higher scores = better paths.
    """
    if not taxonomy_path:
        return 0

    score = 0

    # Bonus for longer paths (more specific)
    score += len(taxonomy_path) * 2

    # Bonus for the organism itself
    organism = taxonomy_path[-1]
    organism_score = _score_organism_preference(organism)
    score += organism_score

    # Bonus for well-formed taxonomic levels
    # Look for expected patterns
    for i, name in enumerate(taxonomy_path):
        name_lower = name.lower()

        # Bonus for standard taxonomic indicators
        if any(indicator in name_lower for indicator in ['cellular organisms', 'eukaryota', 'bacteria', 'archaea']):
            score += 5
        if any(indicator in name_lower for indicator in ['mammalia', 'vertebrata', 'chordata']):
            score += 3
        if any(indicator in name_lower for indicator in ['primates', 'homo']):
            score += 8

    # Penalty for obvious issues
    problematic = ['unclassified', 'uncultured', 'environmental']
    for name in taxonomy_path:
        for prob in problematic:
            if prob in name.lower():
                score -= 5

    return score


def _find_best_organism_path(parent_name: str, children: List[str]) -> Optional[str]:
    """
    Find the best child to continue the path toward an organism.

    Strategy:
    1. Prefer children that lead to organisms (have no children themselves)
    2. Prefer children with common/generic names that are likely to be well-represented
    3. Avoid overly specific names that might be rare
    """
    if not children:
        return None

    # Separate children into leaf nodes (organisms) and internal nodes
    leaf_children = []
    internal_children = []

    for child in children:
        grandchildren = _parent_to_children.get(child, [])
        if not grandchildren:
            leaf_children.append(child)
        else:
            internal_children.append(child)

    # If we have direct organism children, prefer them
    if leaf_children:
        # Score organisms by how "generic" they seem (prefer common organisms)
        scored_leaves = []
        for leaf in leaf_children:
            score = _score_organism_preference(leaf)
            scored_leaves.append((score, leaf))

        # Return the highest scoring organism
        scored_leaves.sort(reverse=True)
        return scored_leaves[0][1]

    # Otherwise, continue with internal nodes, preferring more generic ones
    if internal_children:
        scored_internals = []
        for internal in internal_children:
            score = _score_organism_preference(internal)
            # Bonus for having fewer children (closer to organisms)
            grandchildren_count = len(_parent_to_children.get(internal, []))
            if grandchildren_count < 10:  # Prefer smaller clades
                score += 5
            scored_internals.append((score, internal))

        scored_internals.sort(reverse=True)
        return scored_internals[0][1]

    return None


def get_human_reference_taxonomy() -> List[str]:
    """
    Extract the human taxonomic path from the parent-child mapping data.
    Traces backwards from 'Homo' to the root to build the complete path.

    Returns:
        List of taxonomic names from most general to most specific (ending at Homo)
    """
    if _parent_to_children is None:
        # Fallback hardcoded human taxonomy if no mapping available
        return [
            "cellular organisms", "Eukaryota", "Opisthokonta", "Metazoa", "Eumetazoa",
            "Bilateria", "Deuterostomia", "Chordata", "Craniata", "Vertebrata",
            "Gnathostomata", "Teleostomi", "Euteleostomi", "Sarcopterygii",
            "Dipnotetrapodomorpha", "Tetrapoda", "Amniota", "Mammalia", "Theria",
            "Eutheria", "Boreoeutheria", "Euarchontoglires", "Primates",
            "Haplorrhini", "Simiiformes", "Catarrhini", "Hominoidea",
            "Hominidae", "Homininae", "Homo"
        ]

    # Build child-to-parent mapping
    child_to_parent = {}
    for parent, children in _parent_to_children.items():
        for child in children:
            child_to_parent[child] = parent

    # Trace backwards from Homo to root
    path = []
    current = "Homo"

    # First try to find Homo in the mapping
    if current not in child_to_parent:
        # Try variations
        homo_variations = ["Homo", "Homo sapiens", "homo", "Homo (genus)"]
        for variation in homo_variations:
            if variation in child_to_parent:
                current = variation
                break
        else:
            logger.warning(
                "Could not find Homo in parent-child mapping, using fallback")
            return get_human_reference_taxonomy()  # Use fallback

    # Build path by tracing backwards
    visited = set()
    while current and current not in visited:
        visited.add(current)
        path.append(current)
        current = child_to_parent.get(current)

        # Prevent infinite loops
        if len(path) > 50:
            break

    # Reverse to get root-to-Homo order
    path.reverse()

    if not path or path[-1] != "Homo":
        logger.warning(
            "Failed to build complete human taxonomy path, using fallback")
        return get_human_reference_taxonomy()  # Use fallback

    logger.info(f"Built human reference taxonomy with {len(path)} levels")
    return path


def calculate_human_distance_score(predicted_taxonomy: List[str]) -> float:
    """
    Calculate taxonomic distance from humans (0=very distant, 1=very close).

    This score considers both the depth of common ancestry AND whether the prediction
    follows the correct evolutionary path toward humans.

    Args:
        predicted_taxonomy: List of taxonomic names from general to specific

    Returns:
        float: distance score 0.0-1.0 where:
        - 1.0 = very close to humans (e.g., other primates)
        - 0.5-0.8 = other mammals  
        - 0.3-0.5 = other vertebrates
        - 0.1-0.3 = other deuterostomes
        - 0.0-0.1 = protostomes, bacteria, etc.
    """
    human_ref = get_human_reference_taxonomy()

    if not predicted_taxonomy or not human_ref:
        return 0.0

    # Find the deepest (most recent) common ancestor
    matching_levels = 0
    min_length = min(len(human_ref), len(predicted_taxonomy))

    for i in range(min_length):
        if human_ref[i].lower().strip() == predicted_taxonomy[i].lower().strip():
            matching_levels += 1
        else:
            break

    if matching_levels == 0:
        return 0.0  # No common ancestor found = maximum distance

    # Base score from common ancestry depth
    normalized_depth = matching_levels / len(human_ref)
    base_score = normalized_depth ** 0.3  # Gentler curve

    # Apply biological relationship bonuses/penalties
    predicted_lower = [name.lower() for name in predicted_taxonomy]
    predicted_text = " ".join(predicted_lower)

    # Major evolutionary groups - apply strong bonuses for being on the right path
    if any(term in predicted_text for term in ["primates", "hominidae", "homo"]):
        base_score *= 3.0  # Huge bonus for primates
    elif any(term in predicted_text for term in ["mammalia", "theria", "eutheria"]):
        base_score *= 2.0  # Large bonus for mammals
    elif any(term in predicted_text for term in ["chordata", "vertebrata", "craniata"]):
        base_score *= 1.5  # Moderate bonus for vertebrates
    elif any(term in predicted_text for term in ["deuterostomia"]):
        base_score *= 1.2  # Small bonus for deuterostomes
    elif any(term in predicted_text for term in ["protostomia", "arthropoda", "insecta", "diptera"]):
        base_score *= 0.3  # Heavy penalty for protostomes (wrong branch!)
    elif any(term in predicted_text for term in ["bacteria", "archaea"]):
        base_score *= 0.1  # Very heavy penalty for prokaryotes

    # Additional bonus for very specific human-related predictions
    if any(term in predicted_text for term in ["euarchontoglires", "boreoeutheria"]):
        base_score *= 1.8  # Bonus for being on the placental mammal path

    return min(1.0, base_score)


def _score_organism_preference(name: str) -> float:
    """
    Score how preferable this organism/taxon is for extending the tree.
    Higher scores = more preferable.
    """
    if not name:
        return 0

    name_lower = name.lower()
    score = 0

    # Prefer names that don't have specific strain/isolate indicators
    penalties = [
        'sp.', 'strain', 'isolate', 'clone', 'uncultured', 'unclassified',
        'environmental', 'enrichment', 'culture', 'BOLD:', 'cf.', 'aff.'
    ]

    for penalty in penalties:
        if penalty in name_lower:
            score -= 10

    # Prefer shorter names (often more general)
    if len(name) < 30:
        score += 2
    elif len(name) > 60:
        score -= 5

    # Prefer names without numbers/codes
    if any(char.isdigit() for char in name):
        score -= 3

    # Prefer names that look like proper binomial nomenclature
    words = name.split()
    if len(words) == 2 and not any(word.lower() in ['sp', 'sp.', 'strain'] for word in words):
        score += 5  # Likely a proper species name

    # Prefer common model organisms or well-known taxa
    well_known = [
        'homo', 'sapiens', 'human', 'mouse', 'mus', 'musculus', 'drosophila',
        'melanogaster', 'caenorhabditis', 'elegans', 'saccharomyces', 'cerevisiae',
        'escherichia', 'coli', 'arabidopsis', 'thaliana'
    ]

    for known in well_known:
        if known in name_lower:
            score += 15

    return score


def _find_best_replacement(parent_name: Optional[str], invalid_name: str, remaining_taxonomy: List[str]) -> Optional[str]:
    """
    Find the best replacement for an invalid node.

    Strategy:
    1. Look for valid children of the parent
    2. Prefer names that allow the next level to remain valid (if any)
    3. Prefer names that are similar to the invalid name (fuzzy matching)
    4. Be conservative - return None if no good replacement exists
    """
    if _parent_to_children is None or parent_name is None:
        return None

    valid_children = _parent_to_children.get(parent_name, [])
    if not valid_children:
        return None

    # If there's a next level, prefer replacements that allow it to be valid
    next_name = remaining_taxonomy[1] if len(remaining_taxonomy) > 1 else None

    best_candidate = None
    best_score = -1
    found_chain_preserving = False

    for candidate in valid_children:
        score = 0

        # Strong bonus if this candidate allows the next level to be valid
        if next_name and candidate in _parent_to_children:
            if next_name in _parent_to_children[candidate]:
                score += 100  # Very strong preference for preserving downstream nodes
                found_chain_preserving = True

        # Bonus for name similarity (simple heuristic)
        if invalid_name and candidate:
            # Simple similarity based on common words
            invalid_words = set(invalid_name.lower().split())
            candidate_words = set(candidate.lower().split())
            common_words = invalid_words.intersection(candidate_words)
            score += len(common_words) * 5

        if score > best_score:
            best_score = score
            best_candidate = candidate

    # If we found a candidate that preserves the downstream chain, use it
    if found_chain_preserving:
        return best_candidate

    # If there's a next level but no candidate can preserve it, return None
    # This will cause the function to stop here rather than make a bad replacement
    if next_name:
        logger.info(f"Could not find replacement for '{invalid_name}' that preserves downstream node '{next_name}'")
        return None

    # If this is the last level or we have a reasonable similarity match, return the best candidate
    if best_score > 0:
        return best_candidate

    # Otherwise, be conservative and return None
    return None


@torch.no_grad()
def predict_valid_tree_min_edit(sequence: str) -> dict:
    seq = _truncate_sequence(sequence, MAX_SEQ_LEN_ESM)
    toks = _esm_tokenize_single(seq, _alphabet)
    features = _feature_extractor(toks)

    original_names = predict_taxonomy_for_sequence(seq)

    bins_per_level = _build_bins_per_level(_tokenizer.level_positions)
    levels = sorted(bins_per_level.keys())
    name_to_slot_tok = _build_name_to_slot_tok()

    repaired_names: List[str] = []
    edits: List[Tuple[int, Optional[str], Optional[str]]] = []
    total_logprob = 0.0
    prefix_ids: List[int] = []

    target_depth = len(original_names)

    for i, lvl in enumerate(levels, start=1):
        if i > target_depth:
            break
        level_bins = bins_per_level[lvl]
        parent_name = repaired_names[-1] if repaired_names else None
        orig_name = original_names[i - 1] if i - \
            1 < len(original_names) else None

        candidates, _absent_lp = _simulate_level_candidates_under_prefix(
            features, prefix_ids, level_bins, parent_name)

        chosen_name = None
        chosen_tid = None
        chosen_slot = None

        if orig_name is not None and is_valid_edge(parent_name, orig_name):
            st = name_to_slot_tok.get((lvl, orig_name))
            if st is not None:
                s_slot, s_tid = st
                if s_slot in level_bins:
                    chosen_name, chosen_tid, chosen_slot = orig_name, s_tid, s_slot

        if chosen_name is None:
            chosen = None
            if candidates:
                candidates.sort(key=lambda x: x['score'], reverse=True)
                chosen = candidates[0]
            else:
                sim_ids = list(prefix_ids)
                cont_prefix_fb = 0.0
                best_fb = None
                child_names = None if (
                    _parent_to_children is None or parent_name is None) else _parent_to_children.get(parent_name)
                for j2, slot_idx2 in enumerate(level_bins):
                    if j2 > 0:
                        step_logp_prev = _next_step_logprobs(features, sim_ids)
                        cont_prefix_fb += step_logp_prev[SPECIAL_IDS['CONT']].item()
                        sim_ids.append(SPECIAL_IDS['CONT'])
                    step_logp2 = _next_step_logprobs(features, sim_ids)
                    if child_names:
                        for cn in child_names:
                            st = name_to_slot_tok.get((lvl, cn))
                            if st is None:
                                continue
                            s_slot, s_tid = st
                            if s_slot != slot_idx2:
                                continue
                            lp = step_logp2[s_tid].item()
                            cand2 = {'name': cn, 'tok_id': s_tid, 'score': cont_prefix_fb +
                                     lp, 'slot_idx': slot_idx2, 'bin_index': j2}
                            if best_fb is None or cand2['score'] > best_fb['score']:
                                best_fb = cand2
                    else:
                        allowed_idx2_full = _allowed_idx_tensor(
                            slot_idx2, step_logp2.device)
                        if allowed_idx2_full.numel() == 0:
                            continue
                        mask_keep2 = (allowed_idx2_full != SPECIAL_IDS['PAD']) & (
                            allowed_idx2_full != SPECIAL_IDS['CONT']) & (allowed_idx2_full != SPECIAL_IDS['ABSENT'])
                        allowed_idx2b = allowed_idx2_full[mask_keep2]
                        if allowed_idx2b.numel() == 0:
                            continue
                        vals2 = step_logp2.index_select(0, allowed_idx2b)
                        max_val2, max_pos2 = torch.max(vals2, dim=0)
                        tid2 = allowed_idx2b[max_pos2].item()
                        nm2 = _tokenizer.slot_token_map.get((slot_idx2, tid2))
                        if nm2 is None:
                            continue
                        cand2 = {'name': nm2, 'tok_id': tid2, 'score': cont_prefix_fb +
                                 max_val2.item(), 'slot_idx': slot_idx2, 'bin_index': j2}
                        if best_fb is None or cand2['score'] > best_fb['score']:
                            best_fb = cand2
                if best_fb is not None:
                    chosen = best_fb
            if chosen is None:
                for _ in range(len(level_bins)):
                    step_logp_f = _next_step_logprobs(features, prefix_ids)
                    total_logprob += step_logp_f[SPECIAL_IDS['CONT']].item()
                    prefix_ids.append(SPECIAL_IDS['CONT'])
                repaired_names.append(orig_name or '<UNK>')
                if orig_name != repaired_names[-1]:
                    edits.append((i, orig_name, repaired_names[-1]))
                continue
            chosen_name = chosen['name']
            chosen_tid = chosen['tok_id']
            chosen_slot = chosen['slot_idx']

        for j, slot_idx in enumerate(level_bins):
            step_logp = _next_step_logprobs(features, prefix_ids)
            if slot_idx != chosen_slot:
                total_logprob += step_logp[SPECIAL_IDS['CONT']].item()
                prefix_ids.append(SPECIAL_IDS['CONT'])
                continue
            tok_lp = step_logp[chosen_tid].item()
            total_logprob += tok_lp
            prefix_ids.append(chosen_tid)
            for k in range(j + 1, len(level_bins)):
                step_logp_tail = _next_step_logprobs(features, prefix_ids)
                total_logprob += step_logp_tail[SPECIAL_IDS['CONT']].item()
                prefix_ids.append(SPECIAL_IDS['CONT'])
            break

        repaired_names.append(chosen_name)
        if orig_name != chosen_name:
            edits.append((i, orig_name, chosen_name))

    return {
        'original_names': original_names,
        'names': repaired_names,
        'edits': edits,
        'num_edits': len(edits),
        'logprob': total_logprob,
        'stopped_level': len(repaired_names),
    }


# -------------- AR generate-first + DP validate-later top-k --------------
@torch.no_grad()
def ar_collect_level_candidates(features: torch.Tensor, topn_bin: int = 5) -> Tuple[List[int], List[List[dict]], List[float]]:
    bins_per_level = _build_bins_per_level(_tokenizer.level_positions)
    levels = sorted(bins_per_level.keys())
    ids_so_far: List[int] = []
    level_candidates: List[List[dict]] = []
    absent_scores: List[float] = []
    # 118, should not exceed positional embedding size - 1
    max_seq_len = TAXONOMY_SEQ_LENGTH

    for lvl in levels:
        bins = bins_per_level[lvl]
        cont_prefix = 0.0
        step_logp0 = _next_step_logprobs(features, ids_so_far)
        absent_lp = step_logp0[SPECIAL_IDS['ABSENT']].item()
        absent_scores.append(absent_lp)
        lvl_cands: List[dict] = []
        for j, slot_idx in enumerate(bins):
            if j > 0:
                step_logp_prev = _next_step_logprobs(features, ids_so_far)
                cont_prefix += step_logp_prev[SPECIAL_IDS['CONT']].item()
                if len(ids_so_far) < max_seq_len:
                    ids_so_far.append(SPECIAL_IDS['CONT'])
            step_logp = _next_step_logprobs(features, ids_so_far)
            allowed_idx = _allowed_idx_tensor(slot_idx, step_logp.device)
            if allowed_idx.numel() > 0:
                mask_keep = (allowed_idx != SPECIAL_IDS['PAD']) & (
                    allowed_idx != SPECIAL_IDS['CONT']) & (allowed_idx != SPECIAL_IDS['ABSENT'])
                allowed_idx2 = allowed_idx[mask_keep]
            else:
                allowed_idx2 = allowed_idx
            if allowed_idx2.numel() > 0:
                vals = step_logp.index_select(0, allowed_idx2)
                k = min(topn_bin, vals.numel())
                top_vals, top_pos = torch.topk(vals, k=k)
                top_tok_ids = allowed_idx2.index_select(0, top_pos)
                for lp, tid in zip(top_vals.tolist(), top_tok_ids.tolist()):
                    nm = _tokenizer.slot_token_map.get((slot_idx, tid))
                    if nm is None:
                        continue
                    lvl_cands.append({
                        'name': nm,
                        'tok_id': tid,
                        'logprob': cont_prefix + lp,
                        'slot_idx': slot_idx,
                        'bin_index': j,
                    })
            cont_here = step_logp[SPECIAL_IDS['CONT']].item()
            best_real_lp = None
            best_real_id = None
            if allowed_idx2.numel() > 0:
                max_val, max_pos = torch.max(
                    step_logp.index_select(0, allowed_idx2), dim=0)
                best_real_lp = max_val.item()
                best_real_id = allowed_idx2[max_pos].item()
            if best_real_lp is not None and best_real_lp > cont_here:
                if len(ids_so_far) < max_seq_len:
                    ids_so_far.append(best_real_id)
                for k2 in range(j + 1, len(bins)):
                    if len(ids_so_far) >= max_seq_len:
                        break
                    step_logp_tail = _next_step_logprobs(features, ids_so_far)
                    ids_so_far.append(SPECIAL_IDS['CONT'])
                break
            else:
                if len(ids_so_far) < max_seq_len:
                    ids_so_far.append(SPECIAL_IDS['CONT'])
        level_candidates.append(lvl_cands)
    return levels, level_candidates, absent_scores


def _dp_topk_valid_paths(levels: List[int], level_candidates: List[List[dict]], absent_scores: List[float], k: int = 5) -> List[dict]:
    Beam = Tuple[List[str], float, Optional[str], bool]
    beams: List[Beam] = [([], 0.0, None, False)]
    for i, lvl in enumerate(levels):
        next_beams: List[Beam] = []
        for names, score, last_name, ended in beams:
            if ended:
                next_beams.append((names, score, last_name, True))
                continue
            next_beams.append(
                (names, score + absent_scores[i], last_name, True))
            for cand in level_candidates[i]:
                nm = cand['name']
                if not is_valid_edge(last_name, nm):
                    continue
                next_beams.append(
                    (names + [nm], score + cand['logprob'], nm, False))
        next_beams.sort(key=lambda x: x[1], reverse=True)
        beams = next_beams[: max(k * 3, k)]
    results = []
    seen = set()
    for names, score, last_name, ended in sorted(beams, key=lambda x: x[1], reverse=True):
        key = tuple(names)
        if key in seen:
            continue
        seen.add(key)
        results.append({'names': names, 'logprob': score})
        if len(results) >= k:
            break
    return results


@torch.no_grad()
def predict_top_k_valid_trees_ar_validated(sequence: str, topn_bin: int = 5, k_return: int = 5) -> List[dict]:
    seq = _truncate_sequence(sequence, MAX_SEQ_LEN_ESM)
    toks = _esm_tokenize_single(seq, _alphabet)
    feats = _feature_extractor(toks)
    levels, lvl_cands, absent_scores = ar_collect_level_candidates(
        feats, topn_bin=topn_bin)
    results = _dp_topk_valid_paths(
        levels, lvl_cands, absent_scores, k=k_return)
    return results


# -------------- Beam search with minimal-change repair --------------
class BeamState(tuple):
    __slots__ = ()

    def __new__(cls, names: List[str], prefix_ids: List[int], logprob: float, edits: int):
        return tuple.__new__(cls, (names, prefix_ids, logprob, edits))

    @property
    def names(self):
        return self[0]

    @property
    def prefix_ids(self):
        return self[1]

    @property
    def logprob(self):
        return self[2]

    @property
    def edits(self):
        return self[3]


@torch.no_grad()
def _score_name_as_candidate(features: torch.Tensor, prefix_ids: List[int], level_bins: List[int], lvl: int, name: str) -> Optional[dict]:
    st = _build_name_to_slot_tok().get((lvl, name))
    if st is None:
        return None
    slot_idx, tok_id = st
    if slot_idx not in level_bins:
        return None
    sim_ids = list(prefix_ids)
    cont_prefix = 0.0
    for j, s in enumerate(level_bins):
        step_logp = _next_step_logprobs(features, sim_ids)
        if s == slot_idx:
            tok_lp = step_logp[tok_id].item()
            return {'name': name, 'tok_id': tok_id, 'slot_idx': slot_idx, 'bin_index': j, 'score': cont_prefix + tok_lp}
        cont_lp = step_logp[SPECIAL_IDS['CONT']].item()
        cont_prefix += cont_lp
        sim_ids.append(SPECIAL_IDS['CONT'])
    return None


@torch.no_grad()
def predict_top_k_valid_trees_min_edit(sequence: str, topn_bin: int = 3, beam_size: int = 8, k_return: int = 5, edit_penalty: float = 0.0) -> List[dict]:
    seq = _truncate_sequence(sequence, MAX_SEQ_LEN_ESM)
    toks = _esm_tokenize_single(seq, _alphabet)
    feats = _feature_extractor(toks)

    baseline = predict_taxonomy_for_sequence(seq)
    target_depth = len(baseline)

    bins_per_level = _build_bins_per_level(_tokenizer.level_positions)
    levels = sorted(bins_per_level.keys())

    beams: List[BeamState] = [
        BeamState(names=[], prefix_ids=[], logprob=0.0, edits=0)]

    for i, lvl in enumerate(levels, start=1):
        if i > target_depth:
            break
        level_bins = bins_per_level[lvl]
        next_beams: List[BeamState] = []
        for names, prefix_ids, logp, edits in beams:
            parent_name = names[-1] if names else None
            cand_valid, _abs = _simulate_level_candidates_under_prefix(
                feats, prefix_ids, level_bins, parent_name)
            orig_name = baseline[i - 1] if i - 1 < len(baseline) else None
            considered: List[dict] = []
            if orig_name is not None and is_valid_edge(parent_name, orig_name):
                c = _score_name_as_candidate(
                    feats, prefix_ids, level_bins, lvl, orig_name)
                if c is not None:
                    considered.append({'candidate': c, 'edited': False})
            pool = [
                c for c in cand_valid if orig_name is None or c['name'] != orig_name]
            pool.sort(key=lambda x: x['score'], reverse=True)
            for c in pool[:max(0, topn_bin - len(considered))]:
                considered.append({'candidate': c, 'edited': True})
            if not considered:
                child_names = None if (
                    _parent_to_children is None or parent_name is None) else _parent_to_children.get(parent_name)
                if child_names:
                    scored_kids: List[dict] = []
                    for cn in child_names:
                        c = _score_name_as_candidate(
                            feats, prefix_ids, level_bins, lvl, cn)
                        if c is not None:
                            scored_kids.append(c)
                    scored_kids.sort(key=lambda x: x['score'], reverse=True)
                    for c in scored_kids[:topn_bin]:
                        considered.append(
                            {'candidate': c, 'edited': orig_name != c['name']})
            if not considered and (_parent_to_children is None or parent_name is None or _parent_to_children.get(parent_name) is None):
                scored_any: List[dict] = []
                sim_ids = list(prefix_ids)
                cont_prefix = 0.0
                for j, s in enumerate(level_bins):
                    if j > 0:
                        step_prev = _next_step_logprobs(feats, sim_ids)
                        cont_prefix += step_prev[SPECIAL_IDS['CONT']].item()
                        sim_ids.append(SPECIAL_IDS['CONT'])
                    step = _next_step_logprobs(feats, sim_ids)
                    allowed = _allowed_idx_tensor(s, step.device)
                    if allowed.numel() == 0:
                        continue
                    mask = (allowed != SPECIAL_IDS['PAD']) & (
                        allowed != SPECIAL_IDS['CONT']) & (allowed != SPECIAL_IDS['ABSENT'])
                    allowed2 = allowed[mask]
                    if allowed2.numel() == 0:
                        continue
                    vals = step.index_select(0, allowed2)
                    max_val, max_pos = torch.max(vals, dim=0)
                    tid = allowed2[max_pos].item()
                    nm = _tokenizer.slot_token_map.get((s, tid))
                    if nm is None:
                        continue
                    scored_any.append({'name': nm, 'tok_id': tid, 'slot_idx': s,
                                      'bin_index': j, 'score': cont_prefix + max_val.item()})
                scored_any.sort(key=lambda x: x['score'], reverse=True)
                for c in scored_any[:topn_bin]:
                    considered.append(
                        {'candidate': c, 'edited': orig_name != c['name']})
            if not considered:
                new_prefix = list(prefix_ids)
                add_lp = 0.0
                for _ in range(len(level_bins)):
                    step = _next_step_logprobs(feats, new_prefix)
                    add_lp += step[SPECIAL_IDS['CONT']].item()
                    new_prefix.append(SPECIAL_IDS['CONT'])
                new_names = names + [orig_name or '<UNK>']
                next_beams.append(BeamState(
                    new_names, new_prefix, logp + add_lp - edit_penalty, edits + (0 if orig_name else 1)))
                continue
            considered.sort(
                key=lambda x: x['candidate']['score'], reverse=True)
            for item in considered:
                c = item['candidate']
                edited_flag = item['edited']
                new_prefix = list(prefix_ids)
                add_lp = 0.0
                for j, s in enumerate(level_bins):
                    step = _next_step_logprobs(feats, new_prefix)
                    if s != c['slot_idx']:
                        add_lp += step[SPECIAL_IDS['CONT']].item()
                        new_prefix.append(SPECIAL_IDS['CONT'])
                    else:
                        add_lp += step[c['tok_id']].item()
                        new_prefix.append(c['tok_id'])
                        for k in range(j + 1, len(level_bins)):
                            step_tail = _next_step_logprobs(feats, new_prefix)
                            add_lp += step_tail[SPECIAL_IDS['CONT']].item()
                            new_prefix.append(SPECIAL_IDS['CONT'])
                        break
                new_names = names + [c['name']]
                new_logp = logp + add_lp - \
                    (edit_penalty if edited_flag else 0.0)
                next_beams.append(
                    BeamState(new_names, new_prefix, new_logp, edits + (1 if edited_flag else 0)))
        next_beams.sort(key=lambda x: x.logprob, reverse=True)
        beams = next_beams[:beam_size] if next_beams else beams
        if not beams:
            break

    beams.sort(key=lambda x: x.logprob, reverse=True)
    results = []
    seen = set()
    for b in beams:
        key = tuple(b.names)
        if key in seen:
            continue
        seen.add(key)
        results.append(
            {'names': b.names, 'logprob': b.logprob, 'edits': b.edits})
        if len(results) >= k_return:
            break
    return results


# -------------- Confidence: mean per-rank probability of a given lineage --------------
@torch.no_grad()
def score_path_perrank_logprobs(sequence: str, names: List[str]) -> List[float]:
    """
    Teacher-force `names` through the decoder and return the list of chosen-token
    log-probabilities, one per scored rank (root -> leaf order). Names that are not
    vocabulary tokens at their level (e.g. ancestor nodes filled in by leaf-reconstruct)
    are skipped, so the list covers only the ranks the model actually scored.
    """
    if not names:
        return []
    seq = _truncate_sequence(sequence, MAX_SEQ_LEN_ESM)
    toks = _esm_tokenize_single(seq, _alphabet)
    feats = _feature_extractor(toks)
    if next(_decoder.parameters()).dtype == torch.bfloat16:
        feats = feats.bfloat16()

    bins_per_level = _build_bins_per_level(_tokenizer.level_positions)
    levels = sorted(bins_per_level.keys())
    name_to_slot_tok = _build_name_to_slot_tok()

    prefix_ids: List[int] = []
    per_rank: List[float] = []
    for i, lvl in enumerate(levels, start=1):
        if i > len(names):
            break
        bins = bins_per_level[lvl]
        st = name_to_slot_tok.get((lvl, names[i - 1]))
        if st is None:
            # name not a vocab token at this level: advance prefix, don't score
            for _ in bins:
                prefix_ids.append(SPECIAL_IDS['CONT'])
            continue
        slot, tid = st
        for j, s in enumerate(bins):
            step_logp = _next_step_logprobs(feats, prefix_ids)
            if s == slot:
                per_rank.append(step_logp[tid].item())
                prefix_ids.append(tid)
                for _k in range(j + 1, len(bins)):
                    prefix_ids.append(SPECIAL_IDS['CONT'])
                break
            prefix_ids.append(SPECIAL_IDS['CONT'])
    return per_rank


# Number of leading (broad) taxonomic ranks the confidence is averaged over.
# The model is informative about the broad placement (domain -> phylum/class); the
# deep species-level ranks are noisy and, when included, INVERT the score's correlation
# with accuracy (only correct deep predictions even reach those hard ranks). Empirically
# the geometric-mean per-rank prob over the first ~7 ranks is positively correlated with
# lineage accuracy (r=+0.34 on a 175-seq labeled set across all four superkingdoms),
# whereas over the full path it is anti-correlated (r=-0.27).
BROAD_RANKS = 7


def confidence_from_logprobs(per_rank_logprobs: List[float], k: int = BROAD_RANKS) -> float:
    """
    Broad-placement confidence in (0, 1]: geometric-mean per-rank probability over the
    first `k` scored (broad) ranks. Higher = the model is surer of the broad taxonomic
    placement. This is only weakly predictive of full-lineage correctness and can be high
    even when wrong on out-of-distribution inputs -- it reflects model confidence, not truth.
    """
    import math
    lps = per_rank_logprobs[:k]
    return math.exp(sum(lps) / len(lps)) if lps else 0.0


def score_path_confidence(sequence: str, names: List[str]) -> float:
    """Broad-placement confidence (see `confidence_from_logprobs`) for `names`."""
    return confidence_from_logprobs(score_path_perrank_logprobs(sequence, names))


def score_path_mean_logprob(sequence: str, names: List[str]) -> float:
    """
    Raw geometric-mean per-rank probability over the WHOLE path (exp of the mean
    chosen-token log-prob), in (0, 1]. Faithful to the model's token probabilities, but
    anti-correlated with accuracy because of the deep-rank effect described on
    `BROAD_RANKS`. Kept for diagnostics; use `score_path_confidence` for the headline score.
    """
    import math
    lps = score_path_perrank_logprobs(sequence, names)
    return math.exp(sum(lps) / len(lps)) if lps else 0.0

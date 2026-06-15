import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import esm
import logging

# LoRA Implementation
class LoRALinear(nn.Module):
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha

        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(self.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # nn.init.zeros_(self.lora_B) # Already initialized to zeros

        self.scaling = self.alpha / self.rank
        self.linear.weight.requires_grad = False # Freeze original weights
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False # Freeze original bias


    def forward(self, x):
        original_output = self.linear(x)
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_output + lora_output

    def extra_repr(self):
        return f'rank={self.rank}, alpha={self.alpha}, scaling={self.scaling}, in_features={self.in_features}, out_features={self.out_features}'

def apply_lora_to_model(model, rank, alpha, target_modules):
    if rank == 0: # If rank is 0, LoRA is disabled
        logging.info("LoRA rank is 0, LoRA is disabled.")
        return model

    logging.info(f"Applying LoRA with rank={rank}, alpha={alpha} to modules: {target_modules}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(target_module in name for target_module in target_modules):
            # Need to find the parent module to replace the child.
            # This assumes modules are not nested deeper than one level for replacement.
            name_parts = name.split('.')
            if len(name_parts) > 1:
                parent_name = '.'.join(name_parts[:-1])
                child_name = name_parts[-1]
                parent_module = model.get_submodule(parent_name)
            else: # Top-level module
                parent_module = model
                child_name = name
            
            if hasattr(parent_module, child_name):
                lora_layer = LoRALinear(getattr(parent_module, child_name), rank, alpha)
                setattr(parent_module, child_name, lora_layer)
                logging.info(f"Applied LoRA to {name}")
            else:
                logging.warning(f"Could not find child {child_name} in parent {parent_name if len(name_parts) > 1 else 'model root'}")
    return model


class ESMFeatureExtractor(nn.Module):
    def __init__(self, esm_model, num_attention_heads=5):
        super().__init__()
        self.esm_model = esm_model
        self.embedding_dim = esm_model.embed_dim # Should be 1280 for esm2_t33_650M_UR50D
        self.num_heads = num_attention_heads
        
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(f"Embedding dimension ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})")
        self.head_dim = self.embedding_dim // self.num_heads

        # Make sure ESM parameters are frozen if LoRA is applied or if we only fine-tune the new parts
        for param in self.esm_model.parameters():
            param.requires_grad = False # Will be overridden by LoRA for specific layers

        # Attention pooling layers
        self.q_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.k_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_proj = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.out_proj = nn.Linear(self.embedding_dim, self.embedding_dim)

        # MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim)
        )

    def forward(self, input_ids):
        # Extract embeddings from the ESM model
        # repr_layers should be a list, e.g., [self.esm_model.num_layers]
        esm_output = self.esm_model(input_ids, repr_layers=[self.esm_model.num_layers], return_contacts=False)
        embeddings = esm_output["representations"][self.esm_model.num_layers]  # (B, L, D)
        
        batch_size, seq_len, _ = embeddings.shape

        # Project to Q, K, V
        q = self.q_proj(embeddings)  # (B, L, D)
        k = self.k_proj(embeddings)  # (B, L, D)
        v = self.v_proj(embeddings)  # (B, L, D)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # (B, H, L, Dh)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # (B, H, L, Dh)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2) # (B, H, L, Dh)
        
        # Attention pooling. NOTE: the trained model used
        # xformers.memory_efficient_attention(q, k, v) with q/k/v shaped
        # (B, num_heads, L, head_dim). xformers interprets its inputs as
        # (B, M, H, K) = (batch, seq, heads, head_dim), so it effectively ran
        # attention over the num_heads(=5) axis with L as parallel heads. We
        # reproduce that exact computation natively (scale = 1/sqrt(head_dim))
        # so inference matches the checkpoint without an xformers dependency.
        scale = 1.0 / math.sqrt(self.head_dim)
        qh = q.permute(0, 2, 1, 3)  # (B, L, num_heads, head_dim) = (B, H=L, M=5, K)
        kh = k.permute(0, 2, 1, 3)
        vh = v.permute(0, 2, 1, 3)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale  # (B, L, 5, 5)
        attn_weights = F.softmax(scores, dim=-1)
        oh = torch.matmul(attn_weights, vh)  # (B, L, 5, head_dim)
        attn_output = oh.permute(0, 2, 1, 3)  # (B, 5, L, head_dim) == xformers output layout

        # Reshape back (matches the original post-xformers reshape)
        attn_output = attn_output.transpose(1,2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        
        # Pass through output projection
        attn_output = self.out_proj(attn_output) # (B, L, D)
        
        # Pool the attention output (mean pooling over sequence length)
        pooled_representation = attn_output.mean(dim=1) # (B, D)
        
        # Pass through MLP
        features = self.mlp(pooled_representation) # (B, D)
        return features


class TaxonomyTransformerDecoder(nn.Module):
    def __init__(
        self, embed_dim, vocab_size=15000, num_layers=4, num_heads=8, dropout=0.1, seq_length=118,
        level_positions=None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Learnable positional embeddings for target sequence
        self.positional_embedding = nn.Embedding(seq_length + 1, embed_dim) # +1 for BOS token

        # Per-rank level embedding: tags each decoder-input slot with its taxonomic
        # level (BOS -> level 0). Required to match the trained checkpoint. When
        # level_positions is None the decoder runs without it (legacy behaviour).
        self.use_level_embedding = level_positions is not None
        if self.use_level_embedding:
            num_levels = max(level_positions)  # levels are 1..num_levels
            self.level_embedding = nn.Embedding(num_levels + 1, embed_dim)
            # level id per decoder-input position: [BOS=0, slot0_level, slot1_level, ...]
            level_ids = torch.tensor([0] + list(level_positions), dtype=torch.long)
            self.register_buffer("level_ids_full", level_ids, persistent=False)

        # Transformer Decoder Layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4, # Feed-Forward Dimension
            dropout=dropout,
            activation='gelu', # Recommended activation
            batch_first=True, # Input format (batch, seq, feature)
            norm_first=True,  # Apply LayerNorm before other sublayers (Pre-LN)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output linear head for classification
        self.output_head = nn.Linear(embed_dim, vocab_size)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize token embedding and output head weights
        nn.init.normal_(self.token_embedding.weight, mean=0, std=self.embed_dim**-0.5)
        nn.init.xavier_uniform_(self.output_head.weight)
        if self.output_head.bias is not None:
            nn.init.zeros_(self.output_head.bias)
        # Positional embeddings are usually initialized to zeros or with some scheme
        nn.init.normal_(self.positional_embedding.weight, mean=0, std=self.embed_dim**-0.5)


    def forward(self, memory, target_tokens=None, bos_token_id=0):
        memory_expanded = memory.unsqueeze(1) 
        batch_size = memory.size(0)

        if target_tokens is not None:
            # Teacher forcing mode - used for both training and validation
            # target_tokens: (batch_size, seq_len), ground truth sequence
            # Prepend BOS token for training. Decoder input is [BOS, t1, ..., t_N-1]
            bos_token_tensor = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=target_tokens.device)
            # `decoder_input_tokens` will have length self.seq_length (BOS + first N-1 tokens of target)
            # The actual target for loss will be target_tokens (t1, ..., tN)
            decoder_input_tokens = torch.cat([bos_token_tensor, target_tokens[:, :-1]], dim=1)
            
            current_input_seq_len = decoder_input_tokens.size(1) # Should be self.seq_length

            if current_input_seq_len > self.positional_embedding.num_embeddings:
                 raise ValueError(f"Training input seq length ({current_input_seq_len}) > positional embedding size ({self.positional_embedding.num_embeddings})")

            positions = torch.arange(0, current_input_seq_len, device=memory.device).unsqueeze(0).repeat(batch_size, 1)
            tgt_emb = self.token_embedding(decoder_input_tokens) + self.positional_embedding(positions)
            if self.use_level_embedding:
                lvl_ids = self.level_ids_full[:current_input_seq_len].unsqueeze(0).repeat(batch_size, 1)
                tgt_emb = tgt_emb + self.level_embedding(lvl_ids)

            causal_mask = nn.Transformer.generate_square_subsequent_mask(current_input_seq_len, device=memory.device)

            decoder_output = self.decoder(
                tgt=tgt_emb, memory=memory_expanded, tgt_mask=causal_mask
            )
            # Logits correspond to predictions for t1, t2, ..., tN based on input BOS, t1, ..., tN-1
            logits = self.output_head(decoder_output) # (batch_size, self.seq_length, vocab_size)
            return logits
        else:
            # Inference / Greedy decoding
            # generated_ids will store self.seq_length generated tokens.
            generated_ids = torch.full((batch_size, self.seq_length), -1, dtype=torch.long, device=memory.device) # Use a pad_token_id if available

            # Start with the BOS token.
            current_input_sequence = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=memory.device)

            for step in range(self.seq_length): # Generate self.seq_length tokens
                current_len_for_embedding = current_input_sequence.size(1)
                
                if current_len_for_embedding > self.positional_embedding.num_embeddings:
                    raise ValueError(f"Inference input seq length ({current_len_for_embedding}) > positional embedding size ({self.positional_embedding.num_embeddings})")

                positions = torch.arange(0, current_len_for_embedding, device=memory.device).unsqueeze(0).repeat(batch_size, 1)
                tgt_emb = self.token_embedding(current_input_sequence) + self.positional_embedding(positions)
                if self.use_level_embedding:
                    lvl_ids = self.level_ids_full[:current_len_for_embedding].unsqueeze(0).repeat(batch_size, 1)
                    tgt_emb = tgt_emb + self.level_embedding(lvl_ids)

                causal_mask = nn.Transformer.generate_square_subsequent_mask(current_len_for_embedding, device=memory.device)

                decoder_output = self.decoder(
                    tgt=tgt_emb, 
                    memory=memory_expanded, 
                    tgt_mask=causal_mask
                )
                
                logits_last_token = self.output_head(decoder_output[:, -1, :]) # Get logits for the last token position
                next_token_id = logits_last_token.argmax(dim=-1) # (batch_size)
                
                generated_ids[:, step] = next_token_id
                
                # Append the predicted token for the next iteration's input
                current_input_sequence = torch.cat((current_input_sequence, next_token_id.unsqueeze(1)), dim=1)
                
                # Optional: EOS token check
                # if hasattr(self, 'eos_token_id') and self.eos_token_id is not None and (next_token_id == self.eos_token_id).all():
                #     return generated_ids[:, :step+1] 
            
            return generated_ids 
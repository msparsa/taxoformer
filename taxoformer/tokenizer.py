import json

class PhyloTokenizer:
    def __init__(
        self,
        level_vocabularies: dict = None,
        max_vocab_size: int = 15000,
        mapping_file: str = None,
        max_levels: int = None
    ):
        """
        level_vocabularies: dict mapping level index (1-based) to list of taxa tokens.
        max_vocab_size: maximum number of taxa tokens per slot (reserving 1 for continuation if needed).
        mapping_file: path to JSON file to load/save mappings.
        max_levels: optional override for number of encoding slots.
        """
        # Special constants
        self.PAD_ID = 0
        self.CONT_TOKEN_ID = 1  # All continuation tokens will use ID 1
        self.ABSENT_ID = 2     # Explicit token for levels beyond the tree depth
        #saved_tokens = 0 to 10
        
        # Load from file if requested and no vocabularies provided
        if mapping_file and level_vocabularies is None:
            with open(mapping_file, 'r') as f:
                data = json.load(f)
            self.level_mapping = data['level_mapping']
            self.level_positions = data['level_positions']
            self.token2id = data['token2id']
        else:
            # Build mapping slots sequentially for each level
            mapping_slots = []
            positions = []

            for lvl in sorted(level_vocabularies.keys()):
                taxa = list(level_vocabularies[lvl])
                # If within size, one slot
                if len(taxa) <= max_vocab_size - 1:  # Reserve space for continuation token
                    mapping_slots.append(taxa)
                    positions.append(lvl)
                else:
                    # Split into chunks with continuation token
                    chunk_size = max_vocab_size - 1
                    cont_token = f"[LVL{lvl}_CONT]"
                    for i in range(0, len(taxa), chunk_size):
                        chunk = taxa[i:i + chunk_size]
                        if i == 0:
                            # First chunk doesn't need continuation token in front
                            mapping_slots.append(chunk)
                        else:
                            mapping_slots.append([cont_token] + chunk)
                        positions.append(lvl)

            self.level_mapping = mapping_slots
            self.level_positions = positions

            # Build token2id mapping 
            self.token2id = {"[PAD]": self.PAD_ID,
                             "[ABSENT]": self.ABSENT_ID}
            
            # First assign all continuation tokens the same ID
            for lvl in sorted(list(set(self.level_positions))):
                cont_token = f"[LVL{lvl}_CONT]"
                # See if this token appears in any mapping slot
                for slot_idx, slot_lvl in enumerate(self.level_positions):
                    if slot_lvl == lvl and any(token == cont_token for token in self.level_mapping[slot_idx]):
                        self.token2id[cont_token] = self.CONT_TOKEN_ID
                        break
            
            # Then assign normal tokens starting from ID 2
            next_id = 12
            for slot_idx in range(len(self.level_mapping)):
                for token in self.level_mapping[slot_idx]:
                    if token not in self.token2id:  # Skip tokens that already have IDs
                        # Make sure we never exceed max_vocab_size
                        token_id = next_id
                        if token_id >= max_vocab_size:
                            token_id = 2 + (token_id - 2) % (max_vocab_size - 2)
                        self.token2id[token] = token_id
                        next_id += 1

            # Save to file if requested
            if mapping_file:
                with open(mapping_file, 'w') as f:
                    json.dump({
                        'level_mapping': self.level_mapping,
                        'level_positions': self.level_positions,
                        'token2id': self.token2id
                    }, f)

        # Store a mapping of token positions to handle collisions
        self.slot_token_map = {}
        for slot_idx, tokens in enumerate(self.level_mapping):
            for token in tokens:
                self.slot_token_map[(slot_idx, self.token2id[token])] = token
            
        # Reverse lookup for special tokens only
        self.id2token = {
            self.PAD_ID: "[PAD]",
            self.ABSENT_ID: "[ABSENT]"
        }
        
        # Add continuation tokens to id2token map
        for token, token_id in self.token2id.items():
            if token_id == self.CONT_TOKEN_ID:
                self.id2token[token_id] = token
                break  # Only need one representation
            
        # Determine number of encoding slots
        self.max_levels = max_levels or len(self.level_mapping)
        self.max_vocab_size = max_vocab_size
        
        # During encoding, we'll store the actual taxon for each position
        self.current_encoding = {}

    def encode(self, phylo_path: list):
        """
        Encode a phylogenetic path (list of taxa tokens) into fixed-length token ID list.
        Returns token_ids, level_positions.
        Raises ValueError if any token is not in vocabulary.
        """
        token_ids = [self.PAD_ID] * self.max_levels
        # Reset the current encoding map
        self.current_encoding = {}
        
        # Assign each taxon to its corresponding slot
        for orig_lvl, taxon in enumerate(phylo_path, start=1):
            if taxon not in self.token2id:
                # print(f"FixedTokenizer: Token '{taxon}' not found in vocabulary in path {phylo_path}")
                # continue
                raise ValueError(f"[FixedTokenizer Error] Token '{taxon}' not found in vocabulary in path {phylo_path}")
            
            # Find the appropriate slot for this level
            slot_found = False
            for idx, lvl in enumerate(self.level_positions):
                if lvl == orig_lvl:
                    if taxon in self.level_mapping[idx]:
                        token_ids[idx] = self.token2id[taxon]
                        # Store the actual taxon for this position
                        self.current_encoding[(idx, self.token2id[taxon])] = taxon
                        slot_found = True
                        break
                    else:
                        # Mark with continuation token
                        cont_token = f"[LVL{orig_lvl}_CONT]"
                        if cont_token not in self.token2id:
                            raise ValueError(f"[FixedTokenizer Error]Continuation token {cont_token} not found in vocabulary")
                        token_ids[idx] = self.CONT_TOKEN_ID
            
            if not slot_found:
                raise ValueError(f"[FixedTokenizer Error]No slot found for token '{taxon}' at level {orig_lvl}")

        # mark all slots for levels beyond the true tree depth as ABSENT
        eos_written = False
        for idx, lvl in enumerate(self.level_positions):
            if lvl > len(phylo_path):
                if not eos_written:
                    token_ids[idx] = self.ABSENT_ID      # first empty level → EOS
                    eos_written = True

        return token_ids, self.level_positions

    def decode(self, token_ids: list):
        """
        Decode token ID sequence back to phylogenetic path.
        Uses slot_token_map to resolve token IDs to their correct taxon names.
        """
        phylo_path = []
        seen_levels = set()
        
        for slot_idx, token_id in enumerate(token_ids):
            # stop decoding once you hit an ABSENT marker
            if token_id == self.ABSENT_ID:
                break

            # still skip pads and continuation tokens
            if token_id in (self.PAD_ID, self.CONT_TOKEN_ID):
                continue
                
            level = self.level_positions[slot_idx] if slot_idx < len(self.level_positions) else None
            if level in seen_levels:
                continue
                
            # Use slot_token_map to get the correct token for this slot and ID
            if (slot_idx, token_id) in self.slot_token_map:
                token = self.slot_token_map[(slot_idx, token_id)]
                phylo_path.append(token)
                seen_levels.add(level)
                
        return phylo_path 
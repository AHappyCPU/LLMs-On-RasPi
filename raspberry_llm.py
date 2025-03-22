import os
import gc
import mmap
import time
import json
import numpy as np
import torch
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# --- Improved Tokenizer Implementation with Enhanced Decode Method ---

class NeoxTokenizer:
    """Improved tokenizer for Pythia/NeoX models with proper detokenization"""
    
    def __init__(self, vocab_path: str):
        self.load_vocabulary(vocab_path)
        self.setup_special_tokens()
        
        # Reduce excessive blacklisting of tokens
        self.repetition_prone_tokens = set([
            "Ġodds", "issued", "ranked", "Ġminutes", "Ġports"
        ])
    
    def load_vocabulary(self, vocab_path: str):
        """Load vocabulary from JSON file"""
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab = json.load(f)
                
            # Create reverse mapping (id -> token)
            self.id_to_token = {int(v): k for k, v in self.vocab.items()}
            print(f"Loaded vocabulary with {len(self.vocab)} tokens")
        except Exception as e:
            print(f"Error loading vocabulary: {str(e)}")
            # Initialize empty vocabulary as fallback
            self.vocab = {}
            self.id_to_token = {}
    
    def setup_special_tokens(self):
        """Set up special tokens used by the model"""
        # Find common special tokens in the vocabulary
        self.unk_token_id = self.vocab.get("<unk>", 0)
        self.bos_token_id = self.vocab.get("<|endoftext|>", 
                                          self.vocab.get("<bos>", 
                                                        self.vocab.get("<s>", 0)))
        self.eos_token_id = self.vocab.get("<|endoftext|>", 
                                          self.vocab.get("<eos>", 
                                                        self.vocab.get("</s>", 0)))
        
        # Identify tokens that start with space (important for GPT-style tokenizers)
        self.space_tokens = {token for token in self.vocab.keys() if token.startswith("Ġ")}
        
        # Find the basic space token if it exists
        self.space_token = " " if " " in self.vocab else "Ġ"
        if self.space_token in self.vocab:
            self.space_token_id = self.vocab[self.space_token]
        else:
            # Try to find the most basic space token
            space_tokens = sorted([t for t in self.vocab.keys() if t.strip() == ""], key=len)
            self.space_token = space_tokens[0] if space_tokens else " "
            self.space_token_id = self.vocab.get(self.space_token, 0)
        
        print(f"Special tokens - UNK: {self.unk_token_id}, BOS: {self.bos_token_id}, EOS: {self.eos_token_id}")
    
    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Encode text to token IDs using a greedy approach"""
        if not text:
            # Return batch with just the BOS token
            return torch.tensor([[self.bos_token_id]], dtype=torch.long)
        
        # Clean and normalize text
        text = text.strip()
        
        # Start with BOS token
        token_ids = [self.bos_token_id]
        
        # Basic whitespace tokenization first to handle word boundaries
        words = text.split()
        
        # Process each word
        for i, word in enumerate(words):
            # Add space before word (except first word)
            if i > 0 and self.space_token:
                if self.space_token in self.vocab:
                    token_ids.append(self.vocab[self.space_token])
                else:
                    # Try to find best space token
                    space_token = "Ġ" + word[0:0]  # Empty string with space prefix
                    if space_token in self.vocab:
                        token_ids.append(self.vocab[space_token])
            
            # Add word - first try as is
            if word in self.vocab:
                token_ids.append(self.vocab[word])
                continue
                
            # Try with space prefix for first word
            if i == 0 and ("Ġ" + word) in self.vocab:
                token_ids.append(self.vocab["Ġ" + word])
                continue
            
            # Character-by-character fallback
            for char in word:
                if char in self.vocab:
                    token_ids.append(self.vocab[char])
                else:
                    token_ids.append(self.unk_token_id)
        
        # Truncate to max_length
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Return as tensor with batch dimension
        return torch.tensor([token_ids], dtype=torch.long)
    
    def decode(self, token_ids: List[int]) -> str:
        """Improved decode method with better handling of subwords and spacing"""
        if not token_ids:
            return ""
        
        text = ""
        prev_token = None
        
        for token_id in token_ids:
            if token_id not in self.id_to_token:
                text += "[UNK]"
                continue
                
            token = self.id_to_token[token_id]
            
            # Skip special tokens
            if token in ["<|endoftext|>", "</s>", "<eos>", "<bos>", "<s>"]:
                continue
            
            # Handle GPT-style tokens (starting with Ġ)
            if token.startswith('Ġ'):
                text += ' ' + token[1:]
            # Handle punctuation (no space before)
            elif token in ['.', ',', '!', '?', ':', ';', ')', ']', '}', "'", '"']:
                text += token
            # Handle opening punctuation (space before, not after)
            elif token in ['(', '[', '{']:
                if text and not text.endswith(' '):
                    text += ' '
                text += token
            # Handle other tokens
            else:
                # Only add space if previous token doesn't end with space
                # and current token isn't a continuation of a subword
                if text and not text.endswith(' ') and not (
                    prev_token and not prev_token.startswith('Ġ') and 
                    not prev_token.endswith(('.', ',', '!', '?', ':', ';', ' '))
                ):
                    text += ' '
                text += token
            
            prev_token = token
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix Unicode issues - convert potential UTF-8 encoding errors
        text = text.encode('utf-8', errors='replace').decode('utf-8')
        
        return text


class WeightLoader:
    """Handles efficient loading and dequantization of model weights"""
    
    def __init__(self, model_dir: str, quantization_bits: int = 8):
        self.model_dir = model_dir
        self.quantization_bits = quantization_bits
        self.memory_maps = {}
        self.scales = {}
        self.zero_points = {}
        
        # Load quantization parameters
        self._load_quantization_params()
    
    def _load_quantization_params(self):
        """Load scales and zero points for dequantization"""
        params_path = os.path.join(self.model_dir, "quant_params.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                params = json.load(f)
                self.scales = params["scales"]
                self.zero_points = params["zero_points"]
        else:
            print("Warning: Quantization parameters file not found, using defaults")
            self.scales = {}
            self.zero_points = {}
    
    def create_memory_map(self, component_name: str) -> mmap.mmap:
        """Create memory map for a model component"""
        if component_name in self.memory_maps:
            return self.memory_maps[component_name]
        
        filepath = os.path.join(self.model_dir, f"{component_name}.bin")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weight file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            # Create memory map in read-only mode
            mem_map = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.memory_maps[component_name] = mem_map
            return mem_map
    
    def load_and_dequantize(self, component_name: str, expected_shape: Tuple[int, ...]) -> torch.Tensor:
        """Load and dequantize weights for a specific component"""
        try:
            # Try to create memory map
            mem_map = self.create_memory_map(component_name)
            
            # Read quantized data from memory map
            quantized_data = np.frombuffer(mem_map, dtype=np.uint8)
            
            # Get dequantization parameters
            scale = self.scales.get(component_name, 0.1)
            zero_point = self.zero_points.get(component_name, 0)
            
            # Calculate final size
            final_size = np.prod(expected_shape)
            
            # Dequantize based on quantization bits
            if self.quantization_bits == 4:
                # For 4-bit, each byte contains 2 values
                dequantized = self._dequantize_4bit(
                    quantized_data, scale, zero_point
                )
                # Calculate final shape size and ensure it matches
                if final_size > dequantized.size:
                    print(f"Warning: Expected shape {expected_shape} requires {final_size} elements, "
                          f"but dequantized data has only {dequantized.size} elements. Padding with zeros.")
                    padded = np.zeros(final_size, dtype=np.float32)
                    padded[:dequantized.size] = dequantized
                    dequantized = padded
                else:
                    dequantized = dequantized[:final_size]
            elif self.quantization_bits == 8:
                # For 8-bit, direct dequantization
                dequantized = (quantized_data.astype(np.float32) - zero_point) * scale
                
                # Make sure the array is the right size
                if dequantized.size != final_size:
                    if dequantized.size > final_size:
                        # Truncate if the array is too large
                        dequantized = dequantized[:final_size]
                    else:
                        # Pad with zeros if the array is too small
                        print(f"Warning: Expected shape {expected_shape} requires {final_size} elements, "
                              f"but dequantized data has only {dequantized.size} elements. Padding with zeros.")
                        padded = np.zeros(final_size, dtype=np.float32)
                        padded[:dequantized.size] = dequantized
                        dequantized = padded
            else:
                raise ValueError(f"Unsupported quantization bits: {self.quantization_bits}")
            
            # Reshape to expected shape and convert to torch tensor
            return torch.from_numpy(dequantized.reshape(expected_shape))
            
        except FileNotFoundError as e:
            # Component not found, create a zero tensor as fallback
            print(f"Warning: {str(e)} - Creating zero tensor as fallback")
            return torch.zeros(expected_shape, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {component_name}: {str(e)}")
            return torch.zeros(expected_shape, dtype=torch.float32)
    
    def _dequantize_4bit(self, quantized_data: np.ndarray, scale: float, 
                         zero_point: int) -> np.ndarray:
        """Dequantize 4-bit quantized weights"""
        # Extract high and low 4-bits from each byte
        high_bits = (quantized_data >> 4) & 0xF
        low_bits = quantized_data & 0xF
        
        # Interleave the high and low bits
        result = np.empty(quantized_data.size * 2, dtype=np.uint8)
        result[0::2] = high_bits
        result[1::2] = low_bits
        
        # Dequantize to float32
        return (result.astype(np.float32) - zero_point) * scale


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    layer_norm_eps: float = 1e-12
    
    @classmethod
    def from_json(cls, json_path: str) -> "ModelConfig":
        """Load model configuration from a JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class RaspberryLLM:
    """Memory-efficient LLM implementation for Raspberry Pi"""
    
    def __init__(self, model_dir: str, quantization_bits: int = 8, sliding_window: int = 3):
        self.model_dir = model_dir
        self.quantization_bits = quantization_bits
        self.sliding_window = sliding_window
        
        # Load model configuration
        config_path = os.path.join(model_dir, "config.json")
        self.config = ModelConfig.from_json(config_path)
        
        # Initialize weight loader
        self.weight_loader = WeightLoader(model_dir, quantization_bits)
        
        # Track which layers are currently loaded
        self.loaded_layers = set()
        self.cached_weights = {}
        
        # Initialize tokenizer
        vocab_path = os.path.join(model_dir, "vocab.json")
        self.tokenizer = NeoxTokenizer(vocab_path)
        
        # Load embeddings (these stay in memory permanently)
        print("Loading embeddings...")
        self.token_embeddings = self.load_component(
            "embeddings.word_embeddings", 
            (self.config.vocab_size, self.config.hidden_size)
        )
        self.position_embeddings = self.load_component(
            "embeddings.position_embeddings",
            (self.config.max_position_embeddings, self.config.hidden_size)
        )
        print("Embeddings loaded")
        
        # Initialize KV cache
        self.kv_cache = None
    
    def load_component(self, component_name: str, shape: Tuple[int, ...]) -> torch.Tensor:
        """Load a model component with specific shape"""
        return self.weight_loader.load_and_dequantize(component_name, shape)
    
    def load_layer(self, layer_idx: int):
        """Load a specific transformer layer into memory"""
        if layer_idx in self.loaded_layers:
            return  # Already loaded
        
        start_time = time.time()
        print(f"Loading layer {layer_idx}...")
        
        # Load layer components
        h = self.config.hidden_size
        i = self.config.intermediate_size
        
        # Define components to load for this layer
        layer_components = {
            f"layer.{layer_idx}.attention.self.query.weight": (h, h),
            f"layer.{layer_idx}.attention.self.key.weight": (h, h),
            f"layer.{layer_idx}.attention.self.value.weight": (h, h),
            f"layer.{layer_idx}.attention.output.dense.weight": (h, h),
            f"layer.{layer_idx}.attention.output.LayerNorm.weight": (h,),
            f"layer.{layer_idx}.attention.output.LayerNorm.bias": (h,),
            f"layer.{layer_idx}.intermediate.dense.weight": (i, h),
            f"layer.{layer_idx}.intermediate.dense.bias": (i,),
            f"layer.{layer_idx}.output.dense.weight": (h, i),
            f"layer.{layer_idx}.output.dense.bias": (h,),
            f"layer.{layer_idx}.output.LayerNorm.weight": (h,),
            f"layer.{layer_idx}.output.LayerNorm.bias": (h,),
        }
        
        # Load all components for this layer
        for component_name, shape in layer_components.items():
            self.cached_weights[component_name] = self.load_component(component_name, shape)
        
        self.loaded_layers.add(layer_idx)
        print(f"Layer {layer_idx} loaded in {time.time() - start_time:.2f}s")
    
    def unload_layer(self, layer_idx: int):
        """Explicitly unload a layer to free memory"""
        if layer_idx not in self.loaded_layers:
            return  # Not loaded
        
        # Only unload if we have more than sliding_window layers loaded
        if len(self.loaded_layers) <= self.sliding_window:
            return
        
        print(f"Unloading layer {layer_idx}...")
        
        # Remove all weights associated with this layer
        keys_to_remove = [k for k in self.cached_weights if f"layer.{layer_idx}." in k]
        for key in keys_to_remove:
            del self.cached_weights[key]
        
        # Remove from loaded set
        self.loaded_layers.remove(layer_idx)
        
        # Force garbage collection
        gc.collect()
        try:
            torch.cuda.empty_cache()  # No-op on Raspberry Pi but good practice
        except:
            pass  # Ignore if CUDA not available
        print(f"Layer {layer_idx} unloaded")
    
    def layer_norm(self, hidden_states: torch.Tensor, 
                  weight: torch.Tensor, bias: torch.Tensor, 
                  eps: float = 1e-12) -> torch.Tensor:
        """Apply layer normalization"""
        mean = hidden_states.mean(-1, keepdim=True)
        std = hidden_states.std(-1, keepdim=True, unbiased=False)
        normalized = (hidden_states - mean) / (std + eps)
        return normalized * weight + bias
    
    def forward_layer(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process hidden states through a transformer layer"""
        # Make sure layer is loaded
        if layer_idx not in self.loaded_layers:
            self.load_layer(layer_idx)
        
        try:
            # Get dimensions
            batch_size, seq_len, hidden_dim = hidden_states.shape
            head_size = hidden_dim // self.config.num_attention_heads
            
            # 1. Self-attention
            # Query, Key, Value projections
            q_weight = self.cached_weights[f"layer.{layer_idx}.attention.self.query.weight"]
            k_weight = self.cached_weights[f"layer.{layer_idx}.attention.self.key.weight"]
            v_weight = self.cached_weights[f"layer.{layer_idx}.attention.self.value.weight"]
            
            # Reshape hidden states for matrix multiplication
            hidden_states_2d = hidden_states.reshape(-1, hidden_dim)
            
            # Compute query, key, value projections
            query = torch.matmul(hidden_states_2d, q_weight.t()).reshape(batch_size, seq_len, hidden_dim)
            key = torch.matmul(hidden_states_2d, k_weight.t()).reshape(batch_size, seq_len, hidden_dim)
            value = torch.matmul(hidden_states_2d, v_weight.t()).reshape(batch_size, seq_len, hidden_dim)
            
            # Reshape for attention computation
            query = query.reshape(batch_size, seq_len, self.config.num_attention_heads, head_size)
            key = key.reshape(batch_size, seq_len, self.config.num_attention_heads, head_size)
            value = value.reshape(batch_size, seq_len, self.config.num_attention_heads, head_size)
            
            # Transpose for batch matrix multiplication
            query = query.permute(0, 2, 1, 3)  # (batch, heads, seq, head_size)
            key = key.permute(0, 2, 1, 3)
            value = value.permute(0, 2, 1, 3)
            
            # Attention scores and weights
            attention_scores = torch.matmul(query, key.transpose(2, 3))
            attention_scores = attention_scores / (head_size ** 0.5)
            attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
            
            # Apply attention to values
            context = torch.matmul(attention_weights, value)
            context = context.permute(0, 2, 1, 3).contiguous()  # (batch, seq, heads, head_size)
            context = context.reshape(batch_size, seq_len, hidden_dim)
            
            # Self-attention output projection
            attention_output = torch.matmul(
                context.reshape(-1, hidden_dim), 
                self.cached_weights[f"layer.{layer_idx}.attention.output.dense.weight"].t()
            ).reshape(batch_size, seq_len, hidden_dim)
            
            # First residual connection and layer norm
            attention_output = attention_output + hidden_states
            attention_output = self.layer_norm(
                attention_output,
                self.cached_weights[f"layer.{layer_idx}.attention.output.LayerNorm.weight"],
                self.cached_weights[f"layer.{layer_idx}.attention.output.LayerNorm.bias"],
                self.config.layer_norm_eps
            )
            
            # Feed-forward network
            intermediate = torch.matmul(
                attention_output.reshape(-1, hidden_dim), 
                self.cached_weights[f"layer.{layer_idx}.intermediate.dense.weight"].t()
            ).reshape(batch_size, seq_len, -1)
            
            intermediate = intermediate + self.cached_weights[f"layer.{layer_idx}.intermediate.dense.bias"]
            
            # GELU activation
            intermediate = intermediate * 0.5 * (1.0 + torch.tanh(
                (2 / np.pi) ** 0.5 * (intermediate + 0.044715 * torch.pow(intermediate, 3))
            ))
            
            # Output projection
            layer_output = torch.matmul(
                intermediate.reshape(-1, self.config.intermediate_size), 
                self.cached_weights[f"layer.{layer_idx}.output.dense.weight"].t()
            ).reshape(batch_size, seq_len, hidden_dim)
            
            layer_output = layer_output + self.cached_weights[f"layer.{layer_idx}.output.dense.bias"]
            
            # Second residual connection and layer norm
            layer_output = layer_output + attention_output
            layer_output = self.layer_norm(
                layer_output,
                self.cached_weights[f"layer.{layer_idx}.output.LayerNorm.weight"],
                self.cached_weights[f"layer.{layer_idx}.output.LayerNorm.bias"],
                self.config.layer_norm_eps
            )
            
            return layer_output
            
        except Exception as e:
            print(f"Error in forward_layer {layer_idx}: {str(e)}")
            # Return input as fallback to maintain the flow
            return hidden_states
    
    def embed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings"""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        inputs_embeds = torch.zeros(
            (batch_size, seq_len, self.config.hidden_size), 
            dtype=torch.float32
        )
        
        # Manual embedding lookup (more memory efficient)
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = input_ids[b, s].item()
                if 0 <= token_id < self.config.vocab_size:
                    inputs_embeds[b, s] = self.token_embeddings[token_id]
        
        # Add position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long)
        # Handle positions beyond max_position_embeddings with modulo
        position_ids = position_ids % self.config.max_position_embeddings
        position_embeds = self.position_embeddings[position_ids]
        
        # Combine embeddings
        return inputs_embeds + position_embeds.unsqueeze(0)
    
    def initialize_kv_cache(self, batch_size: int):
        """Initialize key-value cache for generation"""
        head_size = self.config.hidden_size // self.config.num_attention_heads
        
        self.kv_cache = [{
            "key": torch.zeros(
                batch_size, 
                self.config.num_attention_heads, 
                0,  # Initially empty
                head_size
            ),
            "value": torch.zeros(
                batch_size, 
                self.config.num_attention_heads, 
                0,  # Initially empty
                head_size
            )
        } for _ in range(self.config.num_hidden_layers)]
    
    def generate(self, prompt: str, max_new_tokens: int = 20, 
                temperature: float = 0.7, repetition_penalty: float = 1.2, 
                top_k: int = 40) -> str:
        """Generate text auto-regressively with improved repetition handling"""
        print(f"Generating {max_new_tokens} tokens from prompt: '{prompt}'")
        
        # Encode prompt using our improved tokenizer
        input_ids = self.tokenizer.encode(prompt)
        batch_size, seq_len = input_ids.shape
        all_token_ids = input_ids.clone()
        
        # Process full input sequence through all layers
        hidden_states = self.embed_input(input_ids)
        
        print(f"Initial sequence length: {seq_len}")
        
        # Initial full forward pass for context
        for layer_idx in range(self.config.num_hidden_layers):
            # Manage sliding window of loaded layers
            if len(self.loaded_layers) > self.sliding_window:
                # Find oldest layer to unload
                oldest_layer = min(self.loaded_layers)
                self.unload_layer(oldest_layer)
            
            # Forward through layer
            hidden_states = self.forward_layer(layer_idx, hidden_states)
        
        # Initialize KV cache for generation
        self.initialize_kv_cache(batch_size)
        
        # Generated text buffer to collect tokens as we go
        generated_tokens = []
        
        # Track recent tokens for improved repetition detection
        recent_tokens = []
        recent_token_texts = []
        
        # Generate new tokens auto-regressively
        for i in range(max_new_tokens):
            try:
                # Get logits for next token prediction (last hidden state's last token)
                last_hidden = hidden_states[:, -1:, :]
                
                # Get logits by using the word embedding matrix transposed
                logits = torch.matmul(last_hidden, self.token_embeddings.t())
                logits = logits.squeeze(1)  # (batch, vocab_size)
                
                # Apply temperature to control randomness
                logits = logits / max(0.1, temperature)  # Prevent division by zero
                
                # Apply improved repetition penalty - more moderate approach
                if len(recent_tokens) > 0:
                    # Only look at previous tokens up to a reasonable window
                    lookback_window = min(10, len(recent_tokens))
                    prev_tokens = recent_tokens[-lookback_window:]
                    
                    # Apply repetition penalty
                    for token in set(prev_tokens):  # Use set to avoid duplicate penalties
                        logits[0, token] /= repetition_penalty
                    
                    # Detect simple repetition patterns
                    if len(recent_tokens) >= 6:
                        # Check for 2-token repeating pattern
                        if (recent_tokens[-1] == recent_tokens[-3] and 
                            recent_tokens[-2] == recent_tokens[-4]):
                            # Apply stronger penalty to these tokens
                            logits[0, recent_tokens[-1]] /= 2.0
                            logits[0, recent_tokens[-2]] /= 2.0
                        
                        # Check for 3-token repeating pattern
                        if (len(recent_tokens) >= 9 and
                            recent_tokens[-1] == recent_tokens[-4] == recent_tokens[-7] and
                            recent_tokens[-2] == recent_tokens[-5] == recent_tokens[-8] and
                            recent_tokens[-3] == recent_tokens[-6] == recent_tokens[-9]):
                            logits[0, recent_tokens[-1]] /= 3.0
                            logits[0, recent_tokens[-2]] /= 3.0
                            logits[0, recent_tokens[-3]] /= 3.0
                
                # Apply top-k sampling
                if top_k > 0:
                    # Get top-k values and indices
                    top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                    
                    # Create a new logits tensor with -inf everywhere
                    new_logits = torch.full_like(logits, float('-inf'))
                    
                    # Copy the top-k values back 
                    for b in range(batch_size):
                        new_logits[b, top_k_indices[b]] = top_k_values[b]
                    
                    logits = new_logits
                
                # Sample from the filtered distribution
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                
                # Get the actual token ID as an integer
                token_id = next_token_id.item()
                
                # Add to recent tokens
                recent_tokens.append(token_id)
                
                # Get the text representation for this token for better debugging
                token_text = self.tokenizer.id_to_token.get(token_id, "[UNK]")
                recent_token_texts.append(token_text)
                
                # Add to generated tokens list
                generated_tokens.append(token_id)
                
                # Debug info
                print(f"Generated token {i+1}/{max_new_tokens}: {repr(token_text)} (ID: {token_id})")
                
                # Add to output sequence
                all_token_ids = torch.cat([all_token_ids, next_token_id], dim=1)
                
                # Get embedding for new token
                new_token_embed = self.embed_input(next_token_id)
                
                # Process just the new token through all layers with KV caching
                token_hidden = new_token_embed
                
                # Process through all layers with KV cache
                for layer_idx in range(self.config.num_hidden_layers):
                    # Ensure this layer is loaded
                    if layer_idx not in self.loaded_layers:
                        self.load_layer(layer_idx)
                    
                    # Process token with cached context
                    token_hidden = self.forward_layer_with_kv_cache(layer_idx, token_hidden)
                    
                    # Manage sliding window of layers
                    if layer_idx >= self.sliding_window:
                        layer_to_unload = layer_idx - self.sliding_window
                        self.unload_layer(layer_to_unload)
                
                # Update hidden states with new token
                hidden_states = torch.cat([hidden_states, token_hidden], dim=1)
                
                # Check if we should stop generation (e.g., generated EOS token)
                if token_id == self.tokenizer.eos_token_id:
                    print("EOS token generated, stopping early")
                    break
                
            except Exception as e:
                print(f"Error during generation step {i}: {str(e)}")
                # Continue with next token on error
                continue
        
        # Decode all the generated tokens with our improved decoder
        generated_text = self.tokenizer.decode(generated_tokens)
        
        print("Generation complete")
        return prompt + generated_text
    
    def forward_layer_with_kv_cache(self, layer_idx: int, token_hidden: torch.Tensor) -> torch.Tensor:
        """Forward pass with KV cache for efficient generation"""
        # Ensure layer is loaded
        if layer_idx not in self.loaded_layers:
            self.load_layer(layer_idx)
        
        try:
            batch_size, seq_len, hidden_dim = token_hidden.shape
            assert seq_len == 1, "KV cache forward only works with single tokens"
            
            head_size = hidden_dim // self.config.num_attention_heads
            
            # Get weights
            q_weight = self.cached_weights[f"layer.{layer_idx}.attention.self.query.weight"]
            k_weight = self.cached_weights[f"layer.{layer_idx}.attention.self.key.weight"]
            v_weight = self.cached_weights[f"layer.{layer_idx}.attention.self.value.weight"]
            
            # Calculate query, key, value for new token
            query = torch.matmul(token_hidden.reshape(-1, hidden_dim), q_weight.t()).reshape(
                batch_size, 1, self.config.num_attention_heads, head_size).permute(0, 2, 1, 3)
            
            key_new = torch.matmul(token_hidden.reshape(-1, hidden_dim), k_weight.t()).reshape(
                batch_size, 1, self.config.num_attention_heads, head_size).permute(0, 2, 1, 3)
            
            value_new = torch.matmul(token_hidden.reshape(-1, hidden_dim), v_weight.t()).reshape(
                batch_size, 1, self.config.num_attention_heads, head_size).permute(0, 2, 1, 3)
            
            # Update key-value cache
            self.kv_cache[layer_idx]["key"] = torch.cat([self.kv_cache[layer_idx]["key"], key_new], dim=2)
            self.kv_cache[layer_idx]["value"] = torch.cat([self.kv_cache[layer_idx]["value"], value_new], dim=2)
            
            # Get cached keys and values
            key = self.kv_cache[layer_idx]["key"]  # (batch, heads, cache_len + 1, head_size)
            value = self.kv_cache[layer_idx]["value"]
            
            # Attention calculation
            attention_scores = torch.matmul(query, key.transpose(2, 3))
            attention_scores = attention_scores / (head_size ** 0.5)
            attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
            
            # Apply attention to values
            context = torch.matmul(attention_weights, value)  # (batch, heads, 1, head_size)
            context = context.permute(0, 2, 1, 3).reshape(batch_size, 1, hidden_dim)
            
            # Output projection
            attention_output = torch.matmul(
                context.reshape(-1, hidden_dim), 
                self.cached_weights[f"layer.{layer_idx}.attention.output.dense.weight"].t()
            ).reshape(batch_size, 1, hidden_dim)
            
            # First residual connection and layer norm
            attention_output = attention_output + token_hidden
            attention_output = self.layer_norm(
                attention_output,
                self.cached_weights[f"layer.{layer_idx}.attention.output.LayerNorm.weight"],
                self.cached_weights[f"layer.{layer_idx}.attention.output.LayerNorm.bias"],
                self.config.layer_norm_eps
            )
            
            # Feed-forward network
            intermediate = torch.matmul(
                attention_output.reshape(-1, hidden_dim), 
                self.cached_weights[f"layer.{layer_idx}.intermediate.dense.weight"].t()
            ).reshape(batch_size, 1, self.config.intermediate_size)
            
            intermediate = intermediate + self.cached_weights[f"layer.{layer_idx}.intermediate.dense.bias"]
            
            # GELU activation
            intermediate = intermediate * 0.5 * (1.0 + torch.tanh(
                (2 / np.pi) ** 0.5 * (intermediate + 0.044715 * torch.pow(intermediate, 3))
            ))
            
            # Output projection
            layer_output = torch.matmul(
                intermediate.reshape(-1, self.config.intermediate_size), 
                self.cached_weights[f"layer.{layer_idx}.output.dense.weight"].t()
            ).reshape(batch_size, 1, hidden_dim)
            
            layer_output = layer_output + self.cached_weights[f"layer.{layer_idx}.output.dense.bias"]
            
            # Second residual connection and layer norm
            layer_output = layer_output + attention_output
            layer_output = self.layer_norm(
                layer_output,
                self.cached_weights[f"layer.{layer_idx}.output.LayerNorm.weight"],
                self.cached_weights[f"layer.{layer_idx}.output.LayerNorm.bias"],
                self.config.layer_norm_eps
            )
            
            return layer_output
            
        except Exception as e:
            print(f"Error in forward_layer_with_kv_cache {layer_idx}: {str(e)}")
            # Return input as fallback
            return token_hidden


# --- Main execution ---

def main():
    # Model directory (with pre-quantized weights)
    model_dir = "quantized_model"
    
    # Set quantization bits (8-bit for better accuracy)
    quantization_bits = 8
    
    # Set sliding window size (smaller for memory savings)
    sliding_window = 3
    
    # Initialize model
    print("Initializing RaspberryLLM with improved tokenization...")
    model = RaspberryLLM(model_dir, quantization_bits=quantization_bits, sliding_window=sliding_window)
    
    # Example input prompt
    prompt = "In a world where technology"
    
    # Generate text with improved parameters
    start_time = time.time()
    generated_text = model.generate(
        prompt,
        max_new_tokens=30,
        temperature=0.7,        # Lower temperature for more coherent output
        repetition_penalty=1.2, # More moderate repetition penalty
        top_k=40                # Keep top_k the same
    )
    
    # Print stats
    total_time = time.time() - start_time
    tokens_per_second = 30 / total_time
    
    print("\nGeneration Summary:")
    print(f"Full text: {generated_text}")
    print(f"Time taken: {total_time:.2f}s")
    print(f"Generation speed: {tokens_per_second:.2f} tokens/sec")


if __name__ == "__main__":
    main()
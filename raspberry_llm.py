import os
import gc
import mmap
import time
import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# --- Configuration and Model Definition ---

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
        # Create memory map if it doesn't exist
        mem_map = self.create_memory_map(component_name)
        
        # Read quantized data from memory map
        quantized_data = np.frombuffer(mem_map, dtype=np.uint8)
        
        # Get dequantization parameters
        scale = self.scales.get(component_name, 0.1)
        zero_point = self.zero_points.get(component_name, 8)
        
        # Dequantize based on quantization bits
        if self.quantization_bits == 4:
            # For 4-bit, each byte contains 2 values
            dequantized = self._dequantize_4bit(
                quantized_data, scale, zero_point
            )
            # Calculate final shape size and ensure it matches
            final_size = np.prod(expected_shape)
            if final_size > dequantized.size:
                raise ValueError(
                    f"Expected shape {expected_shape} requires {final_size} elements, "
                    f"but dequantized data has only {dequantized.size} elements"
                )
            dequantized = dequantized[:final_size]
        elif self.quantization_bits == 8:
            # For 8-bit, direct dequantization
            dequantized = (quantized_data.astype(np.float32) - zero_point) * scale
            
            # Make sure the array is the right size
            final_size = np.prod(expected_shape)
            if dequantized.size != final_size:
                if dequantized.size > final_size:
                    # Truncate if the array is too large
                    dequantized = dequantized[:final_size]
                else:
                    # Pad with zeros if the array is too small (shouldn't happen)
                    raise ValueError(
                        f"Expected shape {expected_shape} requires {final_size} elements, "
                        f"but dequantized data has only {dequantized.size} elements"
                    )
        else:
            raise ValueError(f"Unsupported quantization bits: {self.quantization_bits}")
        
        # Reshape to expected shape and convert to torch tensor
        return torch.from_numpy(dequantized.reshape(expected_shape))
    
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
    

class RaspberryLLM:
    """Memory-efficient LLM implementation for Raspberry Pi"""
    
    def __init__(self, model_dir: str, quantization_bits: int = 4, sliding_window: int = 2):
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
        
        # KV cache for efficient generation
        self.kv_cache = None
        
        # Load vocabulary (for tokenization/detokenization)
        self.load_vocabulary()
        
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
    
    def load_vocabulary(self):
        """Load tokenizer vocabulary"""
        vocab_path = os.path.join(self.model_dir, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
                self.id_to_token = {int(v): k for k, v in self.vocab.items()}
        else:
            print("Warning: Vocabulary file not found")
            self.vocab = {}
            self.id_to_token = {}
    
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
        
        print(f"Unloading layer {layer_idx}...")
        
        # Remove all weights associated with this layer
        keys_to_remove = [k for k in self.cached_weights if f"layer.{layer_idx}." in k]
        for key in keys_to_remove:
            del self.cached_weights[key]
        
        # Remove from loaded set
        self.loaded_layers.remove(layer_idx)
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()  # No-op on Raspberry Pi but good practice
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
        
        # Get dimensions for debugging
        batch_size, seq_len, hidden_dim = hidden_states.shape
        print(f"Input hidden states shape: {hidden_states.shape}")
        
        # 1. Self-attention
        # Query, Key, Value projections
        q_weight = self.cached_weights[f"layer.{layer_idx}.attention.self.query.weight"]
        k_weight = self.cached_weights[f"layer.{layer_idx}.attention.self.key.weight"]
        v_weight = self.cached_weights[f"layer.{layer_idx}.attention.self.value.weight"]
        
        print(f"Query weight shape: {q_weight.shape}")
        
        # Check dimension compatibility
        if hidden_dim != q_weight.shape[1]:
            raise ValueError(f"Hidden dim {hidden_dim} doesn't match weight dim {q_weight.shape[1]}")
        
        # Reshape hidden states for matrix multiplication if needed
        hidden_states_2d = hidden_states.view(-1, hidden_dim)  # Combine batch and seq dims
        
        # Compute query, key, value projections with proper reshaping
        query = torch.matmul(hidden_states_2d, q_weight.t()).view(batch_size, seq_len, -1)
        key = torch.matmul(hidden_states_2d, k_weight.t()).view(batch_size, seq_len, -1)
        value = torch.matmul(hidden_states_2d, v_weight.t()).view(batch_size, seq_len, -1)
        
        # Reshape for attention computation
        head_size = self.config.hidden_size // self.config.num_attention_heads
        
        query = query.view(batch_size, seq_len, self.config.num_attention_heads, head_size)
        key = key.view(batch_size, seq_len, self.config.num_attention_heads, head_size)
        value = value.view(batch_size, seq_len, self.config.num_attention_heads, head_size)
        
        # Transpose for batch matrix multiplication
        query = query.transpose(1, 2)  # (batch, heads, seq, head_size)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        # Attention scores and weights
        attention_scores = torch.matmul(query, key.transpose(2, 3))
        attention_scores = attention_scores / (head_size ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous()  # (batch, seq, heads, head_size)
        context = context.view(batch_size, seq_len, self.config.hidden_size)
        
        # Self-attention output projection
        attention_output = torch.matmul(
            context, 
            self.cached_weights[f"layer.{layer_idx}.attention.output.dense.weight"].t()
        )
        
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
            attention_output.view(-1, hidden_dim), 
            self.cached_weights[f"layer.{layer_idx}.intermediate.dense.weight"].t()
        ).view(batch_size, seq_len, -1)
        
        intermediate = intermediate + self.cached_weights[f"layer.{layer_idx}.intermediate.dense.bias"]
        
        # GELU activation
        intermediate = intermediate * 0.5 * (1.0 + torch.tanh(
            (2 / np.pi) ** 0.5 * (intermediate + 0.044715 * torch.pow(intermediate, 3))
        ))
        
        # Output projection
        layer_output = torch.matmul(
            intermediate.view(-1, self.config.intermediate_size), 
            self.cached_weights[f"layer.{layer_idx}.output.dense.weight"].t()
        ).view(batch_size, seq_len, hidden_dim)
        
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
    
    def embed_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings"""
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings via lookup
        inputs_embeds = torch.zeros(
            (batch_size, seq_len, self.config.hidden_size), 
            dtype=torch.float32
        )
        
        # Manual embedding lookup to avoid large one-hot matrices
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = input_ids[b, s].item()
                if 0 <= token_id < self.config.vocab_size:
                    inputs_embeds[b, s] = self.token_embeddings[token_id]
        
        # Add position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long)
        position_embeds = self.position_embeddings[position_ids]
        
        # Combine embeddings (broadcasting position embeds to all batches)
        return inputs_embeds + position_embeds.unsqueeze(0)
    
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20, 
                temperature: float = 0.8, repetition_penalty: float = 1.3, 
                top_k: int = 40) -> torch.Tensor:
        """Generate text auto-regressively with sliding window of loaded layers"""
        print(f"Generating {max_new_tokens} tokens...")
        
        batch_size, seq_len = input_ids.shape
        all_token_ids = input_ids.clone()
        
        # Process input sequence
        hidden_states = self.embed_input(input_ids)
        
        # Make sure all layers are loaded initially
        for layer_idx in range(self.config.num_hidden_layers):
            # Manage sliding window of loaded layers
            if len(self.loaded_layers) > self.sliding_window + 1:
                # Find oldest layer to unload
                oldest_layer = min(self.loaded_layers)
                self.unload_layer(oldest_layer)
                
            # Ensure this layer is loaded
            if layer_idx not in self.loaded_layers:
                self.load_layer(layer_idx)
            
            # Forward through layer
            print(f"Processing layer {layer_idx} for input sequence...")
            hidden_states = self.forward_layer(layer_idx, hidden_states)
        
        # Initialize KV cache for efficient generation
        self.kv_cache = [{
            "key": None,
            "value": None
        } for _ in range(self.config.num_hidden_layers)]
        
        # Generate new tokens auto-regressively
        for i in range(max_new_tokens):
            # Get logits for next token prediction (last hidden state's last token)
            last_hidden = hidden_states[:, -1:, :]
            
            # Get logits by using the word embedding matrix transposed
            # This avoids loading a separate LM head
            logits = torch.matmul(last_hidden, self.token_embeddings.t())
            logits = logits.squeeze(1)  # (batch, vocab_size)
            
            # Apply temperature to control randomness
            logits = logits / temperature
            
            # Apply repetition penalty to reduce token repetition
            if i > 0:
                # Get previously generated tokens (up to last 10)
                prev_tokens = all_token_ids[:, -10:] if all_token_ids.size(1) >= 10 else all_token_ids
                # Apply penalty to previously generated tokens
                for prev_token in prev_tokens.unique():
                    token_idx = prev_token.item()
                    # Reduce probability of previously generated tokens
                    logits[:, token_idx] = logits[:, token_idx] / repetition_penalty
            
            # Print top token candidates for debugging
            probs = torch.nn.functional.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=min(5, probs.size(-1)))
            print("\nTop 5 token candidates:")
            for j, (prob, idx) in enumerate(zip(topk_probs[0], topk_indices[0])):
                token = self.id_to_token.get(idx.item(), "[UNK]")
                print(f"  {j+1}: '{token}' (prob: {prob.item():.4f})")
            
            # Apply top-k sampling to filter out unlikely tokens
            if top_k > 0:
                # Zero out all logits below the top k
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Print the generated token
            if next_token_id.item() in self.id_to_token:
                token_str = self.id_to_token[next_token_id.item()]
                print(f"Generated token {i+1}/{max_new_tokens}: {token_str}")
            else:
                print(f"Generated token {i+1}/{max_new_tokens}: [UNKNOWN]")
            
            # Add to output sequence
            all_token_ids = torch.cat([all_token_ids, next_token_id], dim=1)
            
            # Get embedding for new token
            new_token_embed = self.embed_input(next_token_id)
            
            # Append new token embedding to hidden states (for context)
            hidden_states = torch.cat([hidden_states, new_token_embed], dim=1)
            
            # Forward only the new token through all layers
            # This is more efficient than re-computing the entire sequence
            new_hidden = new_token_embed
            
            # Ensure all layers are loaded for processing the new token
            for layer_idx in range(self.config.num_hidden_layers):
                # Make sure this layer is loaded
                if layer_idx not in self.loaded_layers:
                    self.load_layer(layer_idx)
                    
                # Forward pass just for the new token, with KV caching
                new_hidden = self.forward_layer_with_kv_cache(
                    layer_idx, new_hidden, hidden_states, (seq_len + i)
                )
            
            # Update the last hidden state
            hidden_states[:, -1:, :] = new_hidden
            
            # Unload layers to maintain the sliding window
            if len(self.loaded_layers) > self.sliding_window:
                # Find oldest layer to unload
                oldest_layer = min(self.loaded_layers)
                self.unload_layer(oldest_layer)
        
        print("Generation complete")
        return all_token_ids
    
    def forward_layer_with_kv_cache(self, layer_idx: int, token_hidden: torch.Tensor,
                                  context_hidden: torch.Tensor, position: int) -> torch.Tensor:
        """Forward pass with KV cache for efficient generation"""
        # Make sure layer is loaded
        if layer_idx not in self.loaded_layers:
            self.load_layer(layer_idx)
        
        batch_size = token_hidden.shape[0]
        head_size = self.config.hidden_size // self.config.num_attention_heads
        hidden_dim = self.config.hidden_size
        
        # 1. Self-attention for the single token
        # Query, Key, Value projections
        q_weight = self.cached_weights[f"layer.{layer_idx}.attention.self.query.weight"]
        k_weight = self.cached_weights[f"layer.{layer_idx}.attention.self.key.weight"]
        v_weight = self.cached_weights[f"layer.{layer_idx}.attention.self.value.weight"]
        
        # Calculate query for new token
        query = torch.matmul(token_hidden.view(-1, hidden_dim), q_weight.t()).view(
            batch_size, 1, self.config.num_attention_heads, head_size)
        
        # Calculate key and value for new token
        key_new = torch.matmul(token_hidden.view(-1, hidden_dim), k_weight.t()).view(
            batch_size, 1, self.config.num_attention_heads, head_size)
        value_new = torch.matmul(token_hidden.view(-1, hidden_dim), v_weight.t()).view(
            batch_size, 1, self.config.num_attention_heads, head_size)
        
        # Update key-value cache
        if self.kv_cache[layer_idx]["key"] is None:
            # First token in generation - no cache yet
            self.kv_cache[layer_idx]["key"] = key_new
            self.kv_cache[layer_idx]["value"] = value_new
        else:
            # Concatenate new k/v with cached k/v
            self.kv_cache[layer_idx]["key"] = torch.cat(
                [self.kv_cache[layer_idx]["key"], key_new], dim=1)
            self.kv_cache[layer_idx]["value"] = torch.cat(
                [self.kv_cache[layer_idx]["value"], value_new], dim=1)
        
        # Get full key and value tensors from cache
        key = self.kv_cache[layer_idx]["key"]
        value = self.kv_cache[layer_idx]["value"]
        
        # Transpose for attention calculation
        query = query.transpose(1, 2)  # (batch, heads, 1, head_size)
        key = key.transpose(1, 2)      # (batch, heads, seq_len, head_size)
        value = value.transpose(1, 2)  # (batch, heads, seq_len, head_size)
        
        # Attention scores and weights
        attention_scores = torch.matmul(query, key.transpose(2, 3))  # (batch, heads, 1, seq_len)
        attention_scores = attention_scores / (head_size ** 0.5)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, value)  # (batch, heads, 1, head_size)
        context = context.transpose(1, 2).contiguous()    # (batch, 1, heads, head_size)
        context = context.view(batch_size, 1, self.config.hidden_size)
        
        # Self-attention output projection
        attention_output = torch.matmul(
            context.view(-1, hidden_dim), 
            self.cached_weights[f"layer.{layer_idx}.attention.output.dense.weight"].t()
        ).view(batch_size, 1, hidden_dim)
        
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
            attention_output.view(-1, hidden_dim), 
            self.cached_weights[f"layer.{layer_idx}.intermediate.dense.weight"].t()
        ).view(batch_size, 1, -1)
        
        intermediate = intermediate + self.cached_weights[f"layer.{layer_idx}.intermediate.dense.bias"]
        
        # GELU activation
        intermediate = intermediate * 0.5 * (1.0 + torch.tanh(
            (2 / np.pi) ** 0.5 * (intermediate + 0.044715 * torch.pow(intermediate, 3))
        ))
        
        # Output projection
        layer_output = torch.matmul(
            intermediate.view(-1, self.config.intermediate_size), 
            self.cached_weights[f"layer.{layer_idx}.output.dense.weight"].t()
        ).view(batch_size, 1, hidden_dim)
        
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


# --- Helper functions for tokenization and model preparation ---

def tokenize_text(text: str, vocab: Dict[str, int], max_length: int = 512) -> torch.Tensor:
    """Simple word-level tokenization"""
    words = text.split()
    tokens = []
    
    for word in words:
        if word in vocab:
            tokens.append(vocab[word])
        else:
            # Handle unknown words
            tokens.append(vocab.get("<unk>", 0))
    
    # Truncate if needed
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    
    # Convert to tensor and ensure it's 2D (batch_size=1, seq_len)
    return torch.tensor([tokens], dtype=torch.long)

def detokenize_ids(token_ids: List[int], id_to_token: Dict[int, str]) -> str:
    """Convert token IDs back to text"""
    return " ".join([id_to_token.get(id, "<unk>") for id in token_ids])

def prepare_model_directory(model_path: str, output_dir: str, num_bits: int = 4):
    """Prepare model files for efficient loading (simplified)"""
    # This would convert a standard model into our memory-mapped format
    # with proper quantization for Raspberry Pi
    # For a full implementation, this would:
    # 1. Load the original model weights
    # 2. Quantize them to 4 or 8 bits
    # 3. Save in a memory-mapped friendly format
    # 4. Save quantization parameters
    pass


# --- Main execution ---

def main():
    # Model directory (with pre-quantized weights)
    model_dir = "quantized_model"
    
    # Set quantization bits (8-bit instead of 4-bit for better accuracy)
    quantization_bits = 8
    
    # Set sliding window size (decrease for larger models)
    sliding_window = 2
    
    # Initialize model
    print("Initializing RaspberryLLM...")
    model = RaspberryLLM(model_dir, quantization_bits=quantization_bits, sliding_window=sliding_window)
    
    # Example input prompt
    prompt = "In a world where technology"
    input_ids = tokenize_text(prompt, model.vocab)
    
    # Generate text with temperature and repetition penalty
    print(f"Generating from prompt: '{prompt}'")
    start_time = time.time()
    
    output_ids = model.generate(
        input_ids,
        max_new_tokens=30,
        temperature=0.8,       # Controls randomness: 0.7-1.0 is a good range
        repetition_penalty=1.5, # Stronger penalty to reduce repetition: 1.2-1.8 is a good range
        top_k=40                # Limits to top K tokens: 20-50 is a good range
    )
    
    # Print generated text and statistics
    generated_ids = output_ids[0].tolist()[input_ids.shape[1]:]
    generated_text = detokenize_ids(generated_ids, model.id_to_token)
    
    total_time = time.time() - start_time
    tokens_per_second = len(generated_ids) / total_time
    
    print("\nGeneration complete!")
    print(f"Generated text: {prompt} {generated_text}")
    print(f"Time taken: {total_time:.2f}s")
    print(f"Generation speed: {tokens_per_second:.2f} tokens/sec")
    print(f"Memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()

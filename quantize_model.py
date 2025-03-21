import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from tqdm import tqdm
import gc
import time
from safetensors import safe_open
from safetensors.torch import save_file

def quantize_to_8bit(tensor, chunk_size=10000):
    """Quantize a tensor to 8-bit precision using chunk processing"""
    # Process on CPU to save memory
    tensor = tensor.detach().cpu()
    
    # Find min and max values
    min_val = float(tensor.min().item())
    max_val = float(tensor.max().item())
    
    # Calculate scale and zero point
    scale = float((max_val - min_val) / 255)  # 8-bit = 256 values (0-255)
    if scale == 0:  # Handle constant tensors
        scale = 1.0
    zero_point = int(round((-min_val) / scale)) if scale != 0 else 128
    
    # Calculate size
    tensor_size = tensor.numel()
    
    # Allocate output array
    quantized = np.zeros(tensor_size, dtype=np.uint8)
    
    # Process in chunks to minimize memory usage
    tensor_flat = tensor.reshape(-1)
    for i in range(0, tensor_size, chunk_size):
        end_idx = min(i + chunk_size, tensor_size)
        chunk = tensor_flat[i:end_idx]
        
        # Quantize to 8-bit values
        chunk_quantized = torch.clamp(torch.round((chunk - min_val) / scale), 0, 255).to(torch.uint8)
        quantized[i:end_idx] = chunk_quantized.numpy()
    
    # Clean up to free memory
    del tensor
    del tensor_flat
    del chunk_quantized
    gc.collect()
    
    return quantized, scale, zero_point

def convert_model_optimized(model_name, output_dir, use_safe_tensors=True):
    """Convert a HuggingFace model to 8-bit quantized format, optimized for low memory"""
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    
    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Save tokenizer and config
    tokenizer.save_pretrained(output_dir)
    
    # Create our model config
    model_config = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "intermediate_size": getattr(config, "intermediate_size", config.hidden_size * 4),
        "max_position_embeddings": config.max_position_embeddings,
        "layer_norm_eps": 1e-5
    }
    
    print(f"Saving config to {output_dir}...")
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Prepare for quantization
    quant_params = {"scales": {}, "zero_points": {}}
    
    # Load model in safetensors format if possible
    try:
        # Try to load model with low CPU memory usage
        print(f"Loading model {model_name} in half precision...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Trying to load with standard settings...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Get model architecture type (GPT-Neo, Pythia, GPT-2, etc.)
    model_type = config.model_type.lower()
    print(f"Detected model type: {model_type}")
    
    # Map weights based on model architecture
    mapping_info = create_weight_mapping(model, model_type)
    
    # Process all weights
    h = model_config["hidden_size"]
    
    # Process embeddings first
    print("Processing embeddings...")
    
    # Get embedding weights based on model type
    word_embedding = None
    position_embedding = None
    
    # Extract embeddings based on model type
    if model_type == "gpt_neox" or model_type == "pythia":
        if hasattr(model, "gpt_neox"):
            word_embedding = model.gpt_neox.embed_in.weight
            if hasattr(model.gpt_neox, "pos_embed"):
                position_embedding = model.gpt_neox.pos_embed.weight
    elif model_type == "gpt2":
        if hasattr(model, "transformer"):
            word_embedding = model.transformer.wte.weight
            position_embedding = model.transformer.wpe.weight
    elif model_type == "llama" or model_type == "mistral":
        if hasattr(model, "model"):
            word_embedding = model.model.embed_tokens.weight
            # LLaMA/Mistral use rotary embeddings, so we'll create a placeholder
            position_embedding = torch.zeros(model_config["max_position_embeddings"], h)
    
    # Process word embeddings
    if word_embedding is not None:
        component_name = "embeddings.word_embeddings"
        packed, scale, zero_point = quantize_to_8bit(word_embedding)
        with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
            f.write(packed.tobytes())
        quant_params["scales"][component_name] = scale
        quant_params["zero_points"][component_name] = zero_point
    
    # Process position embeddings
    if position_embedding is not None:
        component_name = "embeddings.position_embeddings"
        packed, scale, zero_point = quantize_to_8bit(position_embedding)
        with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
            f.write(packed.tobytes())
        quant_params["scales"][component_name] = scale
        quant_params["zero_points"][component_name] = zero_point
    else:
        # Create placeholder position embeddings if not found
        print("Creating placeholder position embeddings...")
        position_embeddings = torch.zeros(model_config["max_position_embeddings"], h)
        component_name = "embeddings.position_embeddings"
        
        packed, scale, zero_point = quantize_to_8bit(position_embeddings)
        with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
            f.write(packed.tobytes())
        
        quant_params["scales"][component_name] = scale
        quant_params["zero_points"][component_name] = zero_point
    
    # Process layers
    for layer_idx in tqdm(range(model_config["num_hidden_layers"]), desc="Processing layers"):
        print(f"\nProcessing layer {layer_idx}...")
        
        # Get layer based on model type
        layer = None
        if model_type == "gpt_neox" or model_type == "pythia":
            layer = model.gpt_neox.layers[layer_idx]
        elif model_type == "gpt2":
            layer = model.transformer.h[layer_idx]
        elif model_type == "llama" or model_type == "mistral":
            layer = model.model.layers[layer_idx]
        
        if layer is None:
            print(f"Warning: Could not find layer {layer_idx}")
            continue
        
        # Process all weights for this layer based on mapping
        for attr_path, target_path in mapping_info["layer_mappings"].items():
            # Replace layer index placeholder
            source_path = attr_path.replace("{layer_idx}", str(layer_idx))
            
            if isinstance(target_path, list):
                # Handle special cases like QKV weights
                process_special_weights(layer, source_path, target_path, layer_idx, 
                                      model_config, output_dir, quant_params)
            else:
                target_path = target_path.replace("{layer_idx}", str(layer_idx))
                process_weight(layer, source_path, target_path, output_dir, quant_params)
        
        # Clear memory after processing each layer
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save quantization parameters
    with open(os.path.join(output_dir, "quant_params.json"), "w") as f:
        json.dump(quant_params, f, indent=2)
    
    total_time = time.time() - start_time
    print(f"Model conversion complete in {total_time:.2f} seconds")
    print(f"Quantized model saved to {output_dir}")
    
    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()

def process_weight(obj, attr_path, target_path, output_dir, quant_params):
    """Process a single weight tensor"""
    # Follow the attribute path to get the tensor
    parts = attr_path.split('.')
    current = obj
    
    try:
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                print(f"Warning: Attribute {part} not found in {type(current)}")
                return
        
        if not isinstance(current, torch.Tensor):
            print(f"Warning: {attr_path} is not a tensor, but {type(current)}")
            return
        
        tensor = current
        
        # Process the tensor based on its type
        if tensor.dim() <= 1 or tensor.shape[0] == 1 or "bias" in target_path:
            # Save biases and 1D tensors directly as float32
            with open(os.path.join(output_dir, f"{target_path}.bin"), "wb") as f:
                f.write(tensor.cpu().numpy().astype(np.float32).tobytes())
            quant_params["scales"][target_path] = 1.0
            quant_params["zero_points"][target_path] = 0
        else:
            # Quantize matrices
            packed, scale, zero_point = quantize_to_8bit(tensor)
            with open(os.path.join(output_dir, f"{target_path}.bin"), "wb") as f:
                f.write(packed.tobytes())
            quant_params["scales"][target_path] = scale
            quant_params["zero_points"][target_path] = zero_point
            
        print(f"Processed {attr_path} -> {target_path}")
    except Exception as e:
        print(f"Error processing {attr_path}: {str(e)}")

def process_special_weights(layer, source_path, target_paths, layer_idx, 
                           model_config, output_dir, quant_params):
    """Process special weights like QKV that need to be split"""
    # Follow the attribute path to get the tensor
    parts = source_path.split('.')
    current = layer
    
    try:
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                print(f"Warning: Attribute {part} not found in {type(current)}")
                return
        
        if not isinstance(current, torch.Tensor):
            print(f"Warning: {source_path} is not a tensor, but {type(current)}")
            return
        
        tensor = current
        
        # Check if this is QKV weights that need to be split
        if any("query" in path for path in target_paths) and \
           any("key" in path for path in target_paths) and \
           any("value" in path for path in target_paths):
            
            print(f"Processing QKV weight: {source_path}")
            process_qkv_weights(tensor, target_paths, layer_idx, model_config, output_dir, quant_params)
        else:
            print(f"Unhandled special weight: {source_path}")
    
    except Exception as e:
        print(f"Error processing special weight {source_path}: {str(e)}")

def process_qkv_weights(qkv_tensor, target_paths, layer_idx, model_config, output_dir, quant_params):
    """Split and process QKV weights"""
    # Different models store QKV differently, inspect shape
    h = model_config["hidden_size"]
    num_heads = model_config["num_attention_heads"]
    head_dim = h // num_heads
    
    print(f"QKV tensor shape: {qkv_tensor.shape}")
    
    # Try to determine QKV format
    if qkv_tensor.shape[0] == 3 * h:
        # Format: [3*hidden, hidden]
        q_weight = qkv_tensor[:h]
        k_weight = qkv_tensor[h:2*h]
        v_weight = qkv_tensor[2*h:]
    
    elif qkv_tensor.shape[0] == h and qkv_tensor.shape[1] == 3 * h:
        # Format: [hidden, 3*hidden]
        q_weight = qkv_tensor[:, :h]
        k_weight = qkv_tensor[:, h:2*h]
        v_weight = qkv_tensor[:, 2*h:]
    
    elif len(qkv_tensor.shape) >= 3:
        # Format: More complex, reshape first
        qkv_weight = qkv_tensor.view(h, num_heads, 3, head_dim).transpose(1, 2)
        q_weight = qkv_weight[:, 0].reshape(h, h)
        k_weight = qkv_weight[:, 1].reshape(h, h)
        v_weight = qkv_weight[:, 2].reshape(h, h)
    
    else:
        # Try best effort fallback
        qkv_size = qkv_tensor.shape[0] * qkv_tensor.shape[1]
        if qkv_size == 3 * h * h:
            # Reshape to standard format
            qkv_reshaped = qkv_tensor.reshape(3, h, h)
            q_weight = qkv_reshaped[0]
            k_weight = qkv_reshaped[1]
            v_weight = qkv_reshaped[2]
        else:
            raise ValueError(f"Unrecognized QKV format with shape {qkv_tensor.shape}")
    
    # Map target paths correctly
    q_path = next(path for path in target_paths if "query" in path)
    k_path = next(path for path in target_paths if "key" in path)
    v_path = next(path for path in target_paths if "value" in path)
    
    # Replace layer index placeholders
    q_path = q_path.replace("{layer_idx}", str(layer_idx))
    k_path = k_path.replace("{layer_idx}", str(layer_idx))
    v_path = v_path.replace("{layer_idx}", str(layer_idx))
    
    # Save Q weights
    packed, scale, zero_point = quantize_to_8bit(q_weight)
    with open(os.path.join(output_dir, f"{q_path}.bin"), "wb") as f:
        f.write(packed.tobytes())
    quant_params["scales"][q_path] = scale
    quant_params["zero_points"][q_path] = zero_point
    
    # Save K weights
    packed, scale, zero_point = quantize_to_8bit(k_weight)
    with open(os.path.join(output_dir, f"{k_path}.bin"), "wb") as f:
        f.write(packed.tobytes())
    quant_params["scales"][k_path] = scale
    quant_params["zero_points"][k_path] = zero_point
    
    # Save V weights
    packed, scale, zero_point = quantize_to_8bit(v_weight)
    with open(os.path.join(output_dir, f"{v_path}.bin"), "wb") as f:
        f.write(packed.tobytes())
    quant_params["scales"][v_path] = scale
    quant_params["zero_points"][v_path] = zero_point
    
    print(f"Processed QKV weights: {q_path}, {k_path}, {v_path}")

def create_weight_mapping(model, model_type):
    """Create mapping from model attributes to our structure based on model type"""
    mapping_info = {"layer_mappings": {}}
    
    if model_type == "gpt_neox" or model_type == "pythia":
        # Pythia/GPT-NeoX mappings
        mapping_info["layer_mappings"] = {
            "attention.query_key_value.weight": [
                "layer.{layer_idx}.attention.self.query.weight",
                "layer.{layer_idx}.attention.self.key.weight",
                "layer.{layer_idx}.attention.self.value.weight"
            ],
            "attention.dense.weight": "layer.{layer_idx}.attention.output.dense.weight",
            "input_layernorm.weight": "layer.{layer_idx}.attention.output.LayerNorm.weight",
            "input_layernorm.bias": "layer.{layer_idx}.attention.output.LayerNorm.bias",
            "mlp.dense_h_to_4h.weight": "layer.{layer_idx}.intermediate.dense.weight",
            "mlp.dense_h_to_4h.bias": "layer.{layer_idx}.intermediate.dense.bias",
            "mlp.dense_4h_to_h.weight": "layer.{layer_idx}.output.dense.weight",
            "mlp.dense_4h_to_h.bias": "layer.{layer_idx}.output.dense.bias",
            "post_attention_layernorm.weight": "layer.{layer_idx}.output.LayerNorm.weight",
            "post_attention_layernorm.bias": "layer.{layer_idx}.output.LayerNorm.bias"
        }
    
    elif model_type == "gpt2":
        # GPT-2 mappings
        mapping_info["layer_mappings"] = {
            "attn.c_attn.weight": [
                "layer.{layer_idx}.attention.self.query.weight",
                "layer.{layer_idx}.attention.self.key.weight",
                "layer.{layer_idx}.attention.self.value.weight"
            ],
            "attn.c_attn.bias": "layer.{layer_idx}.attention.self.bias",
            "attn.c_proj.weight": "layer.{layer_idx}.attention.output.dense.weight",
            "attn.c_proj.bias": "layer.{layer_idx}.attention.output.dense.bias",
            "ln_1.weight": "layer.{layer_idx}.attention.output.LayerNorm.weight",
            "ln_1.bias": "layer.{layer_idx}.attention.output.LayerNorm.bias",
            "mlp.c_fc.weight": "layer.{layer_idx}.intermediate.dense.weight",
            "mlp.c_fc.bias": "layer.{layer_idx}.intermediate.dense.bias",
            "mlp.c_proj.weight": "layer.{layer_idx}.output.dense.weight",
            "mlp.c_proj.bias": "layer.{layer_idx}.output.dense.bias",
            "ln_2.weight": "layer.{layer_idx}.output.LayerNorm.weight",
            "ln_2.bias": "layer.{layer_idx}.output.LayerNorm.bias"
        }
    
    elif model_type == "llama" or model_type == "mistral":
        # LLaMA/Mistral mappings
        mapping_info["layer_mappings"] = {
            "self_attn.q_proj.weight": "layer.{layer_idx}.attention.self.query.weight",
            "self_attn.k_proj.weight": "layer.{layer_idx}.attention.self.key.weight",
            "self_attn.v_proj.weight": "layer.{layer_idx}.attention.self.value.weight",
            "self_attn.o_proj.weight": "layer.{layer_idx}.attention.output.dense.weight",
            "input_layernorm.weight": "layer.{layer_idx}.attention.output.LayerNorm.weight",
            "post_attention_layernorm.weight": "layer.{layer_idx}.output.LayerNorm.weight",
            "mlp.gate_proj.weight": "layer.{layer_idx}.intermediate.gate.weight",
            "mlp.up_proj.weight": "layer.{layer_idx}.intermediate.dense.weight",
            "mlp.down_proj.weight": "layer.{layer_idx}.output.dense.weight"
        }
        # LLaMA uses RMSNorm without bias
        # Add empty biases for compatibility
        for name in ["self.bias", "output.dense.bias", "output.LayerNorm.bias",
                    "intermediate.dense.bias", "attention.output.LayerNorm.bias"]:
            path = f"layer.{{layer_idx}}.attention.{name}"
            mapping_info["layer_mappings"][f"__{name}"] = path
    
    else:
        print(f"Warning: Unknown model type {model_type}. Creating generic mapping.")
        # Generic mapping as fallback
        mapping_info["layer_mappings"] = {
            "attention.query.weight": "layer.{layer_idx}.attention.self.query.weight",
            "attention.key.weight": "layer.{layer_idx}.attention.self.key.weight",
            "attention.value.weight": "layer.{layer_idx}.attention.self.value.weight",
            "attention.output.weight": "layer.{layer_idx}.attention.output.dense.weight",
            "attention.output.bias": "layer.{layer_idx}.attention.output.dense.bias",
            "layernorm1.weight": "layer.{layer_idx}.attention.output.LayerNorm.weight",
            "layernorm1.bias": "layer.{layer_idx}.attention.output.LayerNorm.bias",
            "intermediate.weight": "layer.{layer_idx}.intermediate.dense.weight",
            "intermediate.bias": "layer.{layer_idx}.intermediate.dense.bias",
            "output.weight": "layer.{layer_idx}.output.dense.weight",
            "output.bias": "layer.{layer_idx}.output.dense.bias",
            "layernorm2.weight": "layer.{layer_idx}.output.LayerNorm.weight",
            "layernorm2.bias": "layer.{layer_idx}.output.LayerNorm.bias"
        }
    
    return mapping_info

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantize a HuggingFace model for RaspberryLLM")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--output", type=str, default="quantized_model", help="Output directory")
    parser.add_argument("--safetensors", action="store_true", help="Use safetensors format")
    args = parser.parse_args()
    
    # Install safetensors if needed
    try:
        import safetensors
    except ImportError:
        print("Installing safetensors...")
        import subprocess
        subprocess.check_call(["pip", "install", "safetensors"])
    
    convert_model_optimized(args.model, args.output, args.safetensors)
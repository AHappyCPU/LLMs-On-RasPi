import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
import gc
import time
import math
from safetensors import safe_open

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

def convert_model_optimized(model_name, output_dir):
    """Convert a HuggingFace model to 8-bit quantized format, optimized for low memory"""
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    
    print(f"Loading config for {model_name}...")
    config = AutoConfig.from_pretrained(model_name)
    
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create our model config
    model_config = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "intermediate_size": config.hidden_size * 4,
        "max_position_embeddings": config.max_position_embeddings,
        "layer_norm_eps": 1e-5
    }
    
    print(f"Saving config to {output_dir}...")
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(model_config, f)
    
    print(f"Saving vocabulary to {output_dir}...")
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(tokenizer.get_vocab(), f)
    
    # Prepare for quantization
    quant_params = {"scales": {}, "zero_points": {}}
    
    # Determine paths for weight files
    if os.path.exists(os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), "models--" + model_name.replace("/", "--"), "model.safetensors")):
        weights_path = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), "models--" + model_name.replace("/", "--"), "model.safetensors")
        print(f"Using cached weights: {weights_path}")
        use_safetensors = True
    elif os.path.exists(os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), "models--" + model_name.replace("/", "--"), "pytorch_model.bin")):
        weights_path = os.path.join(os.path.expanduser("~/.cache/huggingface/hub"), "models--" + model_name.replace("/", "--"), "pytorch_model.bin")
        print(f"Using cached weights: {weights_path}")
        use_safetensors = False
    else:
        print("Model weights not found in cache. Please run 'from transformers import AutoModel; AutoModel.from_pretrained(model_name)' first.")
        return
    
    # Process weights
    h = model_config["hidden_size"]
    i = model_config["intermediate_size"]
    
    # Map of original parameter names to our structure
    # This is for the Pythia model family
    name_mapping = {}
    
    # Add embeddings mapping
    name_mapping["gpt_neox.embed_in.weight"] = "embeddings.word_embeddings"
    
    # Add position embeddings (if rotary, create placeholder)
    if not hasattr(config, "rotary_pct"):
        if "gpt_neox.pos_embed.weight" in name_mapping:
            name_mapping["gpt_neox.pos_embed.weight"] = "embeddings.position_embeddings"
    else:
        print("Model uses rotary embeddings. Creating placeholder position embeddings...")
        position_embeddings = torch.zeros(model_config["max_position_embeddings"], h)
        component_name = "embeddings.position_embeddings"
        
        packed, scale, zero_point = quantize_to_8bit(position_embeddings)
        with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
            f.write(packed.tobytes())
        
        quant_params["scales"][component_name] = scale
        quant_params["zero_points"][component_name] = zero_point
    
    # Add layer mappings
    for layer_idx in range(model_config["num_hidden_layers"]):
        layer_name = f"gpt_neox.layers.{layer_idx}"
        name_mapping[f"{layer_name}.attention.query_key_value.weight"] = [
            f"layer.{layer_idx}.attention.self.query.weight",
            f"layer.{layer_idx}.attention.self.key.weight",
            f"layer.{layer_idx}.attention.self.value.weight"
        ]
        name_mapping[f"{layer_name}.attention.dense.weight"] = f"layer.{layer_idx}.attention.output.dense.weight"
        name_mapping[f"{layer_name}.input_layernorm.weight"] = f"layer.{layer_idx}.attention.output.LayerNorm.weight"
        name_mapping[f"{layer_name}.input_layernorm.bias"] = f"layer.{layer_idx}.attention.output.LayerNorm.bias"
        name_mapping[f"{layer_name}.mlp.dense_h_to_4h.weight"] = f"layer.{layer_idx}.intermediate.dense.weight"
        name_mapping[f"{layer_name}.mlp.dense_h_to_4h.bias"] = f"layer.{layer_idx}.intermediate.dense.bias"
        name_mapping[f"{layer_name}.mlp.dense_4h_to_h.weight"] = f"layer.{layer_idx}.output.dense.weight"
        name_mapping[f"{layer_name}.mlp.dense_4h_to_h.bias"] = f"layer.{layer_idx}.output.dense.bias"
        name_mapping[f"{layer_name}.post_attention_layernorm.weight"] = f"layer.{layer_idx}.output.LayerNorm.weight"
        name_mapping[f"{layer_name}.post_attention_layernorm.bias"] = f"layer.{layer_idx}.output.LayerNorm.bias"
    
    # Process weights based on file format
    if use_safetensors:
        # Process with safetensors
        with safe_open(weights_path, framework="pt") as f:
            # Process embeddings first
            print("Processing embeddings...")
            if "gpt_neox.embed_in.weight" in f.keys():
                embed_tensor = f.get_tensor("gpt_neox.embed_in.weight")
                component_name = "embeddings.word_embeddings"
                
                packed, scale, zero_point = quantize_to_8bit(embed_tensor)
                with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as out_f:
                    out_f.write(packed.tobytes())
                
                quant_params["scales"][component_name] = scale
                quant_params["zero_points"][component_name] = zero_point
            
            # Check for position embeddings
            if "gpt_neox.pos_embed.weight" in f.keys():
                pos_embed_tensor = f.get_tensor("gpt_neox.pos_embed.weight")
                component_name = "embeddings.position_embeddings"
                
                packed, scale, zero_point = quantize_to_8bit(pos_embed_tensor)
                with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as out_f:
                    out_f.write(packed.tobytes())
                
                quant_params["scales"][component_name] = scale
                quant_params["zero_points"][component_name] = zero_point
            
            # Process layers
            for layer_idx in tqdm(range(model_config["num_hidden_layers"]), desc="Processing layers"):
                print(f"\nProcessing layer {layer_idx}...")
                
                layer_name = f"gpt_neox.layers.{layer_idx}"
                
                # Special handling for QKV weights
                if f"{layer_name}.attention.query_key_value.weight" in f.keys():
                    qkv_weight = f.get_tensor(f"{layer_name}.attention.query_key_value.weight")
                    
                    # Split QKV
                    head_dim = h // config.num_attention_heads
                    qkv_weight = qkv_weight.view(h, config.num_attention_heads, 3, head_dim).transpose(1, 2)
                    
                    # Extract Q, K, V
                    q_weight = qkv_weight[:, 0].reshape(h, h)
                    k_weight = qkv_weight[:, 1].reshape(h, h)
                    v_weight = qkv_weight[:, 2].reshape(h, h)
                    
                    # Save Q
                    component_name = f"layer.{layer_idx}.attention.self.query.weight"
                    packed, scale, zero_point = quantize_to_8bit(q_weight)
                    with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as out_f:
                        out_f.write(packed.tobytes())
                    quant_params["scales"][component_name] = scale
                    quant_params["zero_points"][component_name] = zero_point
                    
                    # Save K
                    component_name = f"layer.{layer_idx}.attention.self.key.weight"
                    packed, scale, zero_point = quantize_to_8bit(k_weight)
                    with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as out_f:
                        out_f.write(packed.tobytes())
                    quant_params["scales"][component_name] = scale
                    quant_params["zero_points"][component_name] = zero_point
                    
                    # Save V
                    component_name = f"layer.{layer_idx}.attention.self.value.weight"
                    packed, scale, zero_point = quantize_to_8bit(v_weight)
                    with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as out_f:
                        out_f.write(packed.tobytes())
                    quant_params["scales"][component_name] = scale
                    quant_params["zero_points"][component_name] = zero_point
                
                # Process other layer weights
                for orig_name, mapped_name in name_mapping.items():
                    # Skip QKV (already processed) and embedding layers
                    if "query_key_value" in orig_name or "embed" in orig_name:
                        continue
                    
                    # Only process weights for current layer
                    if not orig_name.startswith(layer_name):
                        continue
                    
                    if orig_name in f.keys():
                        print(f"Processing {orig_name} -> {mapped_name}")
                        tensor = f.get_tensor(orig_name)
                        
                        if tensor.dim() <= 1 or ".bias" in orig_name:
                            # Save biases and 1D tensors directly
                            with open(os.path.join(output_dir, f"{mapped_name}.bin"), "wb") as out_f:
                                out_f.write(tensor.cpu().numpy().astype(np.float32).tobytes())
                            quant_params["scales"][mapped_name] = 1.0
                            quant_params["zero_points"][mapped_name] = 0
                        else:
                            # Quantize matrices
                            packed, scale, zero_point = quantize_to_8bit(tensor)
                            with open(os.path.join(output_dir, f"{mapped_name}.bin"), "wb") as out_f:
                                out_f.write(packed.tobytes())
                            quant_params["scales"][mapped_name] = scale
                            quant_params["zero_points"][mapped_name] = zero_point
                
                # Clear memory
                gc.collect()
    else:
        # Process using PyTorch state dict
        # This is more memory intensive, so we need to be careful
        print("Loading state dict...")
        
        # Load state dict in chunks
        state_dict = torch.load(weights_path, map_location="cpu")
        
        # Process embeddings first
        print("Processing embeddings...")
        if "gpt_neox.embed_in.weight" in state_dict:
            embed_tensor = state_dict["gpt_neox.embed_in.weight"]
            component_name = "embeddings.word_embeddings"
            
            packed, scale, zero_point = quantize_to_8bit(embed_tensor)
            with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
                f.write(packed.tobytes())
            
            quant_params["scales"][component_name] = scale
            quant_params["zero_points"][component_name] = zero_point
            
            # Remove from state dict to free memory
            del state_dict["gpt_neox.embed_in.weight"]
        
        # Process position embeddings
        if "gpt_neox.pos_embed.weight" in state_dict:
            pos_embed_tensor = state_dict["gpt_neox.pos_embed.weight"]
            component_name = "embeddings.position_embeddings"
            
            packed, scale, zero_point = quantize_to_8bit(pos_embed_tensor)
            with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
                f.write(packed.tobytes())
            
            quant_params["scales"][component_name] = scale
            quant_params["zero_points"][component_name] = zero_point
            
            # Remove from state dict to free memory
            del state_dict["gpt_neox.pos_embed.weight"]
        
        # Force garbage collection
        gc.collect()
        
        # Process layers
        for layer_idx in tqdm(range(model_config["num_hidden_layers"]), desc="Processing layers"):
            print(f"\nProcessing layer {layer_idx}...")
            
            layer_name = f"gpt_neox.layers.{layer_idx}"
            
            # Special handling for QKV weights
            qkv_key = f"{layer_name}.attention.query_key_value.weight"
            if qkv_key in state_dict:
                qkv_weight = state_dict[qkv_key]
                
                # Split QKV
                head_dim = h // config.num_attention_heads
                qkv_weight = qkv_weight.view(h, config.num_attention_heads, 3, head_dim).transpose(1, 2)
                
                # Extract Q, K, V
                q_weight = qkv_weight[:, 0].reshape(h, h)
                k_weight = qkv_weight[:, 1].reshape(h, h)
                v_weight = qkv_weight[:, 2].reshape(h, h)
                
                # Save Q
                component_name = f"layer.{layer_idx}.attention.self.query.weight"
                packed, scale, zero_point = quantize_to_8bit(q_weight)
                with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
                    f.write(packed.tobytes())
                quant_params["scales"][component_name] = scale
                quant_params["zero_points"][component_name] = zero_point
                
                # Save K
                component_name = f"layer.{layer_idx}.attention.self.key.weight"
                packed, scale, zero_point = quantize_to_8bit(k_weight)
                with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
                    f.write(packed.tobytes())
                quant_params["scales"][component_name] = scale
                quant_params["zero_points"][component_name] = zero_point
                
                # Save V
                component_name = f"layer.{layer_idx}.attention.self.value.weight"
                packed, scale, zero_point = quantize_to_8bit(v_weight)
                with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
                    f.write(packed.tobytes())
                quant_params["scales"][component_name] = scale
                quant_params["zero_points"][component_name] = zero_point
                
                # Remove from state dict to free memory
                del state_dict[qkv_key]
                del qkv_weight
                del q_weight
                del k_weight
                del v_weight
                gc.collect()
            
            # Process other layer weights
            for orig_name, mapped_name in name_mapping.items():
                # Skip QKV (already processed) and embedding layers
                if "query_key_value" in orig_name or "embed" in orig_name:
                    continue
                
                # Only process weights for current layer
                if not orig_name.startswith(layer_name):
                    continue
                
                if orig_name in state_dict:
                    print(f"Processing {orig_name} -> {mapped_name}")
                    tensor = state_dict[orig_name]
                    
                    if tensor.dim() <= 1 or ".bias" in orig_name:
                        # Save biases and 1D tensors directly
                        with open(os.path.join(output_dir, f"{mapped_name}.bin"), "wb") as f:
                            f.write(tensor.cpu().numpy().astype(np.float32).tobytes())
                        quant_params["scales"][mapped_name] = 1.0
                        quant_params["zero_points"][mapped_name] = 0
                    else:
                        # Quantize matrices
                        packed, scale, zero_point = quantize_to_8bit(tensor)
                        with open(os.path.join(output_dir, f"{mapped_name}.bin"), "wb") as f:
                            f.write(packed.tobytes())
                        quant_params["scales"][mapped_name] = scale
                        quant_params["zero_points"][mapped_name] = zero_point
                    
                    # Remove from state dict to free memory
                    del state_dict[orig_name]
            
            # Clear memory
            gc.collect()
    
    # Save quantization parameters
    with open(os.path.join(output_dir, "quant_params.json"), "w") as f:
        json.dump(quant_params, f)
    
    total_time = time.time() - start_time
    print(f"Model conversion complete in {total_time:.2f} seconds")
    print(f"Quantized model saved to {output_dir}")

if __name__ == "__main__":
    # Convert the actual pretrained model
    # You can change to pythia-410m or pythia-1b for a larger model
    model_name = "EleutherAI/pythia-410m"
    output_dir = "quantized_model"
    
    # Install safetensors if needed
    try:
        import safetensors
    except ImportError:
        print("Installing safetensors...")
        import subprocess
        subprocess.check_call(["pip", "install", "safetensors"])
    
    convert_model_optimized(model_name, output_dir)

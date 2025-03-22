import os
import json
import numpy as np
import torch

def create_missing_layernorm_weights(model_dir):
    """Generate missing LayerNorm weights based on typical values"""
    # Load config to get model dimensions
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    
    # Load quantization parameters
    quant_params_path = os.path.join(model_dir, "quant_params.json")
    if os.path.exists(quant_params_path):
        with open(quant_params_path, "r") as f:
            quant_params = json.load(f)
    else:
        quant_params = {"scales": {}, "zero_points": {}}
    
    # Create missing LayerNorm and bias tensors for each layer
    for layer_idx in range(num_layers):
        # Create attention layer norm weights
        create_layernorm_weights(
            model_dir, 
            f"layer.{layer_idx}.attention.output.LayerNorm", 
            hidden_size,
            quant_params
        )
        
        # Create output layer norm weights
        create_layernorm_weights(
            model_dir, 
            f"layer.{layer_idx}.output.LayerNorm", 
            hidden_size,
            quant_params
        )
        
        # Create missing bias vectors
        create_bias_vector(
            model_dir,
            f"layer.{layer_idx}.intermediate.dense.bias",
            4 * hidden_size,  # Intermediate size is typically 4x hidden size
            quant_params
        )
        
        create_bias_vector(
            model_dir,
            f"layer.{layer_idx}.output.dense.bias",
            hidden_size,
            quant_params
        )
    
    # Save updated quantization parameters
    with open(quant_params_path, "w") as f:
        json.dump(quant_params, f, indent=2)
    
    print(f"Created missing weights for {num_layers} layers with hidden size {hidden_size}")

def create_layernorm_weights(model_dir, name_prefix, size, quant_params):
    """Create LayerNorm weight and bias if missing"""
    weight_path = os.path.join(model_dir, f"{name_prefix}.weight.bin")
    bias_path = os.path.join(model_dir, f"{name_prefix}.bias.bin")
    
    # Create weight if missing (typically ones)
    if not os.path.exists(weight_path):
        print(f"Creating missing {name_prefix}.weight")
        weights = np.ones(size, dtype=np.float32)
        with open(weight_path, "wb") as f:
            f.write(weights.tobytes())
        
        # Add to quant params
        quant_params["scales"][f"{name_prefix}.weight"] = 1.0
        quant_params["zero_points"][f"{name_prefix}.weight"] = 0
    
    # Create bias if missing (typically zeros)
    if not os.path.exists(bias_path):
        print(f"Creating missing {name_prefix}.bias")
        bias = np.zeros(size, dtype=np.float32)
        with open(bias_path, "wb") as f:
            f.write(bias.tobytes())
        
        # Add to quant params
        quant_params["scales"][f"{name_prefix}.bias"] = 1.0
        quant_params["zero_points"][f"{name_prefix}.bias"] = 0

def create_bias_vector(model_dir, name, size, quant_params):
    """Create a bias vector if missing"""
    bias_path = os.path.join(model_dir, f"{name}.bin")
    
    if not os.path.exists(bias_path):
        print(f"Creating missing {name}")
        bias = np.zeros(size, dtype=np.float32)
        with open(bias_path, "wb") as f:
            f.write(bias.tobytes())
        
        # Add to quant params
        quant_params["scales"][name] = 1.0
        quant_params["zero_points"][name] = 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix missing weights in quantized model")
    parser.add_argument("--model_dir", type=str, default="quantized_model", 
                        help="Directory containing quantized model files")
    args = parser.parse_args()
    
    create_missing_layernorm_weights(args.model_dir)
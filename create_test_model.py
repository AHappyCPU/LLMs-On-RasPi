# Update the create_test_model.py file
import os
import json
import numpy as np
import torch

def create_test_model(output_dir, hidden_size=64, vocab_size=1000, num_layers=2):
    """Create a minimal test model with random weights"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create config
    config = {
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "intermediate_size": hidden_size * 4,
        "max_position_embeddings": 128,
        "layer_norm_eps": 1e-12
    }
    
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f)
    
    # Create vocabulary
    vocab = {f"token_{i}": i for i in range(vocab_size)}
    vocab["<unk>"] = 0
    vocab["In"] = 1
    vocab["a"] = 2  
    vocab["world"] = 3
    vocab["where"] = 4
    vocab["technology"] = 5
    
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    
    # Create quantization parameters
    quant_params = {"scales": {}, "zero_points": {}}
    
    # Create embeddings
    embeddings = torch.randn(vocab_size, hidden_size)
    component_name = "embeddings.word_embeddings"
    
    # Quantize to 4-bit
    quantized = torch.clamp(torch.round(embeddings * 2 + 8), 0, 15).to(torch.uint8)
    packed = torch.zeros(vocab_size, hidden_size // 2, dtype=torch.uint8)
    
    for i in range(0, hidden_size, 2):
        if i+1 < hidden_size:
            packed[:, i//2] = (quantized[:, i] << 4) | quantized[:, i+1]
    
    with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
        f.write(packed.numpy().tobytes())
    
    quant_params["scales"][component_name] = 0.5
    quant_params["zero_points"][component_name] = 8
    
    # Create position embeddings
    pos_embeddings = torch.randn(config["max_position_embeddings"], hidden_size)
    component_name = "embeddings.position_embeddings"
    
    quantized = torch.clamp(torch.round(pos_embeddings * 2 + 8), 0, 15).to(torch.uint8)
    packed = torch.zeros(config["max_position_embeddings"], hidden_size // 2, dtype=torch.uint8)
    
    for i in range(0, hidden_size, 2):
        if i+1 < hidden_size:
            packed[:, i//2] = (quantized[:, i] << 4) | quantized[:, i+1]
    
    with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
        f.write(packed.numpy().tobytes())
    
    quant_params["scales"][component_name] = 0.5
    quant_params["zero_points"][component_name] = 8
    
    # Create layer weights
    for layer_idx in range(num_layers):
        # Fixed: Make sure dimensions match correctly
        components = {
            f"layer.{layer_idx}.attention.self.query.weight": (hidden_size, hidden_size),
            f"layer.{layer_idx}.attention.self.key.weight": (hidden_size, hidden_size),
            f"layer.{layer_idx}.attention.self.value.weight": (hidden_size, hidden_size),
            f"layer.{layer_idx}.attention.output.dense.weight": (hidden_size, hidden_size),
            f"layer.{layer_idx}.attention.output.LayerNorm.weight": (hidden_size,),
            f"layer.{layer_idx}.attention.output.LayerNorm.bias": (hidden_size,),
            f"layer.{layer_idx}.intermediate.dense.weight": (hidden_size * 4, hidden_size),
            f"layer.{layer_idx}.intermediate.dense.bias": (hidden_size * 4,),
            f"layer.{layer_idx}.output.dense.weight": (hidden_size, hidden_size * 4),
            f"layer.{layer_idx}.output.dense.bias": (hidden_size,),
            f"layer.{layer_idx}.output.LayerNorm.weight": (hidden_size,),
            f"layer.{layer_idx}.output.LayerNorm.bias": (hidden_size,),
        }
        
        for component_name, shape in components.items():
            tensor = torch.randn(*shape)
            
            if len(shape) == 1:
                # Bias terms, save as float32
                with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
                    f.write(tensor.numpy().astype(np.float32).tobytes())
                quant_params["scales"][component_name] = 1.0
                quant_params["zero_points"][component_name] = 0
            else:
                # Weight matrices, quantize to 4-bit
                quantized = torch.clamp(torch.round(tensor * 2 + 8), 0, 15).to(torch.uint8)
                
                if len(shape) == 2:
                    # Handle 2D matrices
                    rows, cols = shape
                    packed = torch.zeros(rows, (cols + 1) // 2, dtype=torch.uint8)
                    
                    for i in range(0, cols, 2):
                        if i+1 < cols:
                            packed[:, i//2] = (quantized[:, i] << 4) | quantized[:, i+1]
                        else:
                            packed[:, i//2] = quantized[:, i] << 4
                    
                    with open(os.path.join(output_dir, f"{component_name}.bin"), "wb") as f:
                        f.write(packed.numpy().tobytes())
                    
                quant_params["scales"][component_name] = 0.5
                quant_params["zero_points"][component_name] = 8
    
    # Save quantization parameters
    with open(os.path.join(output_dir, "quant_params.json"), "w") as f:
        json.dump(quant_params, f)
    
    print(f"Created test model in {output_dir}")

# Create a small test model
create_test_model("quantized_model") 

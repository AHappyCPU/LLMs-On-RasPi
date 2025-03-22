from transformers import AutoTokenizer
import json
import os

def extract_and_save_vocabulary(model_name, output_dir):
    """Extract vocabulary from a model and save it in a format compatible with RaspberryLLM"""
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get the vocabulary as a dictionary
    vocab = tokenizer.get_vocab()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save vocabulary as JSON
    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"Vocabulary saved to {vocab_path} with {len(vocab)} tokens")
    
    # Create a simple tokenizer test file to verify correct operation
    token_test_path = os.path.join(output_dir, "token_test.py")
    with open(token_test_path, 'w', encoding='utf-8') as f:
        f.write("""
import json

# Load vocabulary
with open('vocab.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# Create reverse mapping
id_to_token = {int(v): k for k, v in vocab.items()}

# Test phrase
phrase = "In a world where technology"
print(f"Test phrase: '{phrase}'")

# Simple tokenization (just for testing)
tokens = phrase.split()
print(f"Tokens: {tokens}")

# Find token IDs
token_ids = []
for token in tokens:
    if token in vocab:
        token_ids.append(vocab[token])
    else:
        # Try with space prefix (common in some tokenizers)
        space_token = " " + token
        if space_token in vocab:
            token_ids.append(vocab[space_token])
        else:
            print(f"Unknown token: '{token}'")
            token_ids.append(0)  # Use 0 or another default

print(f"Token IDs: {token_ids}")

# Convert back to tokens
reconstructed = [id_to_token.get(id, "[UNK]") for id in token_ids]
print(f"Reconstructed: {reconstructed}")
""")
    
    print(f"Created tokenizer test script at {token_test_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and save vocabulary for RaspberryLLM")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-410m", 
                        help="HuggingFace model name")
    parser.add_argument("--output", type=str, default="quantized_model", 
                        help="Output directory")
    
    args = parser.parse_args()
    extract_and_save_vocabulary(args.model, args.output)
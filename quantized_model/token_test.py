
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

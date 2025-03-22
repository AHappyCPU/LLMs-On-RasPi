import re
import json
import torch
from typing import List, Dict, Tuple

class NeoXTokenizer:
    """Simple tokenizer implementation for NeoX/Pythia models"""
    
    def __init__(self, vocab_path: str):
        self.load_vocabulary(vocab_path)
        self.setup_special_tokens()
        
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
        
        # Identify tokens that start with space
        self.space_tokens = {token for token in self.vocab.keys() if token.startswith(" ")}
        
        # Find the basic space token if it exists
        self.space_token = " " if " " in self.vocab else None
        if self.space_token:
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
                    space_token = " " + word[0:0]  # Empty string with space prefix
                    if space_token in self.vocab:
                        token_ids.append(self.vocab[space_token])
            
            # Add word - first try as is
            if word in self.vocab:
                token_ids.append(self.vocab[word])
                continue
                
            # Try with space prefix for first word
            if i == 0 and (" " + word) in self.vocab:
                token_ids.append(self.vocab[" " + word])
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
        """Convert token IDs back to text"""
        if not token_ids or len(token_ids) == 0:
            return ""
        
        text = ""
        for i, token_id in enumerate(token_ids):
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Handle special tokens
                if token == "<|endoftext|>" or token == "</s>" or token == "<eos>":
                    break
                    
                # Handle space tokens properly
                if token.startswith(" "):
                    text += token
                else:
                    # Add a space if it's not the first token and doesn't start with space
                    if i > 0 and not text.endswith(" ") and not token.startswith(" "):
                        text += " "
                    text += token
            else:
                text += " [UNK] "
        
        return text.strip()
    
    def get_vocab_size(self) -> int:
        """Get the size of the vocabulary"""
        return len(self.vocab)


# Test function to verify tokenizer operation
def test_tokenizer(vocab_path: str, test_text: str = "In a world where technology"):
    """Test the tokenizer with a sample text"""
    tokenizer = NeoXTokenizer(vocab_path)
    
    print(f"\nTokenizer test with text: '{test_text}'")
    token_ids = tokenizer.encode(test_text)
    print(f"Encoded token IDs: {token_ids[0].tolist()}")
    
    decoded_text = tokenizer.decode(token_ids[0].tolist())
    print(f"Decoded text: '{decoded_text}'")
    
    return tokenizer

if __name__ == "__main__":
    import sys
    
    vocab_path = "quantized_model/vocab.json" if len(sys.argv) < 2 else sys.argv[1]
    test_tokenizer(vocab_path)
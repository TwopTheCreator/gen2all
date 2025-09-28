"""
Gen2All Tokenization System
Advanced tokenization with multiple algorithms and customizable vocabularies
"""
import re
import json
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Set, Optional, Union, Tuple
from abc import ABC, abstractmethod
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenizerInterface(ABC):
    """Abstract interface for tokenizers"""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        pass
    
    @abstractmethod
    def save_vocab(self, filepath: str):
        """Save vocabulary to file"""
        pass
    
    @abstractmethod
    def load_vocab(self, filepath: str):
        """Load vocabulary from file"""
        pass

class WordTokenizer(TokenizerInterface):
    """Word-based tokenizer"""
    
    def __init__(self, vocab_size: int = None, case_sensitive: bool = None):
        self.vocab_size = vocab_size or config.tokenizer.vocab_size
        self.case_sensitive = case_sensitive if case_sensitive is not None else config.tokenizer.case_sensitive
        
        # Special tokens
        self.special_tokens = config.tokenizer.special_tokens.copy()
        
        # Vocabularies
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        
        # Initialize with special tokens
        self._initialize_special_tokens()
        
        logger.info(f"Initialized WordTokenizer with vocab_size={self.vocab_size}")
    
    def _initialize_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        for token in self.special_tokens.values():
            if token not in self.word_to_id:
                token_id = len(self.word_to_id)
                self.word_to_id[token] = token_id
                self.id_to_word[token_id] = token
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before tokenization"""
        if not self.case_sensitive:
            text = text.lower()
        
        # Clean up whitespace and punctuation
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text"""
        # Simple word extraction - can be enhanced
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        return words
    
    def build_vocab(self, texts: List[str], min_frequency: int = 1):
        """Build vocabulary from training texts"""
        logger.info("Building vocabulary from training texts...")
        
        # Count word frequencies
        word_counts = Counter()
        
        for text in texts:
            processed_text = self._preprocess_text(text)
            words = self._extract_words(processed_text)
            word_counts.update(words)
        
        # Select most frequent words
        available_slots = self.vocab_size - len(self.special_tokens)
        most_common_words = word_counts.most_common(available_slots)
        
        # Add words to vocabulary
        for word, count in most_common_words:
            if count >= min_frequency and word not in self.word_to_id:
                token_id = len(self.word_to_id)
                self.word_to_id[word] = token_id
                self.id_to_word[token_id] = word
        
        logger.info(f"Built vocabulary with {len(self.word_to_id)} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        processed_text = self._preprocess_text(text)
        words = self._extract_words(processed_text)
        
        token_ids = []
        for word in words:
            if word in self.word_to_id:
                token_ids.append(self.word_to_id[word])
            else:
                # Use unknown token
                unk_token = self.special_tokens.get("unk", "[UNK]")
                if unk_token in self.word_to_id:
                    token_ids.append(self.word_to_id[unk_token])
        
        # Limit to max_tokens
        max_tokens = config.tokenizer.max_tokens
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                word = self.id_to_word[token_id]
                # Skip special tokens in decoding
                if word not in self.special_tokens.values():
                    words.append(word)
        
        return ' '.join(words)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.word_to_id)
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to file"""
        vocab_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': {str(k): v for k, v in self.id_to_word.items()},
            'special_tokens': self.special_tokens,
            'config': {
                'vocab_size': self.vocab_size,
                'case_sensitive': self.case_sensitive
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved vocabulary to {filepath}")
    
    def load_vocab(self, filepath: str):
        """Load vocabulary from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word_to_id = vocab_data['word_to_id']
        self.id_to_word = {int(k): v for k, v in vocab_data['id_to_word'].items()}
        self.special_tokens = vocab_data['special_tokens']
        
        # Update config
        vocab_config = vocab_data.get('config', {})
        self.vocab_size = vocab_config.get('vocab_size', self.vocab_size)
        self.case_sensitive = vocab_config.get('case_sensitive', self.case_sensitive)
        
        logger.info(f"Loaded vocabulary from {filepath}")

class CharacterTokenizer(TokenizerInterface):
    """Character-based tokenizer"""
    
    def __init__(self, vocab_size: int = None):
        self.vocab_size = vocab_size or config.tokenizer.vocab_size
        
        # Special tokens
        self.special_tokens = config.tokenizer.special_tokens.copy()
        
        # Vocabularies
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        
        # Initialize with special tokens
        self._initialize_special_tokens()
        
        logger.info(f"Initialized CharacterTokenizer with vocab_size={self.vocab_size}")
    
    def _initialize_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        for token in self.special_tokens.values():
            if token not in self.char_to_id:
                token_id = len(self.char_to_id)
                self.char_to_id[token] = token_id
                self.id_to_char[token_id] = token
    
    def build_vocab(self, texts: List[str]):
        """Build character vocabulary from training texts"""
        logger.info("Building character vocabulary from training texts...")
        
        # Collect all unique characters
        all_chars = set()
        for text in texts:
            all_chars.update(text)
        
        # Sort characters for consistent ordering
        sorted_chars = sorted(all_chars)
        
        # Add characters to vocabulary (respecting vocab_size limit)
        available_slots = self.vocab_size - len(self.special_tokens)
        
        for char in sorted_chars[:available_slots]:
            if char not in self.char_to_id:
                token_id = len(self.char_to_id)
                self.char_to_id[char] = token_id
                self.id_to_char[token_id] = char
        
        logger.info(f"Built character vocabulary with {len(self.char_to_id)} tokens")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        token_ids = []
        
        for char in text:
            if char in self.char_to_id:
                token_ids.append(self.char_to_id[char])
            else:
                # Use unknown token
                unk_token = self.special_tokens.get("unk", "[UNK]")
                if unk_token in self.char_to_id:
                    token_ids.append(self.char_to_id[unk_token])
        
        # Limit to max_tokens
        max_tokens = config.tokenizer.max_tokens
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        chars = []
        for token_id in token_ids:
            if token_id in self.id_to_char:
                char = self.id_to_char[token_id]
                # Skip special tokens in decoding
                if char not in self.special_tokens.values():
                    chars.append(char)
        
        return ''.join(chars)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.char_to_id)
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to file"""
        vocab_data = {
            'char_to_id': self.char_to_id,
            'id_to_char': {str(k): v for k, v in self.id_to_char.items()},
            'special_tokens': self.special_tokens,
            'config': {
                'vocab_size': self.vocab_size
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved character vocabulary to {filepath}")
    
    def load_vocab(self, filepath: str):
        """Load vocabulary from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.char_to_id = vocab_data['char_to_id']
        self.id_to_char = {int(k): v for k, v in vocab_data['id_to_char'].items()}
        self.special_tokens = vocab_data['special_tokens']
        
        # Update config
        vocab_config = vocab_data.get('config', {})
        self.vocab_size = vocab_config.get('vocab_size', self.vocab_size)
        
        logger.info(f"Loaded character vocabulary from {filepath}")

class BPETokenizer(TokenizerInterface):
    """Byte Pair Encoding (BPE) tokenizer - simplified implementation"""
    
    def __init__(self, vocab_size: int = None):
        self.vocab_size = vocab_size or config.tokenizer.vocab_size
        
        # Special tokens
        self.special_tokens = config.tokenizer.special_tokens.copy()
        
        # BPE components
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        
        # Initialize with special tokens
        self._initialize_special_tokens()
        
        logger.info(f"Initialized BPETokenizer with vocab_size={self.vocab_size}")
    
    def _initialize_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        for token in self.special_tokens.values():
            if token not in self.word_to_id:
                token_id = len(self.word_to_id)
                self.word_to_id[token] = token_id
                self.id_to_word[token_id] = token
    
    def _get_word_tokens(self, word: str) -> List[str]:
        """Get tokens for a word using BPE"""
        if not word:
            return []
        
        # Start with characters
        tokens = list(word)
        
        while True:
            # Find the highest-ranked pair to merge
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            if not pairs:
                break
            
            best_pair = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            if best_pair not in self.bpe_ranks:
                break
            
            # Merge the best pair
            first, second = best_pair
            new_tokens = []
            i = 0
            
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(first + second)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            
            tokens = new_tokens
        
        return tokens
    
    def build_vocab(self, texts: List[str], min_frequency: int = 2):
        """Build BPE vocabulary from training texts"""
        logger.info("Building BPE vocabulary from training texts...")
        
        # Extract words and their frequencies
        word_freq = Counter()
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            word_freq.update(words)
        
        # Initialize with single characters
        all_chars = set()
        for word in word_freq:
            all_chars.update(word)
        
        for char in sorted(all_chars):
            if char not in self.word_to_id:
                token_id = len(self.word_to_id)
                self.word_to_id[char] = token_id
                self.id_to_word[token_id] = char
        
        # Build BPE merges
        num_merges = self.vocab_size - len(self.word_to_id)
        
        for merge_idx in range(num_merges):
            # Count all adjacent pairs
            pair_counts = defaultdict(int)
            
            for word, freq in word_freq.items():
                tokens = self._get_word_tokens(word)
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += freq
            
            if not pair_counts:
                break
            
            # Find the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            
            if pair_counts[best_pair] < min_frequency:
                break
            
            # Add merge to BPE ranks
            self.bpe_ranks[best_pair] = merge_idx
            
            # Add merged token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.word_to_id:
                token_id = len(self.word_to_id)
                self.word_to_id[merged_token] = token_id
                self.id_to_word[token_id] = merged_token
        
        logger.info(f"Built BPE vocabulary with {len(self.word_to_id)} tokens and {len(self.bpe_ranks)} merges")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        words = re.findall(r'\b\w+\b', text.lower())
        token_ids = []
        
        for word in words:
            tokens = self._get_word_tokens(word)
            for token in tokens:
                if token in self.word_to_id:
                    token_ids.append(self.word_to_id[token])
                else:
                    # Use unknown token
                    unk_token = self.special_tokens.get("unk", "[UNK]")
                    if unk_token in self.word_to_id:
                        token_ids.append(self.word_to_id[unk_token])
        
        # Limit to max_tokens
        max_tokens = config.tokenizer.max_tokens
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                token = self.id_to_word[token_id]
                if token not in self.special_tokens.values():
                    tokens.append(token)
        
        # Simple reconstruction - can be improved
        return ''.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.word_to_id)
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to file"""
        vocab_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': {str(k): v for k, v in self.id_to_word.items()},
            'bpe_ranks': {f"{k[0]}||{k[1]}": v for k, v in self.bpe_ranks.items()},
            'special_tokens': self.special_tokens,
            'config': {
                'vocab_size': self.vocab_size
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved BPE vocabulary to {filepath}")
    
    def load_vocab(self, filepath: str):
        """Load vocabulary from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word_to_id = vocab_data['word_to_id']
        self.id_to_word = {int(k): v for k, v in vocab_data['id_to_word'].items()}
        
        # Reconstruct BPE ranks
        self.bpe_ranks = {}
        for pair_str, rank in vocab_data['bpe_ranks'].items():
            first, second = pair_str.split('||')
            self.bpe_ranks[(first, second)] = rank
        
        self.special_tokens = vocab_data['special_tokens']
        
        # Update config
        vocab_config = vocab_data.get('config', {})
        self.vocab_size = vocab_config.get('vocab_size', self.vocab_size)
        
        logger.info(f"Loaded BPE vocabulary from {filepath}")

class TokenizerManager:
    """High-level tokenizer management"""
    
    def __init__(self):
        self.tokenizers: Dict[str, TokenizerInterface] = {}
        self.current_tokenizer: Optional[TokenizerInterface] = None
    
    def create_tokenizer(self, name: str, tokenizer_type: str, **kwargs) -> TokenizerInterface:
        """Create and register a new tokenizer"""
        tokenizer_type = tokenizer_type.lower()
        
        if tokenizer_type == "word":
            tokenizer = WordTokenizer(**kwargs)
        elif tokenizer_type == "character":
            tokenizer = CharacterTokenizer(**kwargs)
        elif tokenizer_type == "bpe":
            tokenizer = BPETokenizer(**kwargs)
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
        
        self.tokenizers[name] = tokenizer
        
        if self.current_tokenizer is None:
            self.current_tokenizer = tokenizer
        
        logger.info(f"Created {tokenizer_type} tokenizer: {name}")
        return tokenizer
    
    def get_tokenizer(self, name: str) -> Optional[TokenizerInterface]:
        """Get tokenizer by name"""
        return self.tokenizers.get(name)
    
    def set_current_tokenizer(self, name: str):
        """Set current active tokenizer"""
        if name in self.tokenizers:
            self.current_tokenizer = self.tokenizers[name]
            logger.info(f"Set current tokenizer to: {name}")
        else:
            raise ValueError(f"Tokenizer not found: {name}")
    
    def list_tokenizers(self) -> List[str]:
        """List all available tokenizer names"""
        return list(self.tokenizers.keys())
    
    def train_tokenizer(self, name: str, training_texts: List[str], **kwargs):
        """Train a tokenizer on provided texts"""
        if name not in self.tokenizers:
            raise ValueError(f"Tokenizer not found: {name}")
        
        tokenizer = self.tokenizers[name]
        tokenizer.build_vocab(training_texts, **kwargs)
        
        logger.info(f"Trained tokenizer: {name}")
    
    def encode_text(self, text: str, tokenizer_name: str = None) -> List[int]:
        """Encode text using specified or current tokenizer"""
        tokenizer = self.current_tokenizer
        
        if tokenizer_name:
            tokenizer = self.get_tokenizer(tokenizer_name)
            if not tokenizer:
                raise ValueError(f"Tokenizer not found: {tokenizer_name}")
        
        if not tokenizer:
            raise ValueError("No tokenizer available")
        
        return tokenizer.encode(text)
    
    def decode_tokens(self, token_ids: List[int], tokenizer_name: str = None) -> str:
        """Decode tokens using specified or current tokenizer"""
        tokenizer = self.current_tokenizer
        
        if tokenizer_name:
            tokenizer = self.get_tokenizer(tokenizer_name)
            if not tokenizer:
                raise ValueError(f"Tokenizer not found: {tokenizer_name}")
        
        if not tokenizer:
            raise ValueError("No tokenizer available")
        
        return tokenizer.decode(token_ids)

# Global tokenizer manager instance
tokenizer_manager = TokenizerManager()
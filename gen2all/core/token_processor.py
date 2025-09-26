import re
import json
import numpy as np
import threading
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import pickle
import lz4.frame
import xxhash
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


class BytePairEncoder:
    def __init__(self, vocab_size: int = 65536):
        self.vocab_size = vocab_size
        self.encoder = {}
        self.decoder = {}
        self.bpe_merges = []
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
            '<mask>': 4,
            '<sep>': 5,
            '<cls>': 6,
            '<user>': 7,
            '<assistant>': 8,
            '<system>': 9
        }
        self.cache = {}
        self.cache_lock = threading.RLock()
        
    def train(self, texts: List[str], min_frequency: int = 2):
        word_counts = Counter()
        
        for text in texts:
            words = self._pre_tokenize(text)
            for word in words:
                word_counts[word] += 1
        
        vocab = set()
        for word, count in word_counts.items():
            if count >= min_frequency:
                for char in word:
                    vocab.add(char)
        
        vocab = sorted(list(vocab))
        
        self.encoder = dict(self.special_tokens)
        for i, token in enumerate(vocab):
            self.encoder[token] = len(self.encoder)
        
        word_splits = {}
        for word in word_counts:
            word_splits[word] = list(word)
        
        merges = []
        while len(self.encoder) < self.vocab_size:
            pairs = defaultdict(int)
            
            for word, word_tokens in word_splits.items():
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i + 1])
                    pairs[pair] += word_counts[word]
            
            if not pairs:
                break
                
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < min_frequency:
                break
            
            merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.encoder[new_token] = len(self.encoder)
            
            new_word_splits = {}
            for word in word_splits:
                new_word = self._merge_word(word_splits[word], best_pair)
                new_word_splits[word] = new_word
            word_splits = new_word_splits
        
        self.bpe_merges = merges
        self.decoder = {v: k for k, v in self.encoder.items()}
        
    def _pre_tokenize(self, text: str) -> List[str]:
        pattern = re.compile(r'\w+|\S')
        tokens = pattern.findall(text.lower())
        return tokens
    
    def _merge_word(self, word: List[str], pair: Tuple[str, str]) -> List[str]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return new_word
    
    def encode(self, text: str) -> List[int]:
        with self.cache_lock:
            text_hash = xxhash.xxh64(text).hexdigest()
            if text_hash in self.cache:
                return self.cache[text_hash]
        
        words = self._pre_tokenize(text)
        encoded_tokens = []
        
        for word in words:
            word_tokens = list(word)
            
            for pair in self.bpe_merges:
                word_tokens = self._merge_word(word_tokens, pair)
            
            for token in word_tokens:
                if token in self.encoder:
                    encoded_tokens.append(self.encoder[token])
                else:
                    encoded_tokens.append(self.encoder['<unk>'])
        
        with self.cache_lock:
            if len(self.cache) > 10000:
                self.cache.clear()
            self.cache[text_hash] = encoded_tokens
        
        return encoded_tokens
    
    def decode(self, tokens: List[int]) -> str:
        decoded_tokens = []
        for token in tokens:
            if token in self.decoder:
                decoded_tokens.append(self.decoder[token])
            else:
                decoded_tokens.append('<unk>')
        
        return ''.join(decoded_tokens).replace('</w>', ' ').strip()
    
    def save(self, path: str):
        data = {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'bpe_merges': self.bpe_merges,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.encoder = data['encoder']
        self.decoder = data['decoder']
        self.bpe_merges = data['bpe_merges']
        self.special_tokens = data['special_tokens']
        self.vocab_size = data['vocab_size']


class AdvancedTokenizer:
    def __init__(self, vocab_size: int = 65536, max_length: int = 8192):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.bpe = BytePairEncoder(vocab_size)
        self.conversation_templates = {
            'chat': {
                'system_prefix': '<system>',
                'user_prefix': '<user>',
                'assistant_prefix': '<assistant>',
                'separator': '<sep>',
                'end_token': '<eos>'
            },
            'instruct': {
                'instruction_prefix': '### Instruction:\n',
                'response_prefix': '### Response:\n',
                'separator': '\n\n',
                'end_token': '<eos>'
            }
        }
        
    def train_tokenizer(self, training_texts: List[str]):
        self.bpe.train(training_texts)
        
    def tokenize(self, text: str, add_special_tokens: bool = True, 
                truncation: bool = True, padding: bool = False) -> Dict[str, Any]:
        
        tokens = self.bpe.encode(text)
        
        if add_special_tokens:
            tokens = [self.bpe.encoder['<bos>']] + tokens + [self.bpe.encoder['<eos>']]
        
        if truncation and len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
            tokens[-1] = self.bpe.encoder['<eos>']
        
        attention_mask = [1] * len(tokens)
        
        if padding and len(tokens) < self.max_length:
            pad_length = self.max_length - len(tokens)
            tokens.extend([self.bpe.encoder['<pad>']] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask,
            'length': len([t for t in tokens if t != self.bpe.encoder['<pad>']])
        }
    
    def detokenize(self, tokens: List[int]) -> str:
        filtered_tokens = [t for t in tokens if t not in [
            self.bpe.encoder['<bos>'],
            self.bpe.encoder['<eos>'],
            self.bpe.encoder['<pad>']
        ]]
        return self.bpe.decode(filtered_tokens)
    
    def format_conversation(self, messages: List[Dict[str, str]], 
                           template: str = 'chat') -> str:
        if template not in self.conversation_templates:
            raise ValueError(f"Unknown template: {template}")
        
        template_config = self.conversation_templates[template]
        formatted_text = ""
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                formatted_text += template_config.get('system_prefix', '') + content
            elif role == 'user':
                formatted_text += template_config.get('user_prefix', '') + content
            elif role == 'assistant':
                formatted_text += template_config.get('assistant_prefix', '') + content
            
            formatted_text += template_config.get('separator', '\n')
        
        return formatted_text.strip()
    
    def batch_tokenize(self, texts: List[str], batch_size: int = 32, 
                      num_workers: int = None) -> List[Dict[str, Any]]:
        if num_workers is None:
            num_workers = min(len(texts), mp.cpu_count())
        
        results = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                future = executor.submit(self._process_batch, batch)
                futures.append(future)
            
            for future in futures:
                results.extend(future.result())
        
        return results
    
    def _process_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.tokenize(text) for text in texts]
    
    def analyze_tokenization(self, text: str) -> Dict[str, Any]:
        tokens = self.bpe.encode(text)
        token_strings = [self.bpe.decoder.get(t, '<unk>') for t in tokens]
        
        analysis = {
            'original_text': text,
            'token_count': len(tokens),
            'tokens': tokens,
            'token_strings': token_strings,
            'compression_ratio': len(text) / len(tokens) if tokens else 0,
            'vocabulary_coverage': len(set(tokens)) / len(tokens) if tokens else 0,
            'oov_tokens': sum(1 for t in tokens if t == self.bpe.encoder['<unk>'])
        }
        
        return analysis


class TokenProcessor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.tokenizer = AdvancedTokenizer(
            self.config['vocab_size'],
            self.config['max_length']
        )
        self.preprocessing_rules = self._load_preprocessing_rules()
        self.postprocessing_rules = self._load_postprocessing_rules()
        
        self.processing_stats = defaultdict(int)
        self.lock = threading.RLock()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'vocab_size': 65536,
            'max_length': 8192,
            'enable_preprocessing': True,
            'enable_postprocessing': True,
            'batch_size': 32,
            'num_workers': mp.cpu_count(),
            'cache_size': 10000
        }
    
    def _load_preprocessing_rules(self) -> List[Tuple[str, str]]:
        return [
            (r'\s+', ' '),
            (r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]'),
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]'),
            (r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]'),
            (r'\b\d{2}:\d{2}(?::\d{2})?\b', '[TIME]'),
            (r'\$\d+(?:\.\d{2})?\b', '[MONEY]'),
        ]
    
    def _load_postprocessing_rules(self) -> List[Tuple[str, str]]:
        return [
            (r'\[URL\]', 'URL'),
            (r'\[EMAIL\]', 'email address'),
            (r'\[DATE\]', 'date'),
            (r'\[TIME\]', 'time'),
            (r'\[MONEY\]', 'amount'),
            (r'\s+', ' '),
        ]
    
    def preprocess_text(self, text: str) -> str:
        if not self.config['enable_preprocessing']:
            return text
        
        processed = text
        for pattern, replacement in self.preprocessing_rules:
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
        
        processed = processed.strip()
        return processed
    
    def postprocess_text(self, text: str) -> str:
        if not self.config['enable_postprocessing']:
            return text
        
        processed = text
        for pattern, replacement in self.postprocessing_rules:
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)
        
        processed = processed.strip()
        return processed
    
    def train_tokenizer(self, training_data: List[str], save_path: Optional[str] = None):
        preprocessed_data = [self.preprocess_text(text) for text in training_data]
        
        self.tokenizer.train_tokenizer(preprocessed_data)
        
        if save_path:
            self.tokenizer.bpe.save(save_path)
        
        with self.lock:
            self.processing_stats['tokenizer_trained'] += 1
    
    def load_tokenizer(self, path: str):
        self.tokenizer.bpe.load(path)
        
        with self.lock:
            self.processing_stats['tokenizer_loaded'] += 1
    
    def process_text(self, text: str, add_special_tokens: bool = True,
                    return_tensors: bool = False) -> Dict[str, Any]:
        
        preprocessed = self.preprocess_text(text)
        tokenized = self.tokenizer.tokenize(
            preprocessed,
            add_special_tokens=add_special_tokens,
            truncation=True,
            padding=False
        )
        
        if return_tensors:
            tokenized['input_ids'] = np.array(tokenized['input_ids'])
            tokenized['attention_mask'] = np.array(tokenized['attention_mask'])
        
        with self.lock:
            self.processing_stats['texts_processed'] += 1
        
        return tokenized
    
    def process_conversation(self, messages: List[Dict[str, str]], 
                           template: str = 'chat') -> Dict[str, Any]:
        formatted_text = self.tokenizer.format_conversation(messages, template)
        return self.process_text(formatted_text)
    
    def decode_tokens(self, tokens: List[int], clean_up_tokenization_spaces: bool = True) -> str:
        decoded = self.tokenizer.detokenize(tokens)
        
        if clean_up_tokenization_spaces:
            decoded = re.sub(r' +', ' ', decoded)
            decoded = re.sub(r' ([.!?,:;])', r'\1', decoded)
        
        postprocessed = self.postprocess_text(decoded)
        
        with self.lock:
            self.processing_stats['tokens_decoded'] += 1
        
        return postprocessed
    
    def batch_process(self, texts: List[str], batch_size: Optional[int] = None,
                     num_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        
        batch_size = batch_size or self.config['batch_size']
        num_workers = num_workers or self.config['num_workers']
        
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        results = self.tokenizer.batch_tokenize(
            preprocessed_texts, 
            batch_size=batch_size,
            num_workers=num_workers
        )
        
        with self.lock:
            self.processing_stats['batch_processes'] += 1
            self.processing_stats['texts_processed'] += len(texts)
        
        return results
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        vocab_stats = {
            'vocab_size': self.tokenizer.vocab_size,
            'special_tokens': len(self.tokenizer.bpe.special_tokens),
            'bpe_merges': len(self.tokenizer.bpe.bpe_merges),
            'encoder_size': len(self.tokenizer.bpe.encoder),
            'decoder_size': len(self.tokenizer.bpe.decoder)
        }
        
        return vocab_stats
    
    def benchmark_tokenization(self, test_texts: List[str], 
                             num_runs: int = 3) -> Dict[str, Any]:
        import time
        
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            for text in test_texts:
                self.process_text(text)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        tokens_per_second = len(test_texts) / avg_time
        
        return {
            'average_time': avg_time,
            'tokens_per_second': tokens_per_second,
            'test_texts_count': len(test_texts),
            'runs': num_runs
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.processing_stats)
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        analysis = self.tokenizer.analyze_tokenization(text)
        
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        
        complexity_metrics = {
            **analysis,
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'unique_words': len(set(word.lower() for word in words)),
            'lexical_diversity': len(set(word.lower() for word in words)) / len(words) if words else 0
        }
        
        return complexity_metrics
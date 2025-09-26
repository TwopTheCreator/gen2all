import threading
import time
import pickle
import lz4.frame
import xxhash
import sqlite3
import redis
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, OrderedDict
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc


class AdvancedMemoryPool:
    def __init__(self, initial_size: int = 1024 * 1024 * 1024):
        self.pool_size = initial_size
        self.memory_blocks = {}
        self.free_blocks = defaultdict(list)
        self.allocated_blocks = {}
        self.lock = threading.RLock()
        self.total_allocated = 0
        
    def allocate(self, size: int) -> bytes:
        with self.lock:
            aligned_size = self._align_size(size)
            
            if aligned_size in self.free_blocks and self.free_blocks[aligned_size]:
                block = self.free_blocks[aligned_size].pop()
                self.allocated_blocks[id(block)] = aligned_size
                return block
                
            block = bytearray(aligned_size)
            self.allocated_blocks[id(block)] = aligned_size
            self.total_allocated += aligned_size
            return block
    
    def deallocate(self, block: bytes):
        with self.lock:
            block_id = id(block)
            if block_id in self.allocated_blocks:
                size = self.allocated_blocks.pop(block_id)
                self.free_blocks[size].append(block)
                
    def _align_size(self, size: int) -> int:
        alignment = 64
        return ((size + alignment - 1) // alignment) * alignment
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_free = sum(len(blocks) * size for size, blocks in self.free_blocks.items())
            return {
                'total_allocated': self.total_allocated,
                'total_free_blocks': total_free,
                'active_allocations': len(self.allocated_blocks)
            }


class CompressedStorage:
    def __init__(self, compression_level: int = 4):
        self.compression_level = compression_level
        self.cache = OrderedDict()
        self.max_cache_size = 10000
        self.lock = threading.RLock()
        
    def compress_data(self, data: Any) -> bytes:
        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = lz4.frame.compress(serialized, compression_level=self.compression_level)
        return compressed
    
    def decompress_data(self, compressed_data: bytes) -> Any:
        decompressed = lz4.frame.decompress(compressed_data)
        return pickle.loads(decompressed)
    
    def store(self, key: str, data: Any) -> str:
        with self.lock:
            compressed = self.compress_data(data)
            data_hash = xxhash.xxh64(compressed).hexdigest()
            
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)
                
            self.cache[data_hash] = compressed
            return data_hash
    
    def retrieve(self, data_hash: str) -> Optional[Any]:
        with self.lock:
            if data_hash in self.cache:
                compressed = self.cache[data_hash]
                self.cache.move_to_end(data_hash)
                return self.decompress_data(compressed)
            return None


class PersistentMemoryStore:
    def __init__(self, db_path: str = "gen2all_memory.db", redis_host: str = "localhost", 
                 redis_port: int = 6379):
        self.db_path = db_path
        self.redis_client = None
        self.sqlite_conn = None
        self.lock = threading.RLock()
        
        self._init_sqlite()
        self._init_redis(redis_host, redis_port)
        
    def _init_sqlite(self):
        self.sqlite_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.sqlite_conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_store (
                key TEXT PRIMARY KEY,
                data BLOB,
                timestamp REAL,
                access_count INTEGER DEFAULT 0,
                compression_ratio REAL,
                data_size INTEGER
            )
        """)
        self.sqlite_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_store(timestamp)
        """)
        self.sqlite_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_access_count ON memory_store(access_count)
        """)
        self.sqlite_conn.commit()
    
    def _init_redis(self, host: str, port: int):
        try:
            self.redis_client = redis.Redis(host=host, port=port, decode_responses=False)
            self.redis_client.ping()
        except:
            self.redis_client = None
    
    def store(self, key: str, data: bytes, ttl: Optional[int] = None):
        with self.lock:
            timestamp = time.time()
            
            if self.redis_client:
                try:
                    if ttl:
                        self.redis_client.setex(key, ttl, data)
                    else:
                        self.redis_client.set(key, data)
                    return
                except:
                    pass
            
            compression_ratio = 1.0
            data_size = len(data)
            
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO memory_store 
                (key, data, timestamp, access_count, compression_ratio, data_size)
                VALUES (?, ?, ?, 
                    COALESCE((SELECT access_count FROM memory_store WHERE key = ?), 0),
                    ?, ?)
            """, (key, data, timestamp, key, compression_ratio, data_size))
            self.sqlite_conn.commit()
    
    def retrieve(self, key: str) -> Optional[bytes]:
        with self.lock:
            if self.redis_client:
                try:
                    data = self.redis_client.get(key)
                    if data:
                        return data
                except:
                    pass
            
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT data FROM memory_store WHERE key = ?
            """, (key,))
            result = cursor.fetchone()
            
            if result:
                cursor.execute("""
                    UPDATE memory_store 
                    SET access_count = access_count + 1 
                    WHERE key = ?
                """, (key,))
                self.sqlite_conn.commit()
                return result[0]
                
            return None
    
    def cleanup_old_entries(self, max_age_seconds: int = 86400):
        with self.lock:
            cutoff_time = time.time() - max_age_seconds
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                DELETE FROM memory_store 
                WHERE timestamp < ? AND access_count < 10
            """, (cutoff_time,))
            self.sqlite_conn.commit()
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            cursor = self.sqlite_conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(data_size) as total_size,
                    AVG(access_count) as avg_access_count,
                    AVG(compression_ratio) as avg_compression_ratio
                FROM memory_store
            """)
            result = cursor.fetchone()
            
            redis_stats = {}
            if self.redis_client:
                try:
                    redis_info = self.redis_client.info()
                    redis_stats = {
                        'redis_memory_used': redis_info.get('used_memory', 0),
                        'redis_keys': self.redis_client.dbsize()
                    }
                except:
                    pass
            
            return {
                'sqlite_entries': result[0] or 0,
                'sqlite_total_size': result[1] or 0,
                'avg_access_count': result[2] or 0,
                'avg_compression_ratio': result[3] or 0,
                **redis_stats
            }


class MemoryManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        self.memory_pool = AdvancedMemoryPool(self.config['pool_size'])
        self.compressed_storage = CompressedStorage(self.config['compression_level'])
        self.persistent_store = PersistentMemoryStore(
            self.config['db_path'],
            self.config['redis_host'],
            self.config['redis_port']
        )
        
        self.active_contexts = {}
        self.context_cache = OrderedDict()
        self.attention_cache = {}
        self.gradient_cache = {}
        
        self.lock = threading.RLock()
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        self.stats = defaultdict(int)
        
    def _default_config(self) -> Dict[str, Any]:
        available_memory = psutil.virtual_memory().available
        return {
            'pool_size': min(available_memory // 4, 8 * 1024 * 1024 * 1024),
            'compression_level': 4,
            'db_path': 'gen2all_memory.db',
            'redis_host': 'localhost',
            'redis_port': 6379,
            'max_context_cache': 1000,
            'cleanup_interval': 300,
            'max_memory_usage': 0.8
        }
    
    def allocate_context(self, context_id: str, size_hint: int = None) -> Dict[str, Any]:
        with self.lock:
            if context_id in self.active_contexts:
                return self.active_contexts[context_id]
            
            context = {
                'id': context_id,
                'created_at': time.time(),
                'last_accessed': time.time(),
                'memory_blocks': [],
                'compressed_data': {},
                'attention_states': {},
                'gradient_accumulator': None,
                'token_cache': OrderedDict(),
                'processing_state': 'idle'
            }
            
            if size_hint:
                context['memory_blocks'].append(
                    self.memory_pool.allocate(size_hint)
                )
            
            self.active_contexts[context_id] = context
            self.stats['contexts_created'] += 1
            
            return context
    
    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if context_id in self.active_contexts:
                context = self.active_contexts[context_id]
                context['last_accessed'] = time.time()
                return context
            
            if context_id in self.context_cache:
                context_data = self.context_cache[context_id]
                self.context_cache.move_to_end(context_id)
                
                restored_context = self.compressed_storage.decompress_data(context_data)
                self.active_contexts[context_id] = restored_context
                self.stats['contexts_restored'] += 1
                
                return restored_context
            
            persistent_data = self.persistent_store.retrieve(f"context_{context_id}")
            if persistent_data:
                context = pickle.loads(persistent_data)
                self.active_contexts[context_id] = context
                self.stats['contexts_loaded_from_disk'] += 1
                return context
            
            return None
    
    def store_attention_weights(self, context_id: str, layer_id: int, 
                              attention_weights: np.ndarray):
        with self.lock:
            key = f"{context_id}_{layer_id}"
            
            if key not in self.attention_cache:
                self.attention_cache[key] = {}
            
            compressed_weights = self.compressed_storage.compress_data(attention_weights)
            self.attention_cache[key]['weights'] = compressed_weights
            self.attention_cache[key]['timestamp'] = time.time()
            
            self.stats['attention_weights_stored'] += 1
    
    def get_attention_weights(self, context_id: str, layer_id: int) -> Optional[np.ndarray]:
        with self.lock:
            key = f"{context_id}_{layer_id}"
            
            if key in self.attention_cache:
                compressed_weights = self.attention_cache[key]['weights']
                return self.compressed_storage.decompress_data(compressed_weights)
            
            persistent_key = f"attention_{key}"
            persistent_data = self.persistent_store.retrieve(persistent_key)
            if persistent_data:
                return pickle.loads(persistent_data)
            
            return None
    
    def cache_tokens(self, context_id: str, tokens: List[int], 
                    embeddings: np.ndarray, max_cache_size: int = 8192):
        with self.lock:
            context = self.get_context(context_id)
            if not context:
                return
            
            token_key = xxhash.xxh64(str(tokens)).hexdigest()
            
            if len(context['token_cache']) >= max_cache_size:
                context['token_cache'].popitem(last=False)
            
            context['token_cache'][token_key] = {
                'tokens': tokens,
                'embeddings': self.compressed_storage.compress_data(embeddings),
                'timestamp': time.time()
            }
            
            self.stats['tokens_cached'] += 1
    
    def get_cached_tokens(self, context_id: str, tokens: List[int]) -> Optional[np.ndarray]:
        with self.lock:
            context = self.get_context(context_id)
            if not context:
                return None
            
            token_key = xxhash.xxh64(str(tokens)).hexdigest()
            
            if token_key in context['token_cache']:
                cached_data = context['token_cache'][token_key]
                context['token_cache'].move_to_end(token_key)
                
                embeddings = self.compressed_storage.decompress_data(
                    cached_data['embeddings']
                )
                self.stats['token_cache_hits'] += 1
                return embeddings
            
            self.stats['token_cache_misses'] += 1
            return None
    
    def save_context(self, context_id: str, compress: bool = True):
        with self.lock:
            if context_id not in self.active_contexts:
                return
            
            context = self.active_contexts[context_id]
            
            if compress:
                compressed_context = self.compressed_storage.compress_data(context)
                
                if len(self.context_cache) >= self.config['max_context_cache']:
                    oldest_key = next(iter(self.context_cache))
                    oldest_context = self.context_cache.pop(oldest_key)
                    
                    persistent_key = f"context_{oldest_key}"
                    self.persistent_store.store(persistent_key, oldest_context)
                
                self.context_cache[context_id] = compressed_context
            else:
                persistent_key = f"context_{context_id}"
                context_data = pickle.dumps(context)
                self.persistent_store.store(persistent_key, context_data)
            
            for block in context['memory_blocks']:
                self.memory_pool.deallocate(block)
            
            del self.active_contexts[context_id]
            self.stats['contexts_saved'] += 1
    
    def _cleanup_loop(self):
        while True:
            try:
                time.sleep(self.config['cleanup_interval'])
                self._cleanup_expired_contexts()
                self._cleanup_attention_cache()
                self.persistent_store.cleanup_old_entries()
                gc.collect()
            except Exception as e:
                print(f"Cleanup error: {e}")
    
    def _cleanup_expired_contexts(self):
        with self.lock:
            current_time = time.time()
            expired_contexts = []
            
            for context_id, context in self.active_contexts.items():
                if current_time - context['last_accessed'] > 3600:
                    expired_contexts.append(context_id)
            
            for context_id in expired_contexts:
                self.save_context(context_id)
                self.stats['contexts_expired'] += 1
    
    def _cleanup_attention_cache(self):
        with self.lock:
            current_time = time.time()
            expired_keys = []
            
            for key, data in self.attention_cache.items():
                if current_time - data['timestamp'] > 1800:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.attention_cache[key]
                self.stats['attention_weights_expired'] += 1
    
    def get_memory_stats(self) -> Dict[str, Any]:
        with self.lock:
            memory_info = psutil.virtual_memory()
            
            stats = {
                'system_memory': {
                    'total': memory_info.total,
                    'available': memory_info.available,
                    'percent_used': memory_info.percent
                },
                'memory_pool': self.memory_pool.get_stats(),
                'persistent_store': self.persistent_store.get_stats(),
                'active_contexts': len(self.active_contexts),
                'cached_contexts': len(self.context_cache),
                'attention_cache_size': len(self.attention_cache),
                'operation_stats': dict(self.stats)
            }
            
            return stats
    
    def optimize_memory(self):
        with self.lock:
            memory_usage = psutil.virtual_memory().percent / 100.0
            
            if memory_usage > self.config['max_memory_usage']:
                contexts_to_save = min(
                    len(self.active_contexts) // 2,
                    20
                )
                
                sorted_contexts = sorted(
                    self.active_contexts.items(),
                    key=lambda x: x[1]['last_accessed']
                )
                
                for context_id, _ in sorted_contexts[:contexts_to_save]:
                    self.save_context(context_id)
                
                self._cleanup_attention_cache()
                gc.collect()
                
                self.stats['memory_optimizations'] += 1
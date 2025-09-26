import time
import threading
from typing import Dict, Optional
from collections import defaultdict, deque
import redis
import sqlite3


class TokenBucket:
    def __init__(self, capacity: int, refill_rate: int):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def _refill(self):
        now = time.time()
        time_passed = now - self.last_refill
        
        tokens_to_add = int(time_passed * self.refill_rate)
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now


class SlidingWindow:
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        with self.lock:
            now = time.time()
            cutoff_time = now - self.window_size
            
            while self.requests and self.requests[0] < cutoff_time:
                self.requests.popleft()
            
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False


class RateLimiter:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        self.buckets = {}
        self.windows = {}
        self.user_limits = {}
        
        self.redis_client = None
        if self.config['use_redis']:
            try:
                self.redis_client = redis.Redis(
                    host=self.config['redis_host'],
                    port=self.config['redis_port'],
                    decode_responses=True
                )
                self.redis_client.ping()
            except:
                self.redis_client = None
        
        self.db_path = self.config.get('db_path', 'gen2all_rate_limits.db')
        self._init_database()
        
        self.lock = threading.RLock()
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _default_config(self) -> Dict:
        return {
            'default_rate_limit': 1000,
            'default_burst_limit': 100,
            'window_size': 3600,
            'use_redis': False,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'cleanup_interval': 300,
            'enable_adaptive_limits': True,
            'penalty_multiplier': 0.5
        }
    
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_limits (
                    user_id TEXT PRIMARY KEY,
                    requests_per_hour INTEGER,
                    burst_limit INTEGER,
                    current_usage INTEGER DEFAULT 0,
                    penalty_until REAL DEFAULT 0,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.commit()
    
    def set_user_limits(self, user_id: str, requests_per_hour: int, burst_limit: int):
        with self.lock:
            self.user_limits[user_id] = {
                'requests_per_hour': requests_per_hour,
                'burst_limit': burst_limit,
                'updated_at': time.time()
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO rate_limits 
                    (user_id, requests_per_hour, burst_limit, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, requests_per_hour, burst_limit, time.time(), time.time()))
                conn.commit()
    
    def get_user_limits(self, user_id: str) -> Dict:
        with self.lock:
            if user_id in self.user_limits:
                return self.user_limits[user_id]
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT requests_per_hour, burst_limit, penalty_until
                    FROM rate_limits 
                    WHERE user_id = ?
                """, (user_id,))
                
                result = cursor.fetchone()
                
                if result:
                    limits = {
                        'requests_per_hour': result[0],
                        'burst_limit': result[1],
                        'penalty_until': result[2]
                    }
                    self.user_limits[user_id] = limits
                    return limits
            
            return {
                'requests_per_hour': self.config['default_rate_limit'],
                'burst_limit': self.config['default_burst_limit'],
                'penalty_until': 0
            }
    
    def check_rate_limit(self, api_key: str, tokens_requested: int = 1) -> bool:
        user_id = self._extract_user_id(api_key)
        
        if self._is_penalized(user_id):
            return False
        
        if self.redis_client:
            return self._check_redis_rate_limit(user_id, tokens_requested)
        else:
            return self._check_local_rate_limit(user_id, tokens_requested)
    
    def _extract_user_id(self, api_key: str) -> str:
        return api_key[:16] if api_key else "anonymous"
    
    def _is_penalized(self, user_id: str) -> bool:
        limits = self.get_user_limits(user_id)
        return time.time() < limits.get('penalty_until', 0)
    
    def _check_redis_rate_limit(self, user_id: str, tokens_requested: int) -> bool:
        try:
            limits = self.get_user_limits(user_id)
            
            pipe = self.redis_client.pipeline()
            
            hourly_key = f"rate_limit:{user_id}:hour"
            burst_key = f"rate_limit:{user_id}:burst"
            
            pipe.get(hourly_key)
            pipe.get(burst_key)
            hourly_count, burst_count = pipe.execute()
            
            hourly_count = int(hourly_count or 0)
            burst_count = int(burst_count or 0)
            
            if (hourly_count + tokens_requested > limits['requests_per_hour'] or
                burst_count + tokens_requested > limits['burst_limit']):
                return False
            
            pipe.incr(hourly_key, tokens_requested)
            pipe.expire(hourly_key, 3600)
            pipe.incr(burst_key, tokens_requested)
            pipe.expire(burst_key, 60)
            pipe.execute()
            
            return True
            
        except:
            return self._check_local_rate_limit(user_id, tokens_requested)
    
    def _check_local_rate_limit(self, user_id: str, tokens_requested: int) -> bool:
        with self.lock:
            limits = self.get_user_limits(user_id)
            
            if user_id not in self.buckets:
                self.buckets[user_id] = TokenBucket(
                    capacity=limits['burst_limit'],
                    refill_rate=limits['requests_per_hour'] / 3600
                )
            
            if user_id not in self.windows:
                self.windows[user_id] = SlidingWindow(
                    window_size=3600,
                    max_requests=limits['requests_per_hour']
                )
            
            bucket = self.buckets[user_id]
            window = self.windows[user_id]
            
            if bucket.consume(tokens_requested) and window.is_allowed():
                return True
            
            if self.config['enable_adaptive_limits']:
                self._apply_penalty(user_id)
            
            return False
    
    def _apply_penalty(self, user_id: str):
        penalty_duration = 300
        penalty_until = time.time() + penalty_duration
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE rate_limits 
                SET penalty_until = ? 
                WHERE user_id = ?
            """, (penalty_until, user_id))
            conn.commit()
        
        if user_id in self.user_limits:
            self.user_limits[user_id]['penalty_until'] = penalty_until
    
    def get_remaining_quota(self, api_key: str) -> Dict:
        user_id = self._extract_user_id(api_key)
        limits = self.get_user_limits(user_id)
        
        if self.redis_client:
            try:
                hourly_key = f"rate_limit:{user_id}:hour"
                burst_key = f"rate_limit:{user_id}:burst"
                
                pipe = self.redis_client.pipeline()
                pipe.get(hourly_key)
                pipe.get(burst_key)
                hourly_used, burst_used = pipe.execute()
                
                hourly_used = int(hourly_used or 0)
                burst_used = int(burst_used or 0)
                
                return {
                    'hourly_remaining': max(0, limits['requests_per_hour'] - hourly_used),
                    'burst_remaining': max(0, limits['burst_limit'] - burst_used),
                    'penalty_until': limits.get('penalty_until', 0)
                }
            except:
                pass
        
        with self.lock:
            bucket = self.buckets.get(user_id)
            window = self.windows.get(user_id)
            
            if bucket and window:
                return {
                    'burst_remaining': int(bucket.tokens),
                    'hourly_remaining': limits['requests_per_hour'] - len(window.requests),
                    'penalty_until': limits.get('penalty_until', 0)
                }
        
        return {
            'hourly_remaining': limits['requests_per_hour'],
            'burst_remaining': limits['burst_limit'],
            'penalty_until': limits.get('penalty_until', 0)
        }
    
    def reset_user_limits(self, user_id: str):
        with self.lock:
            if user_id in self.buckets:
                del self.buckets[user_id]
            
            if user_id in self.windows:
                del self.windows[user_id]
            
            if user_id in self.user_limits:
                self.user_limits[user_id]['penalty_until'] = 0
            
            if self.redis_client:
                try:
                    self.redis_client.delete(f"rate_limit:{user_id}:hour")
                    self.redis_client.delete(f"rate_limit:{user_id}:burst")
                except:
                    pass
    
    def _cleanup_loop(self):
        while True:
            try:
                time.sleep(self.config['cleanup_interval'])
                self._cleanup_expired_buckets()
                self._cleanup_expired_penalties()
            except Exception as e:
                print(f"Rate limiter cleanup error: {e}")
    
    def _cleanup_expired_buckets(self):
        with self.lock:
            expired_users = []
            current_time = time.time()
            
            for user_id, bucket in self.buckets.items():
                if current_time - bucket.last_refill > 7200:
                    expired_users.append(user_id)
            
            for user_id in expired_users:
                del self.buckets[user_id]
                if user_id in self.windows:
                    del self.windows[user_id]
    
    def _cleanup_expired_penalties(self):
        current_time = time.time()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE rate_limits 
                SET penalty_until = 0 
                WHERE penalty_until > 0 AND penalty_until < ?
            """, (current_time,))
            conn.commit()
    
    def get_rate_limit_stats(self) -> Dict:
        with self.lock:
            stats = {
                'active_buckets': len(self.buckets),
                'active_windows': len(self.windows),
                'penalized_users': 0,
                'total_users': len(self.user_limits)
            }
            
            current_time = time.time()
            for limits in self.user_limits.values():
                if current_time < limits.get('penalty_until', 0):
                    stats['penalized_users'] += 1
            
            return stats
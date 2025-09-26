import hashlib
import secrets
import sqlite3
import threading
import time
from typing import Dict, List, Optional, Set
import redis
import json
from cryptography.fernet import Fernet
from collections import defaultdict


class APIKeyManager:
    def __init__(self, db_path: str = "gen2all_auth.db"):
        self.db_path = db_path
        self.lock = threading.RLock()
        self._init_database()
        
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_hash TEXT PRIMARY KEY,
                    key_prefix TEXT,
                    user_id TEXT,
                    quota_limit INTEGER DEFAULT -1,
                    quota_used INTEGER DEFAULT 0,
                    rate_limit INTEGER DEFAULT 1000,
                    created_at REAL,
                    last_used REAL,
                    is_active BOOLEAN DEFAULT 1,
                    permissions TEXT DEFAULT '{}',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_key_prefix ON api_keys(key_prefix)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id ON api_keys(user_id)
            """)
            conn.commit()
    
    def generate_api_key(self, user_id: str, quota_limit: int = -1, 
                        rate_limit: int = 1000, permissions: Dict = None) -> str:
        with self.lock:
            key = f"gen2_{secrets.token_urlsafe(48)}"
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            key_prefix = key[:12]
            
            permissions = permissions or {
                'generate': True,
                'batch_generate': True,
                'contexts': True,
                'stats': True
            }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO api_keys 
                    (key_hash, key_prefix, user_id, quota_limit, rate_limit, 
                     created_at, last_used, permissions)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (key_hash, key_prefix, user_id, quota_limit, rate_limit,
                      time.time(), time.time(), json.dumps(permissions)))
                conn.commit()
            
            return key
    
    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        with self.lock:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT user_id, quota_limit, quota_used, rate_limit, 
                           is_active, permissions, metadata
                    FROM api_keys 
                    WHERE key_hash = ?
                """, (key_hash,))
                
                result = cursor.fetchone()
                
                if result and result[4]:
                    cursor.execute("""
                        UPDATE api_keys 
                        SET last_used = ? 
                        WHERE key_hash = ?
                    """, (time.time(), key_hash))
                    conn.commit()
                    
                    return {
                        'user_id': result[0],
                        'quota_limit': result[1],
                        'quota_used': result[2],
                        'rate_limit': result[3],
                        'permissions': json.loads(result[5]),
                        'metadata': json.loads(result[6])
                    }
                
                return None
    
    def update_quota(self, api_key: str, tokens_used: int) -> bool:
        with self.lock:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE api_keys 
                    SET quota_used = quota_used + ? 
                    WHERE key_hash = ?
                """, (tokens_used, key_hash))
                
                return cursor.rowcount > 0
    
    def deactivate_key(self, api_key: str) -> bool:
        with self.lock:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE api_keys 
                    SET is_active = 0 
                    WHERE key_hash = ?
                """, (key_hash,))
                
                return cursor.rowcount > 0
    
    def list_keys(self, user_id: Optional[str] = None) -> List[Dict]:
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if user_id:
                    cursor.execute("""
                        SELECT key_prefix, user_id, quota_limit, quota_used, 
                               rate_limit, created_at, last_used, is_active
                        FROM api_keys 
                        WHERE user_id = ?
                    """, (user_id,))
                else:
                    cursor.execute("""
                        SELECT key_prefix, user_id, quota_limit, quota_used, 
                               rate_limit, created_at, last_used, is_active
                        FROM api_keys
                    """)
                
                results = cursor.fetchall()
                
                return [{
                    'key_prefix': row[0],
                    'user_id': row[1],
                    'quota_limit': row[2],
                    'quota_used': row[3],
                    'rate_limit': row[4],
                    'created_at': row[5],
                    'last_used': row[6],
                    'is_active': bool(row[7])
                } for row in results]


class AuthManager:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        self.api_key_manager = APIKeyManager(self.config['db_path'])
        
        self.session_cache = {}
        self.failed_attempts = defaultdict(int)
        
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
        
        self.lock = threading.RLock()
    
    def _default_config(self) -> Dict:
        return {
            'db_path': 'gen2all_auth.db',
            'use_redis': False,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'session_timeout': 3600,
            'max_failed_attempts': 10,
            'lockout_duration': 300,
            'require_user_registration': False
        }
    
    def register_user(self, user_id: str, quota_limit: int = -1) -> str:
        with self.lock:
            api_key = self.api_key_manager.generate_api_key(
                user_id=user_id,
                quota_limit=quota_limit,
                rate_limit=self.config.get('default_rate_limit', 1000)
            )
            
            return api_key
    
    def validate_api_key(self, api_key: str) -> bool:
        if not api_key or not api_key.startswith('gen2_'):
            return False
        
        with self.lock:
            if api_key in self.session_cache:
                session = self.session_cache[api_key]
                if time.time() - session['last_validated'] < 300:
                    session['last_validated'] = time.time()
                    return True
            
            key_info = self.api_key_manager.validate_api_key(api_key)
            
            if key_info:
                if key_info['quota_limit'] > 0 and key_info['quota_used'] >= key_info['quota_limit']:
                    return False
                
                self.session_cache[api_key] = {
                    'user_id': key_info['user_id'],
                    'permissions': key_info['permissions'],
                    'last_validated': time.time()
                }
                
                if len(self.session_cache) > 1000:
                    oldest_key = min(self.session_cache.keys(), 
                                   key=lambda k: self.session_cache[k]['last_validated'])
                    del self.session_cache[oldest_key]
                
                return True
            
            return False
    
    def get_user_permissions(self, api_key: str) -> Dict:
        with self.lock:
            if api_key in self.session_cache:
                return self.session_cache[api_key]['permissions']
            
            key_info = self.api_key_manager.validate_api_key(api_key)
            if key_info:
                return key_info['permissions']
            
            return {}
    
    def check_permission(self, api_key: str, permission: str) -> bool:
        permissions = self.get_user_permissions(api_key)
        return permissions.get(permission, False)
    
    def update_usage(self, api_key: str, tokens_used: int):
        self.api_key_manager.update_quota(api_key, tokens_used)
    
    def revoke_api_key(self, api_key: str) -> bool:
        with self.lock:
            success = self.api_key_manager.deactivate_key(api_key)
            
            if api_key in self.session_cache:
                del self.session_cache[api_key]
            
            return success
    
    def get_user_stats(self, api_key: str) -> Optional[Dict]:
        key_info = self.api_key_manager.validate_api_key(api_key)
        if key_info:
            return {
                'user_id': key_info['user_id'],
                'quota_limit': key_info['quota_limit'],
                'quota_used': key_info['quota_used'],
                'quota_remaining': key_info['quota_limit'] - key_info['quota_used'] if key_info['quota_limit'] > 0 else -1,
                'rate_limit': key_info['rate_limit']
            }
        return None
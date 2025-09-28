"""
Gen2All Database Management System
Handles all database operations with support for multiple database types
"""
import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from config import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseInterface(ABC):
    """Abstract interface for database operations"""
    
    @abstractmethod
    def connect(self):
        pass
    
    @abstractmethod
    def disconnect(self):
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: tuple = None) -> Any:
        pass
    
    @abstractmethod
    def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        pass
    
    @abstractmethod
    def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        pass

class SQLiteDatabase(DatabaseInterface):
    """SQLite database implementation"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.database.path
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            self.cursor = self.connection.cursor()
            self._create_tables()
            logger.info(f"Connected to SQLite database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from database"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from database")
    
    def _create_tables(self):
        """Create necessary tables"""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                file_path TEXT,
                size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tokenizers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL,
                vocab_data TEXT,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER,
                dataset_id INTEGER,
                status TEXT DEFAULT 'pending',
                progress REAL DEFAULT 0.0,
                logs TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models (id),
                FOREIGN KEY (dataset_id) REFERENCES datasets (id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        
        for table_sql in tables:
            self.cursor.execute(table_sql)
        self.connection.commit()
    
    def execute_query(self, query: str, params: tuple = None) -> Any:
        """Execute a query and return cursor"""
        try:
            if params:
                result = self.cursor.execute(query, params)
            else:
                result = self.cursor.execute(query)
            self.connection.commit()
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            self.connection.rollback()
            raise
    
    def fetch_all(self, query: str, params: tuple = None) -> List[Dict]:
        """Fetch all results from query"""
        try:
            self.execute_query(query, params)
            rows = self.cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Fetch all failed: {e}")
            return []
    
    def fetch_one(self, query: str, params: tuple = None) -> Optional[Dict]:
        """Fetch one result from query"""
        try:
            self.execute_query(query, params)
            row = self.cursor.fetchone()
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Fetch one failed: {e}")
            return None

class DatabaseManager:
    """High-level database management"""
    
    def __init__(self):
        self.db: DatabaseInterface = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database based on configuration"""
        db_type = config.database.type.lower()
        
        if db_type == "sqlite":
            self.db = SQLiteDatabase()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
        
        self.db.connect()
    
    # Model operations
    def create_model(self, name: str, model_type: str, model_config: Dict = None) -> int:
        """Create a new model record"""
        config_json = json.dumps(model_config) if model_config else None
        
        query = """
        INSERT INTO models (name, type, config)
        VALUES (?, ?, ?)
        """
        
        self.db.execute_query(query, (name, model_type, config_json))
        return self.db.cursor.lastrowid
    
    def get_model(self, model_id: int) -> Optional[Dict]:
        """Get model by ID"""
        query = "SELECT * FROM models WHERE id = ?"
        model = self.db.fetch_one(query, (model_id,))
        
        if model and model['config']:
            model['config'] = json.loads(model['config'])
        
        return model
    
    def get_all_models(self) -> List[Dict]:
        """Get all models"""
        query = "SELECT * FROM models ORDER BY created_at DESC"
        models = self.db.fetch_all(query)
        
        for model in models:
            if model['config']:
                model['config'] = json.loads(model['config'])
        
        return models
    
    def update_model(self, model_id: int, updates: Dict):
        """Update model record"""
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            if key in ['name', 'type', 'config']:
                set_clauses.append(f"{key} = ?")
                if key == 'config':
                    params.append(json.dumps(value))
                else:
                    params.append(value)
        
        if set_clauses:
            set_clauses.append("updated_at = ?")
            params.append(datetime.now())
            params.append(model_id)
            
            query = f"UPDATE models SET {', '.join(set_clauses)} WHERE id = ?"
            self.db.execute_query(query, tuple(params))
    
    def delete_model(self, model_id: int):
        """Delete model record"""
        query = "DELETE FROM models WHERE id = ?"
        self.db.execute_query(query, (model_id,))
    
    # Dataset operations
    def create_dataset(self, name: str, description: str = None, 
                      file_path: str = None, size: int = None) -> int:
        """Create a new dataset record"""
        query = """
        INSERT INTO datasets (name, description, file_path, size)
        VALUES (?, ?, ?, ?)
        """
        
        self.db.execute_query(query, (name, description, file_path, size))
        return self.db.cursor.lastrowid
    
    def get_dataset(self, dataset_id: int) -> Optional[Dict]:
        """Get dataset by ID"""
        query = "SELECT * FROM datasets WHERE id = ?"
        return self.db.fetch_one(query, (dataset_id,))
    
    def get_all_datasets(self) -> List[Dict]:
        """Get all datasets"""
        query = "SELECT * FROM datasets ORDER BY created_at DESC"
        return self.db.fetch_all(query)
    
    # Tokenizer operations
    def create_tokenizer(self, name: str, tokenizer_type: str, 
                        vocab_data: Dict = None, tokenizer_config: Dict = None) -> int:
        """Create a new tokenizer record"""
        vocab_json = json.dumps(vocab_data) if vocab_data else None
        config_json = json.dumps(tokenizer_config) if tokenizer_config else None
        
        query = """
        INSERT INTO tokenizers (name, type, vocab_data, config)
        VALUES (?, ?, ?, ?)
        """
        
        self.db.execute_query(query, (name, tokenizer_type, vocab_json, config_json))
        return self.db.cursor.lastrowid
    
    def get_tokenizer(self, tokenizer_id: int) -> Optional[Dict]:
        """Get tokenizer by ID"""
        query = "SELECT * FROM tokenizers WHERE id = ?"
        tokenizer = self.db.fetch_one(query, (tokenizer_id,))
        
        if tokenizer:
            if tokenizer['vocab_data']:
                tokenizer['vocab_data'] = json.loads(tokenizer['vocab_data'])
            if tokenizer['config']:
                tokenizer['config'] = json.loads(tokenizer['config'])
        
        return tokenizer
    
    def get_all_tokenizers(self) -> List[Dict]:
        """Get all tokenizers"""
        query = "SELECT * FROM tokenizers ORDER BY created_at DESC"
        tokenizers = self.db.fetch_all(query)
        
        for tokenizer in tokenizers:
            if tokenizer['vocab_data']:
                tokenizer['vocab_data'] = json.loads(tokenizer['vocab_data'])
            if tokenizer['config']:
                tokenizer['config'] = json.loads(tokenizer['config'])
        
        return tokenizers
    
    # Training session operations
    def create_training_session(self, model_id: int, dataset_id: int) -> int:
        """Create a new training session"""
        query = """
        INSERT INTO training_sessions (model_id, dataset_id, started_at)
        VALUES (?, ?, ?)
        """
        
        self.db.execute_query(query, (model_id, dataset_id, datetime.now()))
        return self.db.cursor.lastrowid
    
    def update_training_progress(self, session_id: int, progress: float, 
                               status: str = None, logs: str = None):
        """Update training session progress"""
        updates = {'progress': progress}
        
        if status:
            updates['status'] = status
        if logs:
            updates['logs'] = logs
            
        if status == 'completed':
            updates['completed_at'] = datetime.now()
        
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            params.append(value)
        
        params.append(session_id)
        query = f"UPDATE training_sessions SET {', '.join(set_clauses)} WHERE id = ?"
        self.db.execute_query(query, tuple(params))
    
    def get_training_session(self, session_id: int) -> Optional[Dict]:
        """Get training session by ID"""
        query = """
        SELECT ts.*, m.name as model_name, d.name as dataset_name
        FROM training_sessions ts
        LEFT JOIN models m ON ts.model_id = m.id
        LEFT JOIN datasets d ON ts.dataset_id = d.id
        WHERE ts.id = ?
        """
        return self.db.fetch_one(query, (session_id,))
    
    def get_all_training_sessions(self) -> List[Dict]:
        """Get all training sessions"""
        query = """
        SELECT ts.*, m.name as model_name, d.name as dataset_name
        FROM training_sessions ts
        LEFT JOIN models m ON ts.model_id = m.id
        LEFT JOIN datasets d ON ts.dataset_id = d.id
        ORDER BY ts.started_at DESC
        """
        return self.db.fetch_all(query)
    
    # User preferences
    def save_user_preferences(self, user_id: str, preferences: Dict):
        """Save user preferences"""
        prefs_json = json.dumps(preferences)
        
        # Try to update existing preferences
        query = "UPDATE user_preferences SET preferences = ?, updated_at = ? WHERE user_id = ?"
        result = self.db.execute_query(query, (prefs_json, datetime.now(), user_id))
        
        # If no rows affected, insert new record
        if self.db.cursor.rowcount == 0:
            query = "INSERT INTO user_preferences (user_id, preferences) VALUES (?, ?)"
            self.db.execute_query(query, (user_id, prefs_json))
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict]:
        """Get user preferences"""
        query = "SELECT preferences FROM user_preferences WHERE user_id = ?"
        result = self.db.fetch_one(query, (user_id,))
        
        if result and result['preferences']:
            return json.loads(result['preferences'])
        
        return None
    
    def close(self):
        """Close database connection"""
        if self.db:
            self.db.disconnect()

# Global database manager instance
db_manager = DatabaseManager()
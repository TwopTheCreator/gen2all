"""
Gen2All Configuration Management System
Handles all configuration and customization options
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    type: str = "sqlite"
    path: str = "gen2all.db"
    host: str = "localhost"
    port: int = 5432
    username: str = ""
    password: str = ""
    
@dataclass
class TokenizerConfig:
    """Tokenizer configuration settings"""
    type: str = "word"  # word, character, custom
    max_tokens: int = 1000
    vocab_size: int = 10000
    special_tokens: Dict[str, str] = None
    case_sensitive: bool = False
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {
                "pad": "[PAD]",
                "unk": "[UNK]",
                "start": "[START]",
                "end": "[END]"
            }

@dataclass
class GUIConfig:
    """GUI configuration settings"""
    theme: str = "dark"
    width: int = 1200
    height: int = 800
    font_family: str = "Arial"
    font_size: int = 12
    
@dataclass
class APIConfig:
    """API configuration settings"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    cors_enabled: bool = True

class ConfigManager:
    """Central configuration management"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.database = DatabaseConfig()
        self.tokenizer = TokenizerConfig()
        self.gui = GUIConfig()
        self.api = APIConfig()
        self.custom_settings: Dict[str, Any] = {}
        
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                
                # Update configurations
                if 'database' in config_data:
                    for key, value in config_data['database'].items():
                        if hasattr(self.database, key):
                            setattr(self.database, key, value)
                
                if 'tokenizer' in config_data:
                    for key, value in config_data['tokenizer'].items():
                        if hasattr(self.tokenizer, key):
                            setattr(self.tokenizer, key, value)
                
                if 'gui' in config_data:
                    for key, value in config_data['gui'].items():
                        if hasattr(self.gui, key):
                            setattr(self.gui, key, value)
                
                if 'api' in config_data:
                    for key, value in config_data['api'].items():
                        if hasattr(self.api, key):
                            setattr(self.api, key, value)
                
                self.custom_settings = config_data.get('custom', {})
                
            except Exception as e:
                print(f"Error loading config: {e}")
    
    def save_config(self):
        """Save configuration to file"""
        config_data = {
            'database': {
                'type': self.database.type,
                'path': self.database.path,
                'host': self.database.host,
                'port': self.database.port,
                'username': self.database.username,
                'password': self.database.password
            },
            'tokenizer': {
                'type': self.tokenizer.type,
                'max_tokens': self.tokenizer.max_tokens,
                'vocab_size': self.tokenizer.vocab_size,
                'special_tokens': self.tokenizer.special_tokens,
                'case_sensitive': self.tokenizer.case_sensitive
            },
            'gui': {
                'theme': self.gui.theme,
                'width': self.gui.width,
                'height': self.gui.height,
                'font_family': self.gui.font_family,
                'font_size': self.gui.font_size
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'debug': self.api.debug,
                'cors_enabled': self.api.cors_enabled
            },
            'custom': self.custom_settings
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_custom_setting(self, key: str, default: Any = None) -> Any:
        """Get custom setting value"""
        return self.custom_settings.get(key, default)
    
    def set_custom_setting(self, key: str, value: Any):
        """Set custom setting value"""
        self.custom_settings[key] = value
        self.save_config()

# Global configuration instance
config = ConfigManager()
import json
import yaml
import os
from typing import Dict, Any, Optional
import threading


class ConfigLoader:
    def __init__(self):
        self.configs = {}
        self.lock = threading.RLock()
    
    def load_config(self, config_path: str, config_type: str = 'auto') -> Dict[str, Any]:
        with self.lock:
            if config_path in self.configs:
                return self.configs[config_path]
            
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            if config_type == 'auto':
                config_type = self._detect_config_type(config_path)
            
            config = self._load_config_file(config_path, config_type)
            
            config = self._resolve_environment_variables(config)
            config = self._merge_with_defaults(config)
            
            self.configs[config_path] = config
            return config
    
    def _detect_config_type(self, config_path: str) -> str:
        extension = os.path.splitext(config_path)[1].lower()
        
        if extension in ['.json']:
            return 'json'
        elif extension in ['.yaml', '.yml']:
            return 'yaml'
        else:
            return 'json'
    
    def _load_config_file(self, config_path: str, config_type: str) -> Dict[str, Any]:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_type == 'json':
                return json.load(f)
            elif config_type == 'yaml':
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config type: {config_type}")
    
    def _resolve_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(config, dict):
            resolved = {}
            for key, value in config.items():
                resolved[key] = self._resolve_environment_variables(value)
            return resolved
        elif isinstance(config, list):
            return [self._resolve_environment_variables(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            default_value = None
            
            if ':' in env_var:
                env_var, default_value = env_var.split(':', 1)
            
            return os.environ.get(env_var, default_value)
        else:
            return config
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        defaults = {
            'engine': {
                'model_config': {
                    'vocab_size': 65536,
                    'd_model': 2048,
                    'num_layers': 48,
                    'num_heads': 32,
                    'd_ff': 8192,
                    'max_seq_length': 8192,
                    'dropout': 0.1
                },
                'generation_config': {
                    'max_length': 2048,
                    'temperature': 0.8,
                    'top_k': 50,
                    'top_p': 0.9,
                    'repetition_penalty': 1.1
                }
            },
            'api': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 4,
                'enable_auth': True,
                'enable_rate_limiting': True
            },
            'memory': {
                'pool_size': 8 * 1024 * 1024 * 1024,
                'compression_level': 4,
                'max_context_cache': 1000
            },
            'training': {
                'batch_size': 8,
                'learning_rate': 1e-4,
                'epochs': 10,
                'save_steps': 1000
            }
        }
        
        return self._deep_merge(defaults, config)
    
    def _deep_merge(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = default.copy()
        
        for key, value in override.items():
            if (key in result and 
                isinstance(result[key], dict) and 
                isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config: Dict[str, Any], config_path: str, config_type: str = 'json'):
        with self.lock:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_type == 'json':
                    json.dump(config, f, indent=2, ensure_ascii=False)
                elif config_type == 'yaml':
                    yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
                else:
                    raise ValueError(f"Unsupported config type: {config_type}")
            
            self.configs[config_path] = config
    
    def reload_config(self, config_path: str) -> Dict[str, Any]:
        with self.lock:
            if config_path in self.configs:
                del self.configs[config_path]
            return self.load_config(config_path)
    
    def get_config_value(self, config: Dict[str, Any], key_path: str, 
                        default: Optional[Any] = None) -> Any:
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set_config_value(self, config: Dict[str, Any], key_path: str, value: Any):
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
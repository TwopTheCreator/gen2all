import logging
import logging.handlers
import os
import json
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'context'):
            log_entry['context'] = record.context
        
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class PerformanceLogger:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}
        self.lock = threading.RLock()
    
    def start_timer(self, operation: str, context: Optional[Dict[str, Any]] = None):
        with self.lock:
            self.timers[operation] = {
                'start_time': time.time(),
                'context': context or {}
            }
    
    def end_timer(self, operation: str, additional_context: Optional[Dict[str, Any]] = None):
        with self.lock:
            if operation not in self.timers:
                self.logger.warning(f"Timer for operation '{operation}' not found")
                return
            
            timer_data = self.timers.pop(operation)
            duration = time.time() - timer_data['start_time']
            
            context = {
                **timer_data['context'],
                **(additional_context or {}),
                'duration_seconds': duration,
                'operation': operation
            }
            
            self.logger.info(
                f"Operation '{operation}' completed in {duration:.4f}s",
                extra={'context': context}
            )
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        context = {
            'type': 'performance_metrics',
            **metrics
        }
        
        self.logger.info("Performance metrics", extra={'context': context})


def setup_logging(config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    config = config or _default_logging_config()
    
    logger = logging.getLogger('gen2all')
    logger.setLevel(getattr(logging, config['level'].upper()))
    
    logger.handlers.clear()
    
    if config['console']['enabled']:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config['console']['level'].upper()))
        
        if config['console']['format'] == 'json':
            console_handler.setFormatter(JSONFormatter())
        else:
            console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            console_handler.setFormatter(logging.Formatter(console_format))
        
        logger.addHandler(console_handler)
    
    if config['file']['enabled']:
        log_dir = os.path.dirname(config['file']['path'])
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            config['file']['path'],
            maxBytes=config['file']['max_size'],
            backupCount=config['file']['backup_count']
        )
        file_handler.setLevel(getattr(logging, config['file']['level'].upper()))
        
        if config['file']['format'] == 'json':
            file_handler.setFormatter(JSONFormatter())
        else:
            file_format = '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
            file_handler.setFormatter(logging.Formatter(file_format))
        
        logger.addHandler(file_handler)
    
    if config.get('performance', {}).get('enabled', False):
        performance_logger = PerformanceLogger(logger)
        logger.performance = performance_logger
    
    logger.propagate = False
    
    return logger


def _default_logging_config() -> Dict[str, Any]:
    return {
        'level': 'INFO',
        'console': {
            'enabled': True,
            'level': 'INFO',
            'format': 'text'
        },
        'file': {
            'enabled': True,
            'level': 'DEBUG',
            'format': 'json',
            'path': './logs/gen2all.log',
            'max_size': 10 * 1024 * 1024,
            'backup_count': 5
        },
        'performance': {
            'enabled': True
        }
    }


class LogContext:
    def __init__(self, logger: logging.Logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation}", extra={'context': self.context})
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        context = {
            **self.context,
            'duration_seconds': duration,
            'success': exc_type is None
        }
        
        if exc_type:
            context['error_type'] = exc_type.__name__
            context['error_message'] = str(exc_val)
            self.logger.error(f"Operation '{self.operation}' failed", extra={'context': context})
        else:
            self.logger.info(f"Operation '{self.operation}' completed", extra={'context': context})
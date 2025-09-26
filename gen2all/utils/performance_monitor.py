import psutil
import threading
import time
import json
from typing import Dict, Any, List, Optional, Callable
from collections import deque, defaultdict
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class SystemMetrics:
    def __init__(self):
        self.process = psutil.Process()
        
    def get_cpu_metrics(self) -> Dict[str, float]:
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'process_cpu_percent': self.process.cpu_percent(),
            'load_avg': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
    
    def get_memory_metrics(self) -> Dict[str, float]:
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        
        return {
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'used_memory_gb': memory.used / (1024**3),
            'memory_percent': memory.percent,
            'process_memory_mb': process_memory.rss / (1024**2),
            'process_memory_percent': self.process.memory_percent()
        }
    
    def get_disk_metrics(self) -> Dict[str, float]:
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        metrics = {
            'total_disk_gb': disk.total / (1024**3),
            'used_disk_gb': disk.used / (1024**3),
            'free_disk_gb': disk.free / (1024**3),
            'disk_percent': (disk.used / disk.total) * 100
        }
        
        if disk_io:
            metrics.update({
                'disk_read_mb': disk_io.read_bytes / (1024**2),
                'disk_write_mb': disk_io.write_bytes / (1024**2),
                'disk_read_count': disk_io.read_count,
                'disk_write_count': disk_io.write_count
            })
        
        return metrics
    
    def get_network_metrics(self) -> Dict[str, float]:
        network = psutil.net_io_counters()
        
        if network:
            return {
                'bytes_sent_mb': network.bytes_sent / (1024**2),
                'bytes_recv_mb': network.bytes_recv / (1024**2),
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv,
                'errors_in': network.errin,
                'errors_out': network.errout
            }
        
        return {}


class GPUMetrics:
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
    
    def get_gpu_metrics(self) -> Dict[str, Any]:
        if not self.cuda_available:
            return {'cuda_available': False}
        
        metrics = {
            'cuda_available': True,
            'device_count': self.device_count,
            'devices': {}
        }
        
        for i in range(self.device_count):
            device_props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_cached = torch.cuda.memory_reserved(i)
            
            metrics['devices'][f'gpu_{i}'] = {
                'name': device_props.name,
                'total_memory_gb': device_props.total_memory / (1024**3),
                'allocated_memory_mb': memory_allocated / (1024**2),
                'cached_memory_mb': memory_cached / (1024**2),
                'memory_utilization': (memory_allocated / device_props.total_memory) * 100,
                'compute_capability': f"{device_props.major}.{device_props.minor}"
            }
        
        return metrics


class PerformanceMonitor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        self.system_metrics = SystemMetrics()
        self.gpu_metrics = GPUMetrics()
        
        self.metrics_history = defaultdict(lambda: deque(maxlen=self.config['history_size']))
        self.alerts = []
        self.callbacks = {}
        
        self.is_monitoring = False
        self.monitor_thread = None
        
        self.lock = threading.RLock()
        
    def _default_config(self) -> Dict[str, Any]:
        return {
            'monitoring_interval': 5.0,
            'history_size': 1000,
            'enable_alerts': True,
            'alert_thresholds': {
                'cpu_percent': 90.0,
                'memory_percent': 90.0,
                'gpu_memory_percent': 95.0,
                'disk_percent': 95.0
            },
            'metrics_to_collect': [
                'cpu', 'memory', 'disk', 'network', 'gpu'
            ],
            'save_to_file': False,
            'metrics_file': './logs/performance_metrics.json'
        }
    
    def start_monitoring(self):
        with self.lock:
            if self.is_monitoring:
                return
            
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        with self.lock:
            self.is_monitoring = False
            
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=2.0)
    
    def _monitoring_loop(self):
        while self.is_monitoring:
            try:
                metrics = self.collect_metrics()
                
                with self.lock:
                    self._store_metrics(metrics)
                    
                    if self.config['enable_alerts']:
                        self._check_alerts(metrics)
                    
                    if self.config['save_to_file']:
                        self._save_metrics_to_file(metrics)
                    
                    self._trigger_callbacks(metrics)
                
                time.sleep(self.config['monitoring_interval'])
                
            except Exception as e:
                print(f"Performance monitoring error: {e}")
                time.sleep(1.0)
    
    def collect_metrics(self) -> Dict[str, Any]:
        timestamp = time.time()
        metrics = {'timestamp': timestamp}
        
        if 'cpu' in self.config['metrics_to_collect']:
            metrics['cpu'] = self.system_metrics.get_cpu_metrics()
        
        if 'memory' in self.config['metrics_to_collect']:
            metrics['memory'] = self.system_metrics.get_memory_metrics()
        
        if 'disk' in self.config['metrics_to_collect']:
            metrics['disk'] = self.system_metrics.get_disk_metrics()
        
        if 'network' in self.config['metrics_to_collect']:
            metrics['network'] = self.system_metrics.get_network_metrics()
        
        if 'gpu' in self.config['metrics_to_collect']:
            metrics['gpu'] = self.gpu_metrics.get_gpu_metrics()
        
        return metrics
    
    def _store_metrics(self, metrics: Dict[str, Any]):
        for category, data in metrics.items():
            if category == 'timestamp':
                continue
                
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        metric_key = f"{category}.{key}"
                        self.metrics_history[metric_key].append({
                            'timestamp': metrics['timestamp'],
                            'value': value
                        })
            elif isinstance(data, (int, float)):
                self.metrics_history[category].append({
                    'timestamp': metrics['timestamp'],
                    'value': data
                })
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        thresholds = self.config['alert_thresholds']
        
        cpu_percent = metrics.get('cpu', {}).get('cpu_percent', 0)
        if cpu_percent > thresholds.get('cpu_percent', 90):
            self._create_alert('high_cpu', f"CPU usage: {cpu_percent:.1f}%", metrics['timestamp'])
        
        memory_percent = metrics.get('memory', {}).get('memory_percent', 0)
        if memory_percent > thresholds.get('memory_percent', 90):
            self._create_alert('high_memory', f"Memory usage: {memory_percent:.1f}%", metrics['timestamp'])
        
        disk_percent = metrics.get('disk', {}).get('disk_percent', 0)
        if disk_percent > thresholds.get('disk_percent', 95):
            self._create_alert('high_disk', f"Disk usage: {disk_percent:.1f}%", metrics['timestamp'])
        
        gpu_metrics = metrics.get('gpu', {})
        if gpu_metrics.get('cuda_available', False):
            for device_id, device_info in gpu_metrics.get('devices', {}).items():
                gpu_memory_percent = device_info.get('memory_utilization', 0)
                if gpu_memory_percent > thresholds.get('gpu_memory_percent', 95):
                    self._create_alert(
                        'high_gpu_memory',
                        f"GPU {device_id} memory usage: {gpu_memory_percent:.1f}%",
                        metrics['timestamp']
                    )
    
    def _create_alert(self, alert_type: str, message: str, timestamp: float):
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': timestamp,
            'resolved': False
        }
        
        self.alerts.append(alert)
        
        if len(self.alerts) > 100:
            self.alerts.pop(0)
        
        print(f"ALERT [{alert_type}]: {message}")
    
    def _save_metrics_to_file(self, metrics: Dict[str, Any]):
        try:
            import os
            os.makedirs(os.path.dirname(self.config['metrics_file']), exist_ok=True)
            
            with open(self.config['metrics_file'], 'a') as f:
                json.dump(metrics, f)
                f.write('\n')
        except Exception as e:
            print(f"Failed to save metrics to file: {e}")
    
    def _trigger_callbacks(self, metrics: Dict[str, Any]):
        for callback_name, callback_func in self.callbacks.items():
            try:
                callback_func(metrics)
            except Exception as e:
                print(f"Callback '{callback_name}' error: {e}")
    
    def register_callback(self, name: str, callback: Callable[[Dict[str, Any]], None]):
        with self.lock:
            self.callbacks[name] = callback
    
    def unregister_callback(self, name: str):
        with self.lock:
            if name in self.callbacks:
                del self.callbacks[name]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        return self.collect_metrics()
    
    def get_metric_history(self, metric_name: str, duration_seconds: Optional[int] = None) -> List[Dict[str, Any]]:
        with self.lock:
            if metric_name not in self.metrics_history:
                return []
            
            history = list(self.metrics_history[metric_name])
            
            if duration_seconds:
                cutoff_time = time.time() - duration_seconds
                history = [h for h in history if h['timestamp'] >= cutoff_time]
            
            return history
    
    def get_metric_statistics(self, metric_name: str, duration_seconds: Optional[int] = None) -> Dict[str, float]:
        history = self.get_metric_history(metric_name, duration_seconds)
        
        if not history:
            return {}
        
        values = [h['value'] for h in history]
        
        return {
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'median': sorted(values)[len(values) // 2],
            'std': np.std(values) if len(values) > 1 else 0.0,
            'count': len(values)
        }
    
    def get_alerts(self, unresolved_only: bool = True) -> List[Dict[str, Any]]:
        with self.lock:
            if unresolved_only:
                return [alert for alert in self.alerts if not alert['resolved']]
            return list(self.alerts)
    
    def resolve_alert(self, alert_index: int):
        with self.lock:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index]['resolved'] = True
    
    def clear_alerts(self):
        with self.lock:
            self.alerts.clear()
    
    def generate_performance_report(self, duration_seconds: int = 3600) -> Dict[str, Any]:
        report = {
            'report_timestamp': time.time(),
            'duration_seconds': duration_seconds,
            'metrics_summary': {},
            'alerts_summary': {
                'total_alerts': len(self.alerts),
                'unresolved_alerts': len([a for a in self.alerts if not a['resolved']])
            }
        }
        
        for metric_name in self.metrics_history.keys():
            stats = self.get_metric_statistics(metric_name, duration_seconds)
            if stats:
                report['metrics_summary'][metric_name] = stats
        
        return report
    
    def __enter__(self):
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()
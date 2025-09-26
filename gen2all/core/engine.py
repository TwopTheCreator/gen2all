import torch
import numpy as np
import threading
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, Future
import multiprocessing as mp
from collections import defaultdict, deque
import gc
import psutil

from .neural_core import NeuralCore, ParallelNeuralCore
from .memory_manager import MemoryManager
from .token_processor import TokenProcessor


class Gen2AllEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        self.memory_manager = MemoryManager(self.config.get('memory_config', {}))
        self.token_processor = TokenProcessor(self.config.get('token_config', {}))
        
        self.model = None
        self.parallel_model = None
        self._initialize_models()
        
        self.generation_cache = {}
        self.context_manager = ContextManager()
        self.inference_pool = ThreadPoolExecutor(max_workers=self.config['max_concurrent_requests'])
        
        self.stats = {
            'requests_processed': 0,
            'tokens_generated': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'contexts_created': 0,
            'memory_optimizations': 0,
            'start_time': time.time()
        }
        
        self.lock = threading.RLock()
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
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
                'repetition_penalty': 1.1,
                'do_sample': True
            },
            'max_concurrent_requests': 16,
            'enable_caching': True,
            'cache_size': 1000,
            'context_window': 8192,
            'batch_size': 8,
            'enable_parallel_processing': True,
            'memory_optimization': True,
            'monitoring_interval': 30
        }
    
    def _initialize_models(self):
        model_config = self.config['model_config']
        
        if self.num_gpus > 1 and self.config['enable_parallel_processing']:
            self.parallel_model = ParallelNeuralCore(model_config, self.num_gpus)
        else:
            self.model = NeuralCore(**model_config)
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)
    
    def load_model(self, checkpoint_path: str):
        if self.parallel_model:
            self.parallel_model.load_checkpoint(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        with self.lock:
            self.stats['model_loaded'] = time.time()
    
    def save_model(self, checkpoint_path: str):
        if self.parallel_model:
            self.parallel_model.save_checkpoint(checkpoint_path)
        else:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'stats': self.stats
            }
            torch.save(checkpoint, checkpoint_path)
    
    def create_context(self, context_id: Optional[str] = None, 
                      system_message: Optional[str] = None) -> str:
        if context_id is None:
            context_id = f"ctx_{int(time.time() * 1000000)}"
        
        context_data = {
            'conversation_history': [],
            'system_message': system_message,
            'token_count': 0,
            'created_at': time.time(),
            'last_used': time.time()
        }
        
        if system_message:
            system_tokens = self.token_processor.process_text(system_message)
            context_data['conversation_history'].append({
                'role': 'system',
                'content': system_message,
                'tokens': system_tokens['input_ids']
            })
            context_data['token_count'] += len(system_tokens['input_ids'])
        
        self.context_manager.create_context(context_id, context_data)
        self.memory_manager.allocate_context(context_id, 1024 * 1024)
        
        with self.lock:
            self.stats['contexts_created'] += 1
        
        return context_id
    
    def generate(self, prompt: str, context_id: Optional[str] = None, 
                generation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        
        generation_config = {**self.config['generation_config'], **(generation_config or {})}
        
        if context_id:
            context = self.context_manager.get_context(context_id)
            if context is None:
                raise ValueError(f"Context {context_id} not found")
        else:
            context_id = self.create_context()
            context = self.context_manager.get_context(context_id)
        
        cache_key = self._generate_cache_key(prompt, context_id, generation_config)
        
        if self.config['enable_caching'] and cache_key in self.generation_cache:
            with self.lock:
                self.stats['cache_hits'] += 1
            return self.generation_cache[cache_key]
        
        with self.lock:
            self.stats['cache_misses'] += 1
        
        try:
            result = self._generate_internal(prompt, context, generation_config)
            
            if self.config['enable_caching']:
                if len(self.generation_cache) >= self.config['cache_size']:
                    oldest_key = next(iter(self.generation_cache))
                    del self.generation_cache[oldest_key]
                self.generation_cache[cache_key] = result
            
            self._update_context(context_id, prompt, result['generated_text'])
            
            with self.lock:
                self.stats['requests_processed'] += 1
                self.stats['tokens_generated'] += result['token_count']
            
            return result
            
        except Exception as e:
            return {
                'generated_text': f"Error during generation: {str(e)}",
                'token_count': 0,
                'generation_time': 0,
                'context_id': context_id,
                'success': False,
                'error': str(e)
            }
    
    def _generate_internal(self, prompt: str, context: Dict[str, Any], 
                          generation_config: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        
        messages = context['conversation_history'] + [{
            'role': 'user',
            'content': prompt
        }]
        
        input_data = self.token_processor.process_conversation(messages)
        input_ids = torch.tensor([input_data['input_ids']], device=self.device)
        attention_mask = torch.tensor([input_data['attention_mask']], device=self.device)
        
        cached_tokens = self.memory_manager.get_cached_tokens(
            context['id'] if 'id' in context else 'default', 
            input_data['input_ids']
        )
        
        if cached_tokens is not None:
            input_embeddings = torch.tensor(cached_tokens, device=self.device)
        else:
            input_embeddings = None
        
        max_length = min(
            generation_config['max_length'],
            self.config['context_window'] - len(input_data['input_ids'])
        )
        
        if self.parallel_model and input_ids.size(0) >= self.config['batch_size']:
            generated_ids = self._parallel_generate(
                input_ids, attention_mask, generation_config, max_length
            )
        else:
            with torch.no_grad():
                self.model.eval()
                generated_ids = self.model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=generation_config['temperature'],
                    top_k=generation_config['top_k'],
                    top_p=generation_config['top_p']
                )
        
        generated_tokens = generated_ids[0, len(input_data['input_ids']):].tolist()
        generated_text = self.token_processor.decode_tokens(generated_tokens)
        
        if input_embeddings is None:
            context_id = context.get('id', 'default')
            self.memory_manager.cache_tokens(
                context_id, 
                input_data['input_ids'], 
                input_ids.cpu().numpy()
            )
        
        generation_time = time.time() - start_time
        
        result = {
            'generated_text': generated_text,
            'token_count': len(generated_tokens),
            'generation_time': generation_time,
            'context_id': context.get('id'),
            'success': True,
            'input_token_count': len(input_data['input_ids']),
            'total_tokens': len(input_data['input_ids']) + len(generated_tokens)
        }
        
        return result
    
    def _parallel_generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                          generation_config: Dict[str, Any], max_length: int) -> torch.Tensor:
        batch_size = input_ids.size(0)
        batches = []
        
        for i in range(0, batch_size, self.config['batch_size']):
            batch = {
                'input_ids': input_ids[i:i + self.config['batch_size']],
                'attention_mask': attention_mask[i:i + self.config['batch_size']]
            }
            batches.append(batch)
        
        results = self.parallel_model.parallel_forward(batches)
        
        generated_sequences = []
        for batch_result in results:
            for sequence in batch_result['logits']:
                generated_sequences.append(sequence)
        
        return torch.stack(generated_sequences)
    
    def _generate_cache_key(self, prompt: str, context_id: str, 
                           generation_config: Dict[str, Any]) -> str:
        import hashlib
        
        cache_data = {
            'prompt': prompt,
            'context_id': context_id,
            'config': generation_config
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _update_context(self, context_id: str, prompt: str, generated_text: str):
        context = self.context_manager.get_context(context_id)
        if context:
            context['conversation_history'].extend([
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': generated_text}
            ])
            context['last_used'] = time.time()
            
            prompt_tokens = self.token_processor.process_text(prompt)
            response_tokens = self.token_processor.process_text(generated_text)
            context['token_count'] += len(prompt_tokens['input_ids']) + len(response_tokens['input_ids'])
            
            if context['token_count'] > self.config['context_window']:
                self._trim_context(context)
    
    def _trim_context(self, context: Dict[str, Any]):
        while context['token_count'] > self.config['context_window'] * 0.75:
            if len(context['conversation_history']) <= 2:
                break
            
            removed_message = context['conversation_history'].pop(1)
            if 'tokens' in removed_message:
                context['token_count'] -= len(removed_message['tokens'])
            else:
                token_data = self.token_processor.process_text(removed_message['content'])
                context['token_count'] -= len(token_data['input_ids'])
    
    def batch_generate(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        futures = []
        
        for request in requests:
            future = self.inference_pool.submit(
                self.generate,
                request['prompt'],
                request.get('context_id'),
                request.get('generation_config')
            )
            futures.append(future)
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as e:
                results.append({
                    'generated_text': f"Error: {str(e)}",
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def get_context_info(self, context_id: str) -> Optional[Dict[str, Any]]:
        context = self.context_manager.get_context(context_id)
        if context:
            return {
                'context_id': context_id,
                'message_count': len(context['conversation_history']),
                'token_count': context['token_count'],
                'created_at': context['created_at'],
                'last_used': context['last_used'],
                'system_message': context.get('system_message')
            }
        return None
    
    def clear_context(self, context_id: str) -> bool:
        success = self.context_manager.remove_context(context_id)
        if success:
            self.memory_manager.save_context(context_id, compress=True)
        return success
    
    def list_contexts(self) -> List[str]:
        return self.context_manager.list_contexts()
    
    def get_engine_stats(self) -> Dict[str, Any]:
        with self.lock:
            current_time = time.time()
            uptime = current_time - self.stats['start_time']
            
            system_stats = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'gpu_memory': self._get_gpu_memory_info() if torch.cuda.is_available() else None
            }
            
            performance_stats = {
                'requests_per_second': self.stats['requests_processed'] / uptime if uptime > 0 else 0,
                'tokens_per_second': self.stats['tokens_generated'] / uptime if uptime > 0 else 0,
                'cache_hit_rate': (self.stats['cache_hits'] / 
                                 (self.stats['cache_hits'] + self.stats['cache_misses'])) 
                                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
            }
            
            return {
                'engine_stats': dict(self.stats),
                'system_stats': system_stats,
                'performance_stats': performance_stats,
                'memory_stats': self.memory_manager.get_memory_stats(),
                'context_stats': self.context_manager.get_stats(),
                'uptime_seconds': uptime
            }
    
    def _get_gpu_memory_info(self) -> Dict[str, Any]:
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_cached = torch.cuda.memory_reserved(i)
            gpu_info[f'gpu_{i}'] = {
                'allocated_mb': memory_allocated / 1024 / 1024,
                'cached_mb': memory_cached / 1024 / 1024
            }
        return gpu_info
    
    def _monitoring_loop(self):
        while self.is_running:
            try:
                time.sleep(self.config['monitoring_interval'])
                
                if self.config['memory_optimization']:
                    self.memory_manager.optimize_memory()
                
                self._cleanup_old_contexts()
                self._cleanup_generation_cache()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                gc.collect()
                
            except Exception as e:
                print(f"Monitoring error: {e}")
    
    def _cleanup_old_contexts(self):
        current_time = time.time()
        old_contexts = []
        
        for context_id in self.context_manager.list_contexts():
            context = self.context_manager.get_context(context_id)
            if context and current_time - context['last_used'] > 3600:
                old_contexts.append(context_id)
        
        for context_id in old_contexts:
            self.clear_context(context_id)
    
    def _cleanup_generation_cache(self):
        if len(self.generation_cache) > self.config['cache_size']:
            items_to_remove = len(self.generation_cache) - self.config['cache_size']
            keys_to_remove = list(self.generation_cache.keys())[:items_to_remove]
            
            for key in keys_to_remove:
                del self.generation_cache[key]
    
    def shutdown(self):
        self.is_running = False
        self.inference_pool.shutdown(wait=True)
        
        for context_id in self.context_manager.list_contexts():
            self.memory_manager.save_context(context_id, compress=True)


class ContextManager:
    def __init__(self):
        self.contexts = {}
        self.lock = threading.RLock()
    
    def create_context(self, context_id: str, context_data: Dict[str, Any]):
        with self.lock:
            context_data['id'] = context_id
            self.contexts[context_id] = context_data
    
    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            return self.contexts.get(context_id)
    
    def remove_context(self, context_id: str) -> bool:
        with self.lock:
            if context_id in self.contexts:
                del self.contexts[context_id]
                return True
            return False
    
    def list_contexts(self) -> List[str]:
        with self.lock:
            return list(self.contexts.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                'active_contexts': len(self.contexts),
                'total_memory_usage': sum(
                    ctx.get('token_count', 0) for ctx in self.contexts.values()
                )
            }
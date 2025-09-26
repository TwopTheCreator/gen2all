import requests
import json
import time
from typing import Dict, List, Any, Optional, Union
import threading
from concurrent.futures import ThreadPoolExecutor, Future
import logging


class Gen2AllClient:
    def __init__(self, api_key: str, base_url: str = "http://localhost:8000", 
                 timeout: int = 300, max_retries: int = 3):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'Gen2All-Python-Client/2.1.4'
        })
        
        self.context_cache = {}
        self.stats = {
            'requests_made': 0,
            'tokens_generated': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.lock = threading.RLock()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     params: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                with self.lock:
                    self.stats['requests_made'] += 1
                
                response = self.session.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 1))
                    self.logger.warning(f"Rate limited. Waiting {retry_after} seconds.")
                    time.sleep(retry_after)
                    continue
                elif response.status_code in [500, 502, 503, 504] and attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Server error {response.status_code}. Retrying in {wait_time}s.")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    self.logger.warning(f"Request timeout. Retrying... ({attempt + 1}/{self.max_retries})")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise
            except requests.exceptions.RequestException as e:
                with self.lock:
                    self.stats['errors'] += 1
                
                if attempt < self.max_retries:
                    self.logger.warning(f"Request failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    raise
        
        raise Exception(f"Failed to complete request after {self.max_retries + 1} attempts")
    
    def generate(self, prompt: str, context_id: Optional[str] = None,
                max_length: Optional[int] = None, temperature: Optional[float] = None,
                top_k: Optional[int] = None, top_p: Optional[float] = None,
                repetition_penalty: Optional[float] = None,
                stop_sequences: Optional[List[str]] = None,
                system_message: Optional[str] = None) -> Dict[str, Any]:
        
        request_data = {'prompt': prompt}
        
        if context_id:
            request_data['context_id'] = context_id
        if max_length is not None:
            request_data['max_length'] = max_length
        if temperature is not None:
            request_data['temperature'] = temperature
        if top_k is not None:
            request_data['top_k'] = top_k
        if top_p is not None:
            request_data['top_p'] = top_p
        if repetition_penalty is not None:
            request_data['repetition_penalty'] = repetition_penalty
        if stop_sequences:
            request_data['stop_sequences'] = stop_sequences
        if system_message:
            request_data['system_message'] = system_message
        
        result = self._make_request('POST', '/generate', data=request_data)
        
        with self.lock:
            if result.get('success', False):
                self.stats['tokens_generated'] += result.get('token_count', 0)
        
        return result
    
    def generate_async(self, prompt: str, context_id: Optional[str] = None,
                      **kwargs) -> Future[Dict[str, Any]]:
        return self.executor.submit(self.generate, prompt, context_id, **kwargs)
    
    def batch_generate(self, prompts: List[str], contexts: Optional[List[str]] = None,
                      generation_config: Optional[Dict[str, Any]] = None,
                      parallel_processing: bool = True) -> List[Dict[str, Any]]:
        
        requests = []
        for i, prompt in enumerate(prompts):
            request_data = {'prompt': prompt}
            
            if contexts and i < len(contexts):
                request_data['context_id'] = contexts[i]
            
            if generation_config:
                request_data.update(generation_config)
            
            requests.append(request_data)
        
        batch_request = {
            'requests': requests,
            'parallel_processing': parallel_processing
        }
        
        result = self._make_request('POST', '/batch_generate', data=batch_request)
        
        with self.lock:
            for response in result.get('results', []):
                if response.get('success', False):
                    self.stats['tokens_generated'] += response.get('token_count', 0)
        
        return result.get('results', [])
    
    def batch_generate_async(self, prompts: List[str], **kwargs) -> Future[List[Dict[str, Any]]]:
        return self.executor.submit(self.batch_generate, prompts, **kwargs)
    
    def create_context(self, system_message: Optional[str] = None,
                      context_id: Optional[str] = None) -> str:
        request_data = {}
        
        if system_message:
            request_data['system_message'] = system_message
        if context_id:
            request_data['context_id'] = context_id
        
        result = self._make_request('POST', '/contexts', data=request_data)
        
        created_context_id = result['context_id']
        
        with self.lock:
            self.context_cache[created_context_id] = {
                'system_message': system_message,
                'created_at': time.time()
            }
        
        return created_context_id
    
    def get_context_info(self, context_id: str) -> Optional[Dict[str, Any]]:
        try:
            return self._make_request('GET', f'/contexts/{context_id}')
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise
    
    def clear_context(self, context_id: str) -> bool:
        try:
            self._make_request('DELETE', f'/contexts/{context_id}')
            
            with self.lock:
                if context_id in self.context_cache:
                    del self.context_cache[context_id]
            
            return True
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return False
            raise
    
    def list_contexts(self) -> List[str]:
        result = self._make_request('GET', '/contexts')
        return result.get('contexts', [])
    
    def get_stats(self) -> Dict[str, Any]:
        server_stats = self._make_request('GET', '/stats')
        
        with self.lock:
            client_stats = dict(self.stats)
            client_stats['uptime_seconds'] = time.time() - client_stats['start_time']
        
        return {
            'server_stats': server_stats,
            'client_stats': client_stats
        }
    
    def chat(self, message: str, context_id: Optional[str] = None,
            system_message: Optional[str] = None, **generation_kwargs) -> Dict[str, Any]:
        
        if context_id is None:
            context_id = self.create_context(system_message=system_message)
        
        return self.generate(
            prompt=message,
            context_id=context_id,
            **generation_kwargs
        )
    
    def chat_stream(self, messages: List[Dict[str, str]], context_id: Optional[str] = None,
                   **generation_kwargs):
        if context_id is None:
            context_id = self.create_context()
        
        for message in messages:
            if message['role'] == 'user':
                response = self.chat(message['content'], context_id, **generation_kwargs)
                yield {
                    'role': 'assistant',
                    'content': response['generated_text'],
                    'context_id': context_id,
                    'metadata': response
                }
    
    def multi_turn_conversation(self, messages: List[Dict[str, str]],
                               system_message: Optional[str] = None,
                               **generation_kwargs) -> List[Dict[str, Any]]:
        
        context_id = self.create_context(system_message=system_message)
        responses = []
        
        for message in messages:
            if message['role'] == 'user':
                response = self.chat(
                    message['content'], 
                    context_id=context_id,
                    **generation_kwargs
                )
                responses.append(response)
        
        return responses
    
    def completion(self, text: str, max_length: int = 100, **kwargs) -> str:
        response = self.generate(
            prompt=text,
            max_length=max_length,
            **kwargs
        )
        return response.get('generated_text', '')
    
    def summarize(self, text: str, max_length: int = 150, **kwargs) -> str:
        prompt = f"Please summarize the following text in {max_length} tokens or less:\n\n{text}\n\nSummary:"
        
        response = self.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=0.7,
            **kwargs
        )
        
        return response.get('generated_text', '').strip()
    
    def answer_question(self, question: str, context: Optional[str] = None, **kwargs) -> str:
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        response = self.generate(
            prompt=prompt,
            temperature=0.5,
            **kwargs
        )
        
        return response.get('generated_text', '').strip()
    
    def translate(self, text: str, target_language: str, **kwargs) -> str:
        prompt = f"Translate the following text to {target_language}:\n\n{text}\n\nTranslation:"
        
        response = self.generate(
            prompt=prompt,
            temperature=0.3,
            **kwargs
        )
        
        return response.get('generated_text', '').strip()
    
    def code_completion(self, code: str, language: Optional[str] = None, **kwargs) -> str:
        if language:
            prompt = f"Complete the following {language} code:\n\n{code}"
        else:
            prompt = f"Complete the following code:\n\n{code}"
        
        response = self.generate(
            prompt=prompt,
            temperature=0.2,
            max_length=500,
            **kwargs
        )
        
        return response.get('generated_text', '').strip()
    
    def creative_writing(self, prompt: str, style: Optional[str] = None, **kwargs) -> str:
        if style:
            full_prompt = f"Write in {style} style: {prompt}"
        else:
            full_prompt = prompt
        
        response = self.generate(
            prompt=full_prompt,
            temperature=0.9,
            max_length=1000,
            **kwargs
        )
        
        return response.get('generated_text', '').strip()
    
    def health_check(self) -> bool:
        try:
            response = self._make_request('GET', '/health')
            return response.get('status') == 'healthy'
        except:
            return False
    
    def close(self):
        self.executor.shutdown(wait=True)
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import asyncio
import time
import threading
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
import json
import hashlib
import secrets
from concurrent.futures import ThreadPoolExecutor
import logging
import redis
import sqlite3
from contextlib import asynccontextmanager

from ..core.engine import Gen2AllEngine
from .auth import AuthManager
from .rate_limiter import RateLimiter


class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="The input prompt for generation")
    context_id: Optional[str] = Field(None, description="Optional context ID for conversation")
    max_length: Optional[int] = Field(512, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.8, description="Sampling temperature")
    top_k: Optional[int] = Field(50, description="Top-k sampling")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling")
    repetition_penalty: Optional[float] = Field(1.1, description="Repetition penalty")
    stop_sequences: Optional[List[str]] = Field(None, description="Stop sequences")
    system_message: Optional[str] = Field(None, description="System message for context")


class BatchGenerationRequest(BaseModel):
    requests: List[GenerationRequest] = Field(..., description="List of generation requests")
    parallel_processing: Optional[bool] = Field(True, description="Enable parallel processing")


class ContextCreationRequest(BaseModel):
    system_message: Optional[str] = Field(None, description="Initial system message")
    context_id: Optional[str] = Field(None, description="Custom context ID")


class GenerationResponse(BaseModel):
    generated_text: str
    token_count: int
    generation_time: float
    context_id: Optional[str] = None
    success: bool = True
    input_token_count: Optional[int] = None
    total_tokens: Optional[int] = None


class BatchGenerationResponse(BaseModel):
    results: List[GenerationResponse]
    total_time: float
    success_count: int
    error_count: int


class ContextInfo(BaseModel):
    context_id: str
    message_count: int
    token_count: int
    created_at: float
    last_used: float
    system_message: Optional[str] = None


class EngineStats(BaseModel):
    requests_processed: int
    tokens_generated: int
    contexts_created: int
    uptime_seconds: float
    cache_hit_rate: float
    system_stats: Dict[str, Any]
    performance_stats: Dict[str, Any]


class Gen2AllServer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        self.app = FastAPI(
            title="Gen2All API",
            description="Advanced AI Generation API with unlimited quota",
            version="2.1.4",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        self.engine = Gen2AllEngine(self.config.get('engine_config', {}))
        self.auth_manager = AuthManager(self.config.get('auth_config', {}))
        self.rate_limiter = RateLimiter(self.config.get('rate_limit_config', {}))
        
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_workers'])
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'start_time': time.time()
        }
        
        self._setup_middleware()
        self._setup_routes()
        
        self.lock = threading.RLock()
    
    def _default_config(self) -> Dict[str, Any]:
        return {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 4,
            'max_workers': 32,
            'enable_auth': True,
            'enable_rate_limiting': True,
            'cors_origins': ['*'],
            'log_level': 'INFO',
            'request_timeout': 300,
            'max_request_size': 1024 * 1024,
            'engine_config': {},
            'auth_config': {},
            'rate_limit_config': {}
        }
    
    def _setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config['cors_origins'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*'],
        )
        
        @self.app.middleware("http")
        async def request_logging(request: Request, call_next):
            start_time = time.time()
            
            with self.lock:
                self.request_stats['total_requests'] += 1
            
            try:
                response = await call_next(request)
                
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(process_time)
                
                with self.lock:
                    if response.status_code < 400:
                        self.request_stats['successful_requests'] += 1
                    else:
                        self.request_stats['failed_requests'] += 1
                
                return response
                
            except Exception as e:
                with self.lock:
                    self.request_stats['failed_requests'] += 1
                raise
    
    def _setup_routes(self):
        security = HTTPBearer() if self.config['enable_auth'] else None
        
        @self.app.get("/")
        async def root():
            return {
                "service": "Gen2All API",
                "version": "2.1.4",
                "status": "operational",
                "features": [
                    "unlimited_quota",
                    "high_memory_capacity",
                    "advanced_context_management",
                    "parallel_processing",
                    "intelligent_caching"
                ]
            }
        
        @self.app.post("/generate", response_model=GenerationResponse)
        async def generate(
            request: GenerationRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security) if security else None
        ):
            if self.config['enable_auth']:
                api_key = credentials.credentials
                if not self.auth_manager.validate_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
                
                if self.config['enable_rate_limiting']:
                    if not self.rate_limiter.check_rate_limit(api_key):
                        raise HTTPException(status_code=429, detail="Rate limit exceeded")
            
            try:
                generation_config = {
                    'max_length': request.max_length,
                    'temperature': request.temperature,
                    'top_k': request.top_k,
                    'top_p': request.top_p,
                    'repetition_penalty': request.repetition_penalty
                }
                
                context_id = request.context_id
                if not context_id and request.system_message:
                    context_id = self.engine.create_context(system_message=request.system_message)
                
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.engine.generate,
                    request.prompt,
                    context_id,
                    generation_config
                )
                
                return GenerationResponse(**result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/batch_generate", response_model=BatchGenerationResponse)
        async def batch_generate(
            request: BatchGenerationRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security) if security else None
        ):
            if self.config['enable_auth']:
                api_key = credentials.credentials
                if not self.auth_manager.validate_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
            
            start_time = time.time()
            
            try:
                batch_requests = []
                for req in request.requests:
                    generation_config = {
                        'max_length': req.max_length,
                        'temperature': req.temperature,
                        'top_k': req.top_k,
                        'top_p': req.top_p,
                        'repetition_penalty': req.repetition_penalty
                    }
                    
                    batch_requests.append({
                        'prompt': req.prompt,
                        'context_id': req.context_id,
                        'generation_config': generation_config
                    })
                
                results = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self.engine.batch_generate,
                    batch_requests
                )
                
                generation_responses = [GenerationResponse(**result) for result in results]
                success_count = sum(1 for r in generation_responses if r.success)
                error_count = len(generation_responses) - success_count
                
                return BatchGenerationResponse(
                    results=generation_responses,
                    total_time=time.time() - start_time,
                    success_count=success_count,
                    error_count=error_count
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/contexts", response_model=Dict[str, str])
        async def create_context(
            request: ContextCreationRequest,
            credentials: HTTPAuthorizationCredentials = Depends(security) if security else None
        ):
            if self.config['enable_auth']:
                api_key = credentials.credentials
                if not self.auth_manager.validate_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
            
            try:
                context_id = self.engine.create_context(
                    context_id=request.context_id,
                    system_message=request.system_message
                )
                
                return {"context_id": context_id}
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/contexts/{context_id}", response_model=ContextInfo)
        async def get_context_info(
            context_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(security) if security else None
        ):
            if self.config['enable_auth']:
                api_key = credentials.credentials
                if not self.auth_manager.validate_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
            
            context_info = self.engine.get_context_info(context_id)
            if not context_info:
                raise HTTPException(status_code=404, detail="Context not found")
            
            return ContextInfo(**context_info)
        
        @self.app.delete("/contexts/{context_id}")
        async def clear_context(
            context_id: str,
            credentials: HTTPAuthorizationCredentials = Depends(security) if security else None
        ):
            if self.config['enable_auth']:
                api_key = credentials.credentials
                if not self.auth_manager.validate_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
            
            success = self.engine.clear_context(context_id)
            if not success:
                raise HTTPException(status_code=404, detail="Context not found")
            
            return {"message": "Context cleared successfully"}
        
        @self.app.get("/contexts")
        async def list_contexts(
            credentials: HTTPAuthorizationCredentials = Depends(security) if security else None
        ):
            if self.config['enable_auth']:
                api_key = credentials.credentials
                if not self.auth_manager.validate_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
            
            contexts = self.engine.list_contexts()
            return {"contexts": contexts}
        
        @self.app.get("/stats", response_model=EngineStats)
        async def get_stats(
            credentials: HTTPAuthorizationCredentials = Depends(security) if security else None
        ):
            if self.config['enable_auth']:
                api_key = credentials.credentials
                if not self.auth_manager.validate_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
            
            engine_stats = self.engine.get_engine_stats()
            
            with self.lock:
                api_stats = dict(self.request_stats)
            
            combined_stats = {
                **engine_stats['engine_stats'],
                'api_requests_total': api_stats['total_requests'],
                'api_requests_successful': api_stats['successful_requests'],
                'api_requests_failed': api_stats['failed_requests'],
                'api_uptime_seconds': time.time() - api_stats['start_time'],
                'system_stats': engine_stats['system_stats'],
                'performance_stats': engine_stats['performance_stats'],
                'cache_hit_rate': engine_stats['performance_stats']['cache_hit_rate']
            }
            
            return EngineStats(**combined_stats)
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "2.1.4",
                "memory_usage": self.engine.get_engine_stats()['system_stats']['memory_percent'],
                "active_contexts": len(self.engine.list_contexts())
            }
        
        @self.app.post("/admin/optimize")
        async def optimize_memory(
            credentials: HTTPAuthorizationCredentials = Depends(security) if security else None
        ):
            if self.config['enable_auth']:
                api_key = credentials.credentials
                if not self.auth_manager.validate_api_key(api_key):
                    raise HTTPException(status_code=401, detail="Invalid API key")
            
            self.engine.memory_manager.optimize_memory()
            return {"message": "Memory optimization completed"}
    
    def run(self):
        logging.basicConfig(level=getattr(logging, self.config['log_level']))
        
        uvicorn.run(
            self.app,
            host=self.config['host'],
            port=self.config['port'],
            workers=1,
            access_log=True,
            use_colors=True
        )
    
    def shutdown(self):
        self.engine.shutdown()
        self.executor.shutdown(wait=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Gen2All API Server')
    parser.add_argument('--config', type=str, help='Path to server config file')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    
    args = parser.parse_args()
    
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    config.update({
        'host': args.host,
        'port': args.port,
        'workers': args.workers
    })
    
    server = Gen2AllServer(config)
    server.run()


if __name__ == "__main__":
    main()
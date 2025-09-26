import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import time
import os
from typing import Dict, List, Any, Optional, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from collections import defaultdict
import gc
import psutil

from .neural_core import NeuralCore, ParallelNeuralCore
from .memory_manager import MemoryManager
from .token_processor import TokenProcessor


class TextDataset(Dataset):
    def __init__(self, texts: List[str], token_processor: TokenProcessor, 
                 max_length: int = 8192):
        self.texts = texts
        self.token_processor = token_processor
        self.max_length = max_length
        self.tokenized_data = self._tokenize_texts()
    
    def _tokenize_texts(self):
        tokenized = []
        for text in self.texts:
            tokens = self.token_processor.process_text(
                text, 
                add_special_tokens=True, 
                return_tensors=True
            )
            if len(tokens['input_ids']) <= self.max_length:
                tokenized.append(tokens)
        return tokenized
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class AdvancedTrainer:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        self.memory_manager = MemoryManager(self.config.get('memory_config', {}))
        self.token_processor = TokenProcessor(self.config.get('token_config', {}))
        
        self.model = None
        self.parallel_model = None
        self.optimizer = None
        self.scheduler = None
        
        self.training_stats = {
            'total_steps': 0,
            'total_epochs': 0,
            'total_tokens_processed': 0,
            'training_time': 0,
            'best_loss': float('inf'),
            'learning_rate_history': [],
            'loss_history': [],
            'perplexity_history': []
        }
        
        self.lock = threading.RLock()
        
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
            'training_config': {
                'learning_rate': 1e-4,
                'weight_decay': 0.01,
                'warmup_steps': 4000,
                'max_grad_norm': 1.0,
                'gradient_accumulation_steps': 8,
                'mixed_precision': True,
                'lr_scheduler': 'cosine',
                'min_lr': 1e-6
            },
            'batch_size': 8,
            'epochs': 10,
            'save_steps': 1000,
            'eval_steps': 500,
            'logging_steps': 100,
            'output_dir': './checkpoints',
            'resume_from_checkpoint': None,
            'enable_parallel_training': True,
            'dataloader_num_workers': 4,
            'pin_memory': True
        }
    
    def initialize_model(self):
        model_config = self.config['model_config']
        
        if self.num_gpus > 1 and self.config['enable_parallel_training']:
            self.parallel_model = ParallelNeuralCore(model_config, self.num_gpus)
            self.model = self.parallel_model.models[0]
        else:
            self.model = NeuralCore(**model_config)
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model initialized with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def initialize_optimizer(self):
        training_config = self.config['training_config']
        
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'bias' not in n and 'norm' not in n],
                'weight_decay': training_config['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'bias' in n or 'norm' in n],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            lr=training_config['learning_rate'],
            betas=(0.9, 0.95),
            eps=1e-8
        )
        
        if training_config['lr_scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=training_config['warmup_steps'],
                eta_min=training_config['min_lr']
            )
        elif training_config['lr_scheduler'] == 'linear_warmup':
            self.scheduler = self._create_linear_warmup_scheduler()
    
    def _create_linear_warmup_scheduler(self):
        def lr_lambda(step):
            warmup_steps = self.config['training_config']['warmup_steps']
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def prepare_data(self, training_data: List[str], validation_data: Optional[List[str]] = None):
        print("Preparing training data...")
        
        if not hasattr(self.token_processor.tokenizer.bpe, 'encoder') or not self.token_processor.tokenizer.bpe.encoder:
            print("Training tokenizer...")
            self.token_processor.train_tokenizer(training_data)
        
        train_dataset = TextDataset(
            training_data,
            self.token_processor,
            self.config['model_config']['max_seq_length']
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['dataloader_num_workers'],
            pin_memory=self.config['pin_memory'],
            drop_last=True
        )
        
        val_dataloader = None
        if validation_data:
            val_dataset = TextDataset(
                validation_data,
                self.token_processor,
                self.config['model_config']['max_seq_length']
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=self.config['dataloader_num_workers'],
                pin_memory=self.config['pin_memory']
            )
        
        return train_dataloader, val_dataloader
    
    def train(self, training_data: List[str], validation_data: Optional[List[str]] = None):
        print("Starting training...")
        start_time = time.time()
        
        self.initialize_model()
        self.initialize_optimizer()
        
        if self.config['resume_from_checkpoint']:
            self.load_checkpoint(self.config['resume_from_checkpoint'])
        
        train_dataloader, val_dataloader = self.prepare_data(training_data, validation_data)
        
        scaler = torch.cuda.amp.GradScaler() if self.config['training_config']['mixed_precision'] else None
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        self.model.train()
        step = 0
        
        for epoch in range(self.config['epochs']):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask']
                    )
                    
                    logits = outputs['logits']
                    
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = batch['labels'][..., 1:].contiguous()
                    
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    loss = loss / self.config['training_config']['gradient_accumulation_steps']
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if (step + 1) % self.config['training_config']['gradient_accumulation_steps'] == 0:
                    if scaler:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training_config']['max_grad_norm']
                        )
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config['training_config']['max_grad_norm']
                        )
                        self.optimizer.step()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                
                step += 1
                
                if step % self.config['logging_steps'] == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    perplexity = torch.exp(loss).item() if loss.item() < 10 else float('inf')
                    
                    with self.lock:
                        self.training_stats['loss_history'].append(loss.item())
                        self.training_stats['learning_rate_history'].append(current_lr)
                        self.training_stats['perplexity_history'].append(perplexity)
                    
                    print(f"Step {step}: Loss={loss.item():.4f}, LR={current_lr:.2e}, PPL={perplexity:.2f}")
                
                if step % self.config['eval_steps'] == 0 and val_dataloader:
                    val_loss = self.evaluate(val_dataloader)
                    print(f"Validation Loss: {val_loss:.4f}")
                    self.model.train()
                
                if step % self.config['save_steps'] == 0:
                    self.save_checkpoint(f"{self.config['output_dir']}/checkpoint-{step}")
                
                if step % 1000 == 0:
                    self.memory_manager.optimize_memory()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            with self.lock:
                self.training_stats['total_epochs'] = epoch + 1
                self.training_stats['total_steps'] = step
                
                if avg_epoch_loss < self.training_stats['best_loss']:
                    self.training_stats['best_loss'] = avg_epoch_loss
                    self.save_checkpoint(f"{self.config['output_dir']}/best_model")
            
            print(f"Epoch {epoch + 1}/{self.config['epochs']} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        total_training_time = time.time() - start_time
        with self.lock:
            self.training_stats['training_time'] = total_training_time
        
        print(f"Training completed in {total_training_time:.2f} seconds")
        self.save_checkpoint(f"{self.config['output_dir']}/final_model")
        
        return self.training_stats
    
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                logits = outputs['logits']
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch['labels'][..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, path: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_stats': self.training_stats,
            'config': self.config
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path + '.pt')
        
        with open(path + '_stats.json', 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path + '.pt', map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        print(f"Checkpoint loaded from {path}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        with self.lock:
            return self.training_stats.copy()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Gen2All Model Training')
    parser.add_argument('--config', type=str, help='Path to training config file')
    parser.add_argument('--data', type=str, required=True, help='Path to training data file')
    parser.add_argument('--val_data', type=str, help='Path to validation data file')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    
    args = parser.parse_args()
    
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    config['output_dir'] = args.output_dir
    
    trainer = AdvancedTrainer(config)
    
    with open(args.data, 'r') as f:
        training_data = [line.strip() for line in f if line.strip()]
    
    validation_data = None
    if args.val_data:
        with open(args.val_data, 'r') as f:
            validation_data = [line.strip() for line in f if line.strip()]
    
    trainer.train(training_data, validation_data)


if __name__ == "__main__":
    main()
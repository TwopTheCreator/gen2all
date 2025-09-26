import os
import json
from gen2all.core.trainer import AdvancedTrainer


def prepare_training_data():
    """Prepare sample training data for demonstration"""
    training_texts = [
        "The quick brown fox jumps over the lazy dog. This is a sample sentence for training.",
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Natural language processing enables computers to understand and generate human language.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Transformers have revolutionized the field of natural language processing.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        "Large language models can generate human-like text with remarkable quality.",
        "Training neural networks requires large datasets and computational resources.",
        "Gradient descent is an optimization algorithm used to train machine learning models.",
        "Backpropagation is the process of computing gradients in neural networks.",
        "Overfitting occurs when a model performs well on training data but poorly on new data.",
        "Regularization techniques help prevent overfitting in machine learning models.",
        "Cross-validation is used to evaluate model performance on unseen data.",
        "Feature engineering involves creating informative input variables for models.",
        "Hyperparameter tuning is the process of optimizing model configuration.",
        "Ensemble methods combine multiple models to improve prediction accuracy.",
        "Convolutional neural networks are particularly effective for image processing.",
        "Recurrent neural networks can process sequences of variable length.",
        "Transfer learning allows models to leverage knowledge from pre-trained networks.",
        "Data preprocessing is crucial for training effective machine learning models.",
    ] * 50  # Repeat to create more training data
    
    validation_texts = [
        "Artificial intelligence is transforming various industries and applications.",
        "Neural networks are inspired by the structure and function of biological neurons.",
        "Supervised learning uses labeled examples to train predictive models.",
        "Unsupervised learning discovers patterns in data without explicit labels.",
        "Reinforcement learning trains agents to make decisions through trial and error.",
    ] * 10
    
    return training_texts, validation_texts


def create_training_config():
    """Create a configuration for training"""
    config = {
        'model_config': {
            'vocab_size': 32000,  # Smaller for demo
            'd_model': 512,       # Smaller for demo
            'num_layers': 12,     # Smaller for demo
            'num_heads': 8,       # Smaller for demo
            'd_ff': 2048,         # Smaller for demo
            'max_seq_length': 1024,
            'dropout': 0.1
        },
        'training_config': {
            'learning_rate': 5e-4,
            'weight_decay': 0.01,
            'warmup_steps': 1000,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 4,
            'mixed_precision': True,
            'lr_scheduler': 'cosine',
            'min_lr': 1e-6
        },
        'batch_size': 4,
        'epochs': 3,
        'save_steps': 100,
        'eval_steps': 50,
        'logging_steps': 10,
        'output_dir': './demo_checkpoints',
        'enable_parallel_training': True,
        'dataloader_num_workers': 2,
        'pin_memory': True
    }
    
    return config


def main():
    print("🚀 Gen2All Training Example")
    print("=" * 50)
    
    # Prepare training data
    print("📚 Preparing training data...")
    training_texts, validation_texts = prepare_training_data()
    print(f"✅ Training samples: {len(training_texts)}")
    print(f"✅ Validation samples: {len(validation_texts)}")
    
    # Create training configuration
    print("\n⚙️ Creating training configuration...")
    config = create_training_config()
    
    # Save configuration
    os.makedirs('./demo_checkpoints', exist_ok=True)
    with open('./demo_checkpoints/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("✅ Configuration saved")
    
    # Initialize trainer
    print("\n🤖 Initializing trainer...")
    trainer = AdvancedTrainer(config)
    
    # Start training
    print("\n🎯 Starting training process...")
    print("Note: This is a demonstration with a small model and dataset")
    print("For production, use larger models and more data")
    print("-" * 50)
    
    try:
        training_stats = trainer.train(training_texts, validation_texts)
        
        print("\n🎉 Training completed successfully!")
        print("=" * 50)
        print("📊 Training Statistics:")
        print(f"   Total epochs: {training_stats['total_epochs']}")
        print(f"   Total steps: {training_stats['total_steps']}")
        print(f"   Training time: {training_stats['training_time']:.2f} seconds")
        print(f"   Best loss: {training_stats['best_loss']:.4f}")
        print(f"   Tokens processed: {training_stats['total_tokens_processed']}")
        
        # Show loss history
        if training_stats['loss_history']:
            print(f"\n📈 Loss progression:")
            loss_history = training_stats['loss_history']
            for i in range(0, len(loss_history), max(1, len(loss_history) // 10)):
                step = (i // config['training_config']['gradient_accumulation_steps']) * config['logging_steps']
                print(f"   Step {step}: {loss_history[i]:.4f}")
        
        print(f"\n💾 Model checkpoints saved in: {config['output_dir']}")
        print("📝 Files saved:")
        print("   - final_model.pt (final model)")
        print("   - best_model.pt (best model)")
        print("   - final_model_stats.json (training statistics)")
        
        print("\n🚀 Next steps:")
        print("1. Load the trained model in your application")
        print("2. Use it with Gen2AllEngine for inference")
        print("3. Continue training with more data if needed")
        
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        print("🔄 Saving current progress...")
        trainer.save_checkpoint(f"{config['output_dir']}/interrupted_checkpoint")
        print("✅ Progress saved. You can resume training later.")
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        print("💡 Tips:")
        print("- Check your system has enough memory")
        print("- Reduce batch_size or model dimensions if needed")
        print("- Ensure PyTorch is properly installed")


if __name__ == "__main__":
    main()
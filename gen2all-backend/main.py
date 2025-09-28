"""
Gen2All - AI Backend Management System
Main entry point for the application
"""
import sys
import argparse
import threading
import time
from typing import Optional

# Import all components
from config import config
from database import db_manager
from tokenizer import tokenizer_manager
from gui import main as gui_main
from api import run_api_server

def setup_default_tokenizer():
    """Setup a default tokenizer if none exists"""
    try:
        if not tokenizer_manager.list_tokenizers():
            print("Creating default tokenizer...")
            tokenizer = tokenizer_manager.create_tokenizer("default", "word")
            
            # Train with sample text
            sample_texts = [
                "Hello, this is a sample text for training the tokenizer.",
                "The Gen2All system provides advanced AI backend capabilities.",
                "Machine learning models require proper tokenization for text processing.",
                "Python is a powerful programming language for AI development."
            ]
            
            tokenizer.build_vocab(sample_texts)
            tokenizer_manager.set_current_tokenizer("default")
            print("Default tokenizer created and trained successfully!")
            
    except Exception as e:
        print(f"Warning: Failed to setup default tokenizer: {e}")

def run_gui():
    """Run the GUI application"""
    try:
        print("Starting Gen2All GUI...")
        gui_main()
    except Exception as e:
        print(f"GUI Error: {e}")

def run_api():
    """Run the API server"""
    try:
        print(f"Starting Gen2All API server on {config.api.host}:{config.api.port}")
        run_api_server()
    except Exception as e:
        print(f"API Error: {e}")

def run_both():
    """Run both GUI and API server"""
    # Start API server in background thread
    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()
    
    # Give API server time to start
    time.sleep(2)
    
    # Run GUI in main thread
    run_gui()

def show_info():
    """Show system information"""
    print("\n" + "="*60)
    print("Gen2All - AI Backend Management System")
    print("="*60)
    print(f"Version: 1.0.0")
    print(f"Python Version: {sys.version}")
    print()
    
    print("Configuration:")
    print(f"  Database Type: {config.database.type}")
    print(f"  Database Path: {config.database.path}")
    print(f"  GUI Theme: {config.gui.theme}")
    print(f"  API Host: {config.api.host}:{config.api.port}")
    print()
    
    print("Components Status:")
    
    # Database status
    try:
        if db_manager.db and db_manager.db.connection:
            print("  ✓ Database: Connected")
            
            # Get table counts
            models = db_manager.get_all_models()
            datasets = db_manager.get_all_datasets()
            tokenizers = db_manager.get_all_tokenizers()
            
            print(f"    - Models: {len(models)}")
            print(f"    - Datasets: {len(datasets)}")
            print(f"    - Tokenizers: {len(tokenizers)}")
        else:
            print("  ✗ Database: Disconnected")
    except Exception as e:
        print(f"  ✗ Database: Error - {e}")
    
    # Tokenizer status
    try:
        tokenizer_names = tokenizer_manager.list_tokenizers()
        print(f"  ✓ Tokenizers: {len(tokenizer_names)} available")
        
        if tokenizer_manager.current_tokenizer:
            current_name = None
            for name, tokenizer in tokenizer_manager.tokenizers.items():
                if tokenizer == tokenizer_manager.current_tokenizer:
                    current_name = name
                    break
            print(f"    - Current: {current_name}")
            print(f"    - Vocab Size: {tokenizer_manager.current_tokenizer.get_vocab_size()}")
        else:
            print("    - No active tokenizer")
            
        for name in tokenizer_names:
            tokenizer = tokenizer_manager.get_tokenizer(name)
            print(f"    - {name}: {tokenizer.__class__.__name__} ({tokenizer.get_vocab_size()} tokens)")
            
    except Exception as e:
        print(f"  ✗ Tokenizers: Error - {e}")
    
    print("\n" + "="*60)

def interactive_demo():
    """Run interactive demo"""
    print("\n" + "="*50)
    print("Gen2All Interactive Demo")
    print("="*50)
    
    try:
        # Test tokenization
        print("\n1. Testing Tokenization:")
        test_text = "Hello, Gen2All is an amazing AI backend system!"
        print(f"   Input: '{test_text}'")
        
        if tokenizer_manager.current_tokenizer:
            tokens = tokenizer_manager.encode_text(test_text)
            decoded = tokenizer_manager.decode_tokens(tokens)
            
            print(f"   Encoded: {tokens}")
            print(f"   Decoded: '{decoded}'")
            print(f"   Token count: {len(tokens)}")
        else:
            print("   No tokenizer available")
        
        # Test database
        print("\n2. Testing Database:")
        models = db_manager.get_all_models()
        print(f"   Models in database: {len(models)}")
        
        for model in models[:3]:  # Show first 3 models
            print(f"   - {model['name']} ({model['type']})")
        
        # Create test model
        print("\n3. Creating Test Model:")
        test_model_name = f"demo_model_{int(time.time())}"
        model_id = db_manager.create_model(
            test_model_name, 
            "test",
            {"demo": True, "created_by": "interactive_demo"}
        )
        print(f"   Created model: {test_model_name} (ID: {model_id})")
        
        # Test tokenizer creation
        print("\n4. Creating Test Tokenizer:")
        test_tokenizer_name = f"demo_tokenizer_{int(time.time())}"
        demo_tokenizer = tokenizer_manager.create_tokenizer(
            test_tokenizer_name, "character", vocab_size=1000
        )
        
        # Train with sample data
        sample_data = [
            "Gen2All provides comprehensive AI backend functionality.",
            "The system handles databases, tokenization, and GUI components.",
            "Python-based architecture ensures flexibility and extensibility."
        ]
        
        demo_tokenizer.build_vocab(sample_data)
        print(f"   Created and trained: {test_tokenizer_name}")
        print(f"   Vocabulary size: {demo_tokenizer.get_vocab_size()}")
        
        # Test the new tokenizer
        test_encode = demo_tokenizer.encode("Test encoding with new tokenizer!")
        test_decode = demo_tokenizer.decode(test_encode)
        print(f"   Test encode: {test_encode}")
        print(f"   Test decode: '{test_decode}'")
        
        print("\n" + "="*50)
        print("Demo completed successfully!")
        print("="*50)
        
    except Exception as e:
        print(f"Demo error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Gen2All - AI Backend Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run GUI application
  python main.py --api             # Run API server only
  python main.py --both            # Run both GUI and API
  python main.py --info            # Show system information
  python main.py --demo            # Run interactive demo
        """
    )
    
    parser.add_argument('--gui', action='store_true', default=False,
                       help='Run GUI application (default when no other option specified)')
    parser.add_argument('--api', action='store_true', default=False,
                       help='Run API server only')
    parser.add_argument('--both', action='store_true', default=False,
                       help='Run both GUI and API server')
    parser.add_argument('--info', action='store_true', default=False,
                       help='Show system information and status')
    parser.add_argument('--demo', action='store_true', default=False,
                       help='Run interactive demo')
    parser.add_argument('--config', type=str, metavar='FILE',
                       help='Use custom configuration file')
    
    args = parser.parse_args()
    
    # Load custom config if specified
    if args.config:
        global config
        from config import ConfigManager
        config = ConfigManager(args.config)
    
    print("Gen2All - AI Backend Management System")
    print("=====================================")
    
    try:
        # Initialize system
        print("Initializing system components...")
        setup_default_tokenizer()
        print("System ready!")
        
        # Execute requested action
        if args.info:
            show_info()
        elif args.demo:
            interactive_demo()
        elif args.api:
            run_api()
        elif args.both:
            run_both()
        elif args.gui:
            run_gui()
        else:
            # Default: run GUI if no specific option is chosen
            run_gui()
            
    except KeyboardInterrupt:
        print("\nShutting down Gen2All...")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Cleanup
        try:
            if db_manager:
                db_manager.close()
        except:
            pass

if __name__ == "__main__":
    main()
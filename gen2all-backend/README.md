# Gen2All - AI Backend Management System

A comprehensive Python-based AI backend system that handles databases, tokenization, and GUI management with full customization capabilities.

## Features

- **Model Management**: Create, update, and manage AI models with configurable architectures
- **Advanced Tokenization**: Support for Word, Character, and BPE tokenizers with customizable vocabularies
- **Database Operations**: SQLite-based storage with full CRUD operations
- **Modern GUI**: tkinter-based desktop interface with dark/light themes
- **REST API**: Flask-based API for external integrations
- **Configuration System**: Comprehensive settings management
- **Training Sessions**: Track and manage model training processes

## Installation

This system uses only Python standard library and Flask (which should work in WebContainer):

```bash
pip install flask flask-cors
```

## Quick Start

### Run GUI Application
```bash
python main.py
```

### Run API Server Only
```bash
python main.py --api
```

### Run Both GUI and API
```bash
python main.py --both
```

### Show System Information
```bash
python main.py --info
```

### Run Interactive Demo
```bash
python main.py --demo
```

## Architecture

### Core Components

1. **Configuration System** (`config.py`)
   - Centralized configuration management
   - JSON-based settings with defaults
   - Runtime configuration updates

2. **Database Layer** (`database.py`)
   - Abstract database interface
   - SQLite implementation with migrations
   - Model, dataset, and tokenizer management

3. **Tokenization Engine** (`tokenizer.py`)
   - Multiple tokenizer types (Word, Character, BPE)
   - Vocabulary management
   - Training and inference capabilities

4. **GUI Interface** (`gui.py`)
   - Modern tkinter-based desktop application
   - Tabbed interface for different functions
   - Theme support (dark/light)

5. **REST API** (`api.py`)
   - Flask-based web API
   - Complete CRUD operations
   - CORS support for web integration

### Key Features

#### Tokenization
- **Word Tokenizer**: Traditional word-based tokenization with customizable vocabulary
- **Character Tokenizer**: Character-level tokenization for fine-grained control
- **BPE Tokenizer**: Simplified Byte Pair Encoding implementation
- **Configurable**: Vocabulary size, case sensitivity, special tokens

#### Database Management
- **Model Storage**: Store model configurations and metadata
- **Dataset Tracking**: Manage training datasets and their properties
- **Training Sessions**: Track training progress and logs
- **User Preferences**: Customizable user settings

#### GUI Features
- **Model Management**: Create, edit, and delete models
- **Tokenizer Training**: Train tokenizers on custom datasets
- **Database Browser**: Query and explore database contents
- **Settings Panel**: Configure all system parameters

#### API Endpoints
- **Models**: `/models` - CRUD operations for AI models
- **Tokenizers**: `/tokenizers` - Tokenizer management and inference
- **Datasets**: `/datasets` - Dataset management
- **Training**: `/training-sessions` - Training session management
- **Database**: `/database` - Database operations and queries

## Configuration

The system uses a JSON configuration file (`config.json`) with the following structure:

```json
{
  "database": {
    "type": "sqlite",
    "path": "gen2all.db"
  },
  "tokenizer": {
    "type": "word",
    "vocab_size": 10000,
    "max_tokens": 1000,
    "case_sensitive": false
  },
  "gui": {
    "theme": "dark",
    "width": 1200,
    "height": 800
  },
  "api": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": false
  }
}
```

## Usage Examples

### Creating a Model
```python
from database import db_manager

model_id = db_manager.create_model(
    name="my_transformer",
    model_type="transformer",
    model_config={
        "layers": 12,
        "hidden_size": 768,
        "attention_heads": 12
    }
)
```

### Training a Tokenizer
```python
from tokenizer import tokenizer_manager

# Create tokenizer
tokenizer = tokenizer_manager.create_tokenizer("my_tokenizer", "word")

# Train on data
training_texts = ["Sample text 1", "Sample text 2", ...]
tokenizer.build_vocab(training_texts)

# Use tokenizer
tokens = tokenizer.encode("Hello, world!")
decoded = tokenizer.decode(tokens)
```

### API Usage
```bash
# Create a model via API
curl -X POST http://localhost:5000/models \
  -H "Content-Type: application/json" \
  -d '{"name": "test_model", "type": "transformer"}'

# Encode text with tokenizer
curl -X POST http://localhost:5000/tokenizers/default/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, Gen2All!"}'
```

## Customization

### Adding New Tokenizer Types
Extend the `TokenizerInterface` class and implement required methods:

```python
class CustomTokenizer(TokenizerInterface):
    def encode(self, text: str) -> List[int]:
        # Implementation here
        pass
    
    def decode(self, token_ids: List[int]) -> str:
        # Implementation here
        pass
```

### Database Extensions
Add new tables by modifying the `_create_tables` method in `SQLiteDatabase`:

```python
def _create_tables(self):
    # Add your custom table definitions
    custom_table = """
    CREATE TABLE IF NOT EXISTS custom_table (
        id INTEGER PRIMARY KEY,
        data TEXT
    )
    """
    self.cursor.execute(custom_table)
```

### GUI Customization
Create new tabs by extending the main GUI:

```python
class CustomFrame(ThemedGUI):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.frame = ttk.Frame(parent)
        self.create_widgets()
```

## Development

The system is designed with modularity in mind:

- **Separation of Concerns**: Each component has a specific responsibility
- **Interface-Based Design**: Abstract interfaces allow easy extension
- **Configuration-Driven**: Behavior controlled through configuration
- **Error Handling**: Comprehensive error handling throughout
- **Logging**: Built-in logging for debugging and monitoring

## Troubleshooting

### Common Issues

1. **Database Connection**: Ensure the database file is accessible
2. **GUI Dependencies**: tkinter should be available in standard Python
3. **API Port**: Check that the configured port is available
4. **File Permissions**: Ensure write access for database and config files

### Debug Mode

Enable debug mode in configuration:
```json
{
  "api": {
    "debug": true
  }
}
```

## License

This project is open source and available under the MIT License.
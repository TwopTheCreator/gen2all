"""
Gen2All REST API Server
Flask-based REST API for external integrations
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

from config import config
from database import db_manager
from tokenizer import tokenizer_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Enable CORS if configured
if config.api.cors_enabled:
    CORS(app)

# Error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found', 'message': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# Configuration endpoints
@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    try:
        config_data = {
            'database': {
                'type': config.database.type,
                'path': config.database.path
            },
            'tokenizer': {
                'type': config.tokenizer.type,
                'vocab_size': config.tokenizer.vocab_size,
                'max_tokens': config.tokenizer.max_tokens,
                'case_sensitive': config.tokenizer.case_sensitive
            },
            'gui': {
                'theme': config.gui.theme,
                'width': config.gui.width,
                'height': config.gui.height
            },
            'api': {
                'host': config.api.host,
                'port': config.api.port,
                'debug': config.api.debug
            }
        }
        return jsonify(config_data)
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        data = request.get_json()
        
        if 'database' in data:
            for key, value in data['database'].items():
                if hasattr(config.database, key):
                    setattr(config.database, key, value)
        
        if 'tokenizer' in data:
            for key, value in data['tokenizer'].items():
                if hasattr(config.tokenizer, key):
                    setattr(config.tokenizer, key, value)
        
        if 'gui' in data:
            for key, value in data['gui'].items():
                if hasattr(config.gui, key):
                    setattr(config.gui, key, value)
        
        if 'api' in data:
            for key, value in data['api'].items():
                if hasattr(config.api, key):
                    setattr(config.api, key, value)
        
        config.save_config()
        
        return jsonify({'message': 'Configuration updated successfully'})
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return jsonify({'error': str(e)}), 500

# Model endpoints
@app.route('/models', methods=['GET'])
def get_models():
    """Get all models"""
    try:
        models = db_manager.get_all_models()
        return jsonify({'models': models})
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['POST'])
def create_model():
    """Create a new model"""
    try:
        data = request.get_json()
        
        if not data.get('name'):
            return jsonify({'error': 'Model name is required'}), 400
        
        name = data['name']
        model_type = data.get('type', 'transformer')
        model_config = data.get('config', {})
        
        model_id = db_manager.create_model(name, model_type, model_config)
        
        return jsonify({
            'message': f'Model created successfully',
            'model_id': model_id
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/<int:model_id>', methods=['GET'])
def get_model(model_id):
    """Get specific model"""
    try:
        model = db_manager.get_model(model_id)
        
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        return jsonify({'model': model})
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/<int:model_id>', methods=['PUT'])
def update_model(model_id):
    """Update specific model"""
    try:
        data = request.get_json()
        
        # Verify model exists
        model = db_manager.get_model(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        # Update model
        updates = {}
        for key in ['name', 'type', 'config']:
            if key in data:
                updates[key] = data[key]
        
        if updates:
            db_manager.update_model(model_id, updates)
        
        return jsonify({'message': 'Model updated successfully'})
    except Exception as e:
        logger.error(f"Error updating model {model_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models/<int:model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete specific model"""
    try:
        # Verify model exists
        model = db_manager.get_model(model_id)
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        db_manager.delete_model(model_id)
        
        return jsonify({'message': 'Model deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}")
        return jsonify({'error': str(e)}), 500

# Dataset endpoints
@app.route('/datasets', methods=['GET'])
def get_datasets():
    """Get all datasets"""
    try:
        datasets = db_manager.get_all_datasets()
        return jsonify({'datasets': datasets})
    except Exception as e:
        logger.error(f"Error getting datasets: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/datasets', methods=['POST'])
def create_dataset():
    """Create a new dataset"""
    try:
        data = request.get_json()
        
        if not data.get('name'):
            return jsonify({'error': 'Dataset name is required'}), 400
        
        name = data['name']
        description = data.get('description')
        file_path = data.get('file_path')
        size = data.get('size')
        
        dataset_id = db_manager.create_dataset(name, description, file_path, size)
        
        return jsonify({
            'message': 'Dataset created successfully',
            'dataset_id': dataset_id
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/datasets/<int:dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    """Get specific dataset"""
    try:
        dataset = db_manager.get_dataset(dataset_id)
        
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        return jsonify({'dataset': dataset})
    except Exception as e:
        logger.error(f"Error getting dataset {dataset_id}: {e}")
        return jsonify({'error': str(e)}), 500

# Tokenizer endpoints
@app.route('/tokenizers', methods=['GET'])
def get_tokenizers():
    """Get all tokenizers"""
    try:
        tokenizer_names = tokenizer_manager.list_tokenizers()
        tokenizers = []
        
        for name in tokenizer_names:
            tokenizer = tokenizer_manager.get_tokenizer(name)
            if tokenizer:
                tokenizers.append({
                    'name': name,
                    'vocab_size': tokenizer.get_vocab_size(),
                    'type': tokenizer.__class__.__name__
                })
        
        return jsonify({'tokenizers': tokenizers})
    except Exception as e:
        logger.error(f"Error getting tokenizers: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/tokenizers', methods=['POST'])
def create_tokenizer():
    """Create a new tokenizer"""
    try:
        data = request.get_json()
        
        if not data.get('name'):
            return jsonify({'error': 'Tokenizer name is required'}), 400
        
        name = data['name']
        tokenizer_type = data.get('type', 'word')
        vocab_size = data.get('vocab_size', config.tokenizer.vocab_size)
        case_sensitive = data.get('case_sensitive', config.tokenizer.case_sensitive)
        
        tokenizer = tokenizer_manager.create_tokenizer(
            name, tokenizer_type,
            vocab_size=vocab_size,
            case_sensitive=case_sensitive
        )
        
        # Save to database
        tokenizer_config = {
            'vocab_size': vocab_size,
            'case_sensitive': case_sensitive
        }
        db_manager.create_tokenizer(name, tokenizer_type, {}, tokenizer_config)
        
        return jsonify({
            'message': 'Tokenizer created successfully',
            'name': name,
            'type': tokenizer_type
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating tokenizer: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/tokenizers/<tokenizer_name>/train', methods=['POST'])
def train_tokenizer(tokenizer_name):
    """Train a specific tokenizer"""
    try:
        data = request.get_json()
        
        if not data.get('training_texts'):
            return jsonify({'error': 'Training texts are required'}), 400
        
        training_texts = data['training_texts']
        
        if not isinstance(training_texts, list):
            return jsonify({'error': 'Training texts must be a list'}), 400
        
        tokenizer = tokenizer_manager.get_tokenizer(tokenizer_name)
        if not tokenizer:
            return jsonify({'error': 'Tokenizer not found'}), 404
        
        # Train tokenizer
        tokenizer.build_vocab(training_texts)
        
        return jsonify({
            'message': f'Tokenizer {tokenizer_name} trained successfully',
            'vocab_size': tokenizer.get_vocab_size()
        })
        
    except Exception as e:
        logger.error(f"Error training tokenizer {tokenizer_name}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/tokenizers/<tokenizer_name>/encode', methods=['POST'])
def encode_text(tokenizer_name):
    """Encode text using specific tokenizer"""
    try:
        data = request.get_json()
        
        if not data.get('text'):
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        
        token_ids = tokenizer_manager.encode_text(text, tokenizer_name)
        
        return jsonify({
            'text': text,
            'token_ids': token_ids,
            'num_tokens': len(token_ids)
        })
        
    except Exception as e:
        logger.error(f"Error encoding text with {tokenizer_name}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/tokenizers/<tokenizer_name>/decode', methods=['POST'])
def decode_tokens(tokenizer_name):
    """Decode token IDs using specific tokenizer"""
    try:
        data = request.get_json()
        
        if not data.get('token_ids'):
            return jsonify({'error': 'Token IDs are required'}), 400
        
        token_ids = data['token_ids']
        
        if not isinstance(token_ids, list):
            return jsonify({'error': 'Token IDs must be a list'}), 400
        
        decoded_text = tokenizer_manager.decode_tokens(token_ids, tokenizer_name)
        
        return jsonify({
            'token_ids': token_ids,
            'text': decoded_text,
            'num_tokens': len(token_ids)
        })
        
    except Exception as e:
        logger.error(f"Error decoding tokens with {tokenizer_name}: {e}")
        return jsonify({'error': str(e)}), 500

# Training session endpoints
@app.route('/training-sessions', methods=['GET'])
def get_training_sessions():
    """Get all training sessions"""
    try:
        sessions = db_manager.get_all_training_sessions()
        return jsonify({'training_sessions': sessions})
    except Exception as e:
        logger.error(f"Error getting training sessions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/training-sessions', methods=['POST'])
def create_training_session():
    """Create a new training session"""
    try:
        data = request.get_json()
        
        model_id = data.get('model_id')
        dataset_id = data.get('dataset_id')
        
        if not model_id or not dataset_id:
            return jsonify({'error': 'Model ID and Dataset ID are required'}), 400
        
        # Verify model and dataset exist
        model = db_manager.get_model(model_id)
        dataset = db_manager.get_dataset(dataset_id)
        
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        session_id = db_manager.create_training_session(model_id, dataset_id)
        
        return jsonify({
            'message': 'Training session created successfully',
            'session_id': session_id
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating training session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/training-sessions/<int:session_id>', methods=['GET'])
def get_training_session(session_id):
    """Get specific training session"""
    try:
        session = db_manager.get_training_session(session_id)
        
        if not session:
            return jsonify({'error': 'Training session not found'}), 404
        
        return jsonify({'training_session': session})
    except Exception as e:
        logger.error(f"Error getting training session {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/training-sessions/<int:session_id>/progress', methods=['PUT'])
def update_training_progress(session_id):
    """Update training session progress"""
    try:
        data = request.get_json()
        
        # Verify session exists
        session = db_manager.get_training_session(session_id)
        if not session:
            return jsonify({'error': 'Training session not found'}), 404
        
        progress = data.get('progress')
        status = data.get('status')
        logs = data.get('logs')
        
        if progress is not None:
            if not (0 <= progress <= 1.0):
                return jsonify({'error': 'Progress must be between 0 and 1'}), 400
        
        db_manager.update_training_progress(session_id, progress, status, logs)
        
        return jsonify({'message': 'Training progress updated successfully'})
    except Exception as e:
        logger.error(f"Error updating training progress {session_id}: {e}")
        return jsonify({'error': str(e)}), 500

# Database endpoints
@app.route('/database/info', methods=['GET'])
def get_database_info():
    """Get database information"""
    try:
        # Get table information
        tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
        tables = db_manager.db.fetch_all(tables_query)
        
        table_info = []
        for table in tables:
            table_name = table['name']
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count_result = db_manager.db.fetch_one(count_query)
            count = count_result['count'] if count_result else 0
            
            table_info.append({
                'name': table_name,
                'record_count': count
            })
        
        return jsonify({
            'database_type': config.database.type,
            'database_path': config.database.path,
            'tables': table_info,
            'connected': bool(db_manager.db.connection)
        })
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/database/query', methods=['POST'])
def execute_database_query():
    """Execute database query"""
    try:
        data = request.get_json()
        
        if not data.get('query'):
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        
        if not query.lower().startswith('select'):
            return jsonify({'error': 'Only SELECT queries are allowed'}), 400
        
        results = db_manager.db.fetch_all(query)
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return jsonify({'error': str(e)}), 500

def run_api_server():
    """Run the Flask API server"""
    try:
        logger.info(f"Starting Gen2All API server on {config.api.host}:{config.api.port}")
        app.run(
            host=config.api.host,
            port=config.api.port,
            debug=config.api.debug,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        raise

if __name__ == "__main__":
    run_api_server()
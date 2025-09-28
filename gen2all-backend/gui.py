"""
Gen2All GUI System
Modern desktop GUI using tkinter with advanced features
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import json
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from config import config
from database import db_manager
from tokenizer import tokenizer_manager

class ThemedGUI:
    """Base class for themed GUI components"""
    
    def __init__(self):
        self.colors = self._get_theme_colors()
        self.fonts = self._get_theme_fonts()
    
    def _get_theme_colors(self) -> Dict[str, str]:
        """Get theme colors based on configuration"""
        if config.gui.theme == "dark":
            return {
                'bg': '#2b2b2b',
                'fg': '#ffffff',
                'select_bg': '#404040',
                'select_fg': '#ffffff',
                'button_bg': '#404040',
                'button_fg': '#ffffff',
                'entry_bg': '#404040',
                'entry_fg': '#ffffff',
                'frame_bg': '#353535',
                'accent': '#0078d4',
                'success': '#107c10',
                'warning': '#ff8c00',
                'error': '#d13438'
            }
        else:
            return {
                'bg': '#ffffff',
                'fg': '#000000',
                'select_bg': '#0078d4',
                'select_fg': '#ffffff',
                'button_bg': '#f0f0f0',
                'button_fg': '#000000',
                'entry_bg': '#ffffff',
                'entry_fg': '#000000',
                'frame_bg': '#f5f5f5',
                'accent': '#0078d4',
                'success': '#107c10',
                'warning': '#ff8c00',
                'error': '#d13438'
            }
    
    def _get_theme_fonts(self) -> Dict[str, tuple]:
        """Get theme fonts based on configuration"""
        font_family = config.gui.font_family
        font_size = config.gui.font_size
        
        return {
            'default': (font_family, font_size),
            'heading': (font_family, font_size + 4, 'bold'),
            'subheading': (font_family, font_size + 2, 'bold'),
            'small': (font_family, font_size - 2),
            'code': ('Consolas', font_size)
        }
    
    def configure_style(self):
        """Configure ttk styles for theming"""
        style = ttk.Style()
        
        # Configure styles based on theme
        style.configure('TNotebook', background=self.colors['bg'])
        style.configure('TNotebook.Tab', 
                       background=self.colors['button_bg'],
                       foreground=self.colors['button_fg'],
                       padding=[12, 8])
        style.map('TNotebook.Tab',
                 background=[('selected', self.colors['accent']),
                            ('active', self.colors['select_bg'])])
        
        style.configure('TFrame', background=self.colors['frame_bg'])
        style.configure('TLabel', 
                       background=self.colors['frame_bg'],
                       foreground=self.colors['fg'])
        
        style.configure('TButton',
                       background=self.colors['button_bg'],
                       foreground=self.colors['button_fg'],
                       padding=[8, 4])
        style.map('TButton',
                 background=[('active', self.colors['accent']),
                            ('pressed', self.colors['select_bg'])])
        
        style.configure('TEntry',
                       background=self.colors['entry_bg'],
                       foreground=self.colors['entry_fg'])

class ModelManagementFrame(ThemedGUI):
    """Frame for managing AI models"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        self.create_widgets()
        self.load_models()
    
    def create_widgets(self):
        """Create model management widgets"""
        # Title
        title = ttk.Label(self.frame, text="Model Management", 
                         font=self.fonts['heading'])
        title.pack(pady=(0, 20))
        
        # Model creation frame
        create_frame = ttk.LabelFrame(self.frame, text="Create New Model", 
                                     padding=10)
        create_frame.pack(fill='x', pady=(0, 20))
        
        # Model name
        ttk.Label(create_frame, text="Model Name:").grid(row=0, column=0, 
                                                        sticky='w', pady=5)
        self.model_name_var = tk.StringVar()
        name_entry = ttk.Entry(create_frame, textvariable=self.model_name_var, 
                              width=30)
        name_entry.grid(row=0, column=1, sticky='ew', pady=5, padx=(10, 0))
        
        # Model type
        ttk.Label(create_frame, text="Model Type:").grid(row=1, column=0, 
                                                        sticky='w', pady=5)
        self.model_type_var = tk.StringVar(value="transformer")
        type_combo = ttk.Combobox(create_frame, textvariable=self.model_type_var,
                                 values=["transformer", "rnn", "cnn", "custom"],
                                 state="readonly", width=27)
        type_combo.grid(row=1, column=1, sticky='ew', pady=5, padx=(10, 0))
        
        # Create button
        create_btn = ttk.Button(create_frame, text="Create Model",
                               command=self.create_model)
        create_btn.grid(row=2, column=0, columnspan=2, pady=10)
        
        create_frame.columnconfigure(1, weight=1)
        
        # Models list frame
        list_frame = ttk.LabelFrame(self.frame, text="Existing Models", 
                                   padding=10)
        list_frame.pack(fill='both', expand=True)
        
        # Treeview for models
        columns = ('ID', 'Name', 'Type', 'Created')
        self.model_tree = ttk.Treeview(list_frame, columns=columns, 
                                      show='headings', height=10)
        
        for col in columns:
            self.model_tree.heading(col, text=col)
            self.model_tree.column(col, width=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical',
                                 command=self.model_tree.yview)
        self.model_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.model_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Buttons frame
        buttons_frame = ttk.Frame(list_frame)
        buttons_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(buttons_frame, text="Refresh", 
                  command=self.load_models).pack(side='left', padx=(0, 10))
        ttk.Button(buttons_frame, text="Delete Selected", 
                  command=self.delete_selected_model).pack(side='left')
    
    def create_model(self):
        """Create a new model"""
        name = self.model_name_var.get().strip()
        model_type = self.model_type_var.get()
        
        if not name:
            messagebox.showerror("Error", "Please enter a model name")
            return
        
        try:
            model_config = {
                'architecture': model_type,
                'parameters': {},
                'training_config': {}
            }
            
            model_id = db_manager.create_model(name, model_type, model_config)
            messagebox.showinfo("Success", f"Model '{name}' created with ID: {model_id}")
            
            # Clear form
            self.model_name_var.set("")
            self.model_type_var.set("transformer")
            
            # Refresh list
            self.load_models()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create model: {str(e)}")
    
    def load_models(self):
        """Load and display models"""
        try:
            # Clear existing items
            for item in self.model_tree.get_children():
                self.model_tree.delete(item)
            
            # Load models from database
            models = db_manager.get_all_models()
            
            for model in models:
                created_at = model['created_at'][:19] if model['created_at'] else ''
                
                self.model_tree.insert('', 'end', values=(
                    model['id'],
                    model['name'],
                    model['type'],
                    created_at
                ))
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")
    
    def delete_selected_model(self):
        """Delete selected model"""
        selection = self.model_tree.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a model to delete")
            return
        
        item = self.model_tree.item(selection[0])
        model_id = item['values'][0]
        model_name = item['values'][1]
        
        if messagebox.askyesno("Confirm Delete", 
                              f"Are you sure you want to delete model '{model_name}'?"):
            try:
                db_manager.delete_model(model_id)
                messagebox.showinfo("Success", f"Model '{model_name}' deleted")
                self.load_models()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete model: {str(e)}")

class TokenizerFrame(ThemedGUI):
    """Frame for tokenizer management"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        self.create_widgets()
        self.load_tokenizers()
    
    def create_widgets(self):
        """Create tokenizer widgets"""
        # Title
        title = ttk.Label(self.frame, text="Tokenizer Management", 
                         font=self.fonts['heading'])
        title.pack(pady=(0, 20))
        
        # Create tokenizer frame
        create_frame = ttk.LabelFrame(self.frame, text="Create Tokenizer", 
                                     padding=10)
        create_frame.pack(fill='x', pady=(0, 20))
        
        # Tokenizer name
        ttk.Label(create_frame, text="Name:").grid(row=0, column=0, 
                                                  sticky='w', pady=5)
        self.tokenizer_name_var = tk.StringVar()
        name_entry = ttk.Entry(create_frame, textvariable=self.tokenizer_name_var,
                              width=20)
        name_entry.grid(row=0, column=1, sticky='ew', pady=5, padx=(10, 0))
        
        # Tokenizer type
        ttk.Label(create_frame, text="Type:").grid(row=0, column=2, 
                                                  sticky='w', pady=5, padx=(20, 0))
        self.tokenizer_type_var = tk.StringVar(value="word")
        type_combo = ttk.Combobox(create_frame, textvariable=self.tokenizer_type_var,
                                 values=["word", "character", "bpe"], 
                                 state="readonly", width=15)
        type_combo.grid(row=0, column=3, sticky='ew', pady=5, padx=(10, 0))
        
        # Vocab size
        ttk.Label(create_frame, text="Vocab Size:").grid(row=1, column=0, 
                                                         sticky='w', pady=5)
        self.vocab_size_var = tk.IntVar(value=config.tokenizer.vocab_size)
        vocab_entry = ttk.Entry(create_frame, textvariable=self.vocab_size_var,
                               width=20)
        vocab_entry.grid(row=1, column=1, sticky='ew', pady=5, padx=(10, 0))
        
        # Case sensitive
        self.case_sensitive_var = tk.BooleanVar(value=config.tokenizer.case_sensitive)
        case_check = ttk.Checkbutton(create_frame, text="Case Sensitive",
                                    variable=self.case_sensitive_var)
        case_check.grid(row=1, column=2, columnspan=2, sticky='w', 
                       pady=5, padx=(20, 0))
        
        # Create button
        create_btn = ttk.Button(create_frame, text="Create Tokenizer",
                               command=self.create_tokenizer)
        create_btn.grid(row=2, column=0, columnspan=4, pady=10)
        
        create_frame.columnconfigure(1, weight=1)
        create_frame.columnconfigure(3, weight=1)
        
        # Training frame
        train_frame = ttk.LabelFrame(self.frame, text="Train Tokenizer", 
                                    padding=10)
        train_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Training text
        ttk.Label(train_frame, text="Training Text:").pack(anchor='w')
        
        self.training_text = scrolledtext.ScrolledText(train_frame, height=10,
                                                      font=self.fonts['code'])
        self.training_text.pack(fill='both', expand=True, pady=(5, 10))
        
        # Training buttons
        train_buttons = ttk.Frame(train_frame)
        train_buttons.pack(fill='x')
        
        ttk.Button(train_buttons, text="Load File",
                  command=self.load_training_file).pack(side='left', padx=(0, 10))
        ttk.Button(train_buttons, text="Train Selected",
                  command=self.train_tokenizer).pack(side='left')
        
        # Test frame
        test_frame = ttk.LabelFrame(self.frame, text="Test Tokenization", 
                                   padding=10)
        test_frame.pack(fill='x')
        
        # Test input
        ttk.Label(test_frame, text="Test Text:").pack(anchor='w')
        self.test_text_var = tk.StringVar(value="Hello, this is a test sentence!")
        test_entry = ttk.Entry(test_frame, textvariable=self.test_text_var,
                              font=self.fonts['default'])
        test_entry.pack(fill='x', pady=(5, 10))
        
        # Test buttons
        test_buttons = ttk.Frame(test_frame)
        test_buttons.pack(fill='x')
        
        ttk.Button(test_buttons, text="Encode",
                  command=self.test_encode).pack(side='left', padx=(0, 10))
        ttk.Button(test_buttons, text="Decode",
                  command=self.test_decode).pack(side='left')
        
        # Results
        self.test_result_var = tk.StringVar()
        result_label = ttk.Label(test_frame, textvariable=self.test_result_var,
                                font=self.fonts['small'], wraplength=800)
        result_label.pack(fill='x', pady=(10, 0))
    
    def create_tokenizer(self):
        """Create a new tokenizer"""
        name = self.tokenizer_name_var.get().strip()
        tokenizer_type = self.tokenizer_type_var.get()
        vocab_size = self.vocab_size_var.get()
        case_sensitive = self.case_sensitive_var.get()
        
        if not name:
            messagebox.showerror("Error", "Please enter a tokenizer name")
            return
        
        try:
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
            
            messagebox.showinfo("Success", f"Tokenizer '{name}' created")
            
            # Clear form
            self.tokenizer_name_var.set("")
            self.vocab_size_var.set(config.tokenizer.vocab_size)
            self.case_sensitive_var.set(config.tokenizer.case_sensitive)
            
            self.load_tokenizers()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create tokenizer: {str(e)}")
    
    def load_tokenizers(self):
        """Load available tokenizers"""
        try:
            tokenizer_names = tokenizer_manager.list_tokenizers()
            # Update UI with available tokenizers
            # This could be enhanced with a proper list widget
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load tokenizers: {str(e)}")
    
    def load_training_file(self):
        """Load training text from file"""
        file_path = filedialog.askopenfilename(
            title="Select Training Text File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.training_text.delete(1.0, tk.END)
                self.training_text.insert(1.0, content)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def train_tokenizer(self):
        """Train selected tokenizer"""
        training_content = self.training_text.get(1.0, tk.END).strip()
        
        if not training_content:
            messagebox.showerror("Error", "Please provide training text")
            return
        
        # Get current tokenizer
        if not tokenizer_manager.current_tokenizer:
            messagebox.showerror("Error", "No tokenizer selected")
            return
        
        try:
            # Split into sentences/documents for training
            training_texts = [line.strip() for line in training_content.split('\n') 
                            if line.strip()]
            
            # Train in background thread
            def train_thread():
                try:
                    tokenizer_manager.current_tokenizer.build_vocab(training_texts)
                    messagebox.showinfo("Success", "Tokenizer training completed!")
                except Exception as e:
                    messagebox.showerror("Error", f"Training failed: {str(e)}")
            
            threading.Thread(target=train_thread, daemon=True).start()
            messagebox.showinfo("Training", "Training started in background...")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {str(e)}")
    
    def test_encode(self):
        """Test tokenization encoding"""
        test_text = self.test_text_var.get()
        
        if not tokenizer_manager.current_tokenizer:
            messagebox.showerror("Error", "No tokenizer available")
            return
        
        try:
            token_ids = tokenizer_manager.encode_text(test_text)
            self.test_result_var.set(f"Encoded: {token_ids}")
        except Exception as e:
            messagebox.showerror("Error", f"Encoding failed: {str(e)}")
    
    def test_decode(self):
        """Test tokenization decoding"""
        # For this demo, we'll use the last encoded result
        if not tokenizer_manager.current_tokenizer:
            messagebox.showerror("Error", "No tokenizer available")
            return
        
        try:
            test_text = self.test_text_var.get()
            token_ids = tokenizer_manager.encode_text(test_text)
            decoded_text = tokenizer_manager.decode_tokens(token_ids)
            self.test_result_var.set(f"Decoded: '{decoded_text}'")
        except Exception as e:
            messagebox.showerror("Error", f"Decoding failed: {str(e)}")

class DatabaseFrame(ThemedGUI):
    """Frame for database management"""
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.frame = ttk.Frame(parent)
        
        self.create_widgets()
        self.load_database_info()
    
    def create_widgets(self):
        """Create database management widgets"""
        # Title
        title = ttk.Label(self.frame, text="Database Management", 
                         font=self.fonts['heading'])
        title.pack(pady=(0, 20))
        
        # Database info frame
        info_frame = ttk.LabelFrame(self.frame, text="Database Information", 
                                   padding=10)
        info_frame.pack(fill='x', pady=(0, 20))
        
        self.db_info_text = scrolledtext.ScrolledText(info_frame, height=8,
                                                     font=self.fonts['code'],
                                                     state='disabled')
        self.db_info_text.pack(fill='both', expand=True)
        
        # Query frame
        query_frame = ttk.LabelFrame(self.frame, text="Query Database", 
                                    padding=10)
        query_frame.pack(fill='both', expand=True)
        
        # Query input
        ttk.Label(query_frame, text="SQL Query:").pack(anchor='w')
        
        self.query_text = scrolledtext.ScrolledText(query_frame, height=6,
                                                   font=self.fonts['code'])
        self.query_text.pack(fill='x', pady=(5, 10))
        
        # Query buttons
        query_buttons = ttk.Frame(query_frame)
        query_buttons.pack(fill='x', pady=(0, 10))
        
        ttk.Button(query_buttons, text="Execute Query",
                  command=self.execute_query).pack(side='left', padx=(0, 10))
        ttk.Button(query_buttons, text="Clear Results",
                  command=self.clear_results).pack(side='left')
        
        # Results
        ttk.Label(query_frame, text="Results:").pack(anchor='w')
        
        self.results_text = scrolledtext.ScrolledText(query_frame, height=10,
                                                     font=self.fonts['code'],
                                                     state='disabled')
        self.results_text.pack(fill='both', expand=True, pady=(5, 0))
    
    def load_database_info(self):
        """Load and display database information"""
        try:
            # Enable text widget
            self.db_info_text.config(state='normal')
            self.db_info_text.delete(1.0, tk.END)
            
            # Get database statistics
            info = []
            info.append(f"Database Type: {config.database.type}")
            info.append(f"Database Path: {config.database.path}")
            info.append(f"Connection Status: {'Connected' if db_manager.db.connection else 'Disconnected'}")
            info.append("")
            
            # Get table information
            tables_query = "SELECT name FROM sqlite_master WHERE type='table'"
            tables = db_manager.db.fetch_all(tables_query)
            
            info.append("Tables:")
            for table in tables:
                table_name = table['name']
                count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                count_result = db_manager.db.fetch_one(count_query)
                count = count_result['count'] if count_result else 0
                info.append(f"  - {table_name}: {count} records")
            
            # Display info
            self.db_info_text.insert(1.0, '\n'.join(info))
            self.db_info_text.config(state='disabled')
            
        except Exception as e:
            self.db_info_text.config(state='normal')
            self.db_info_text.delete(1.0, tk.END)
            self.db_info_text.insert(1.0, f"Error loading database info: {str(e)}")
            self.db_info_text.config(state='disabled')
    
    def execute_query(self):
        """Execute SQL query and display results"""
        query = self.query_text.get(1.0, tk.END).strip()
        
        if not query:
            messagebox.showerror("Error", "Please enter a SQL query")
            return
        
        try:
            # Enable results text widget
            self.results_text.config(state='normal')
            self.results_text.delete(1.0, tk.END)
            
            # Execute query
            if query.lower().startswith('select'):
                results = db_manager.db.fetch_all(query)
                
                if results:
                    # Format results as table
                    if results:
                        headers = list(results[0].keys())
                        
                        # Headers
                        header_line = " | ".join(f"{h:15}" for h in headers)
                        separator_line = "-" * len(header_line)
                        
                        self.results_text.insert(tk.END, header_line + "\n")
                        self.results_text.insert(tk.END, separator_line + "\n")
                        
                        # Data rows
                        for row in results[:100]:  # Limit to 100 rows
                            row_line = " | ".join(f"{str(row[h]):15}" for h in headers)
                            self.results_text.insert(tk.END, row_line + "\n")
                        
                        if len(results) > 100:
                            self.results_text.insert(tk.END, f"\n... showing first 100 of {len(results)} results")
                    else:
                        self.results_text.insert(tk.END, "No results found")
                else:
                    self.results_text.insert(tk.END, "No results returned")
            else:
                # Non-select query
                db_manager.db.execute_query(query)
                self.results_text.insert(tk.END, "Query executed successfully")
                
                # Refresh database info
                self.load_database_info()
            
        except Exception as e:
            self.results_text.insert(tk.END, f"Error executing query: {str(e)}")
        finally:
            self.results_text.config(state='disabled')
    
    def clear_results(self):
        """Clear query results"""
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.config(state='disabled')

class MainGUI(ThemedGUI):
    """Main GUI application"""
    
    def __init__(self):
        super().__init__()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Gen2All - AI Backend Management")
        self.root.geometry(f"{config.gui.width}x{config.gui.height}")
        self.root.configure(bg=self.colors['bg'])
        
        # Configure styles
        self.configure_style()
        
        # Create widgets
        self.create_widgets()
        
        # Initialize components
        self.initialize_components()
    
    def create_widgets(self):
        """Create main GUI widgets"""
        # Main menu
        self.create_menu()
        
        # Main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_tabs()
        
        # Status bar
        self.create_status_bar()
    
    def create_menu(self):
        """Create main menu bar"""
        menubar = tk.Menu(self.root, bg=self.colors['bg'], fg=self.colors['fg'])
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg'], 
                           fg=self.colors['fg'])
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Project", command=self.new_project)
        file_menu.add_command(label="Open Project", command=self.open_project)
        file_menu.add_command(label="Save Project", command=self.save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg'], 
                            fg=self.colors['fg'])
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Settings", command=self.show_settings)
        tools_menu.add_command(label="Logs", command=self.show_logs)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg'], 
                           fg=self.colors['fg'])
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_tabs(self):
        """Create main application tabs"""
        # Model Management tab
        self.model_frame = ModelManagementFrame(self.notebook)
        self.notebook.add(self.model_frame.frame, text="Models")
        
        # Tokenizer tab
        self.tokenizer_frame = TokenizerFrame(self.notebook)
        self.notebook.add(self.tokenizer_frame.frame, text="Tokenizers")
        
        # Database tab
        self.database_frame = DatabaseFrame(self.notebook)
        self.notebook.add(self.database_frame.frame, text="Database")
        
        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text="Settings")
        self.create_settings_tab()
    
    def create_settings_tab(self):
        """Create settings tab"""
        # Title
        title = ttk.Label(self.settings_frame, text="Application Settings", 
                         font=self.fonts['heading'])
        title.pack(pady=(20, 30))
        
        # GUI Settings
        gui_frame = ttk.LabelFrame(self.settings_frame, text="GUI Settings", 
                                  padding=15)
        gui_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        # Theme
        ttk.Label(gui_frame, text="Theme:").grid(row=0, column=0, sticky='w', pady=5)
        self.theme_var = tk.StringVar(value=config.gui.theme)
        theme_combo = ttk.Combobox(gui_frame, textvariable=self.theme_var,
                                  values=["light", "dark"], state="readonly", width=15)
        theme_combo.grid(row=0, column=1, sticky='w', padx=(10, 0))
        
        # Font size
        ttk.Label(gui_frame, text="Font Size:").grid(row=1, column=0, sticky='w', pady=5)
        self.font_size_var = tk.IntVar(value=config.gui.font_size)
        font_spin = ttk.Spinbox(gui_frame, from_=8, to=20, textvariable=self.font_size_var,
                               width=15)
        font_spin.grid(row=1, column=1, sticky='w', padx=(10, 0))
        
        # Database Settings
        db_frame = ttk.LabelFrame(self.settings_frame, text="Database Settings", 
                                 padding=15)
        db_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        # Database path
        ttk.Label(db_frame, text="Database Path:").grid(row=0, column=0, sticky='w', pady=5)
        self.db_path_var = tk.StringVar(value=config.database.path)
        db_entry = ttk.Entry(db_frame, textvariable=self.db_path_var, width=40)
        db_entry.grid(row=0, column=1, sticky='ew', padx=(10, 10))
        
        ttk.Button(db_frame, text="Browse", 
                  command=self.browse_database).grid(row=0, column=2)
        
        db_frame.columnconfigure(1, weight=1)
        
        # Tokenizer Settings
        tok_frame = ttk.LabelFrame(self.settings_frame, text="Tokenizer Settings", 
                                  padding=15)
        tok_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        # Default vocab size
        ttk.Label(tok_frame, text="Default Vocab Size:").grid(row=0, column=0, 
                                                              sticky='w', pady=5)
        self.default_vocab_var = tk.IntVar(value=config.tokenizer.vocab_size)
        vocab_entry = ttk.Entry(tok_frame, textvariable=self.default_vocab_var, width=15)
        vocab_entry.grid(row=0, column=1, sticky='w', padx=(10, 0))
        
        # Max tokens
        ttk.Label(tok_frame, text="Max Tokens:").grid(row=1, column=0, sticky='w', pady=5)
        self.max_tokens_var = tk.IntVar(value=config.tokenizer.max_tokens)
        max_tokens_entry = ttk.Entry(tok_frame, textvariable=self.max_tokens_var, width=15)
        max_tokens_entry.grid(row=1, column=1, sticky='w', padx=(10, 0))
        
        # Save button
        save_btn = ttk.Button(self.settings_frame, text="Save Settings",
                             command=self.save_settings)
        save_btn.pack(pady=20)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill='x', side='bottom')
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        status_label.pack(side='left', padx=10, pady=5)
        
        # Connection status
        self.connection_var = tk.StringVar(value="Database: Connected")
        connection_label = ttk.Label(self.status_frame, textvariable=self.connection_var)
        connection_label.pack(side='right', padx=10, pady=5)
    
    def initialize_components(self):
        """Initialize application components"""
        try:
            # Test database connection
            if db_manager.db and db_manager.db.connection:
                self.connection_var.set("Database: Connected")
            else:
                self.connection_var.set("Database: Disconnected")
            
            # Create default tokenizer if none exists
            if not tokenizer_manager.list_tokenizers():
                tokenizer_manager.create_tokenizer("default", "word")
            
            self.status_var.set("Application initialized successfully")
            
        except Exception as e:
            self.status_var.set(f"Initialization error: {str(e)}")
    
    def new_project(self):
        """Create new project"""
        messagebox.showinfo("New Project", "New project functionality not implemented yet")
    
    def open_project(self):
        """Open existing project"""
        messagebox.showinfo("Open Project", "Open project functionality not implemented yet")
    
    def save_project(self):
        """Save current project"""
        messagebox.showinfo("Save Project", "Save project functionality not implemented yet")
    
    def browse_database(self):
        """Browse for database file"""
        file_path = filedialog.asksaveasfilename(
            title="Select Database File",
            defaultextension=".db",
            filetypes=[("SQLite files", "*.db"), ("All files", "*.*")]
        )
        
        if file_path:
            self.db_path_var.set(file_path)
    
    def save_settings(self):
        """Save application settings"""
        try:
            # Update configuration
            config.gui.theme = self.theme_var.get()
            config.gui.font_size = self.font_size_var.get()
            config.database.path = self.db_path_var.get()
            config.tokenizer.vocab_size = self.default_vocab_var.get()
            config.tokenizer.max_tokens = self.max_tokens_var.get()
            
            # Save configuration
            config.save_config()
            
            self.status_var.set("Settings saved successfully")
            messagebox.showinfo("Success", "Settings saved. Please restart the application for all changes to take effect.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def show_settings(self):
        """Show settings dialog"""
        self.notebook.select(3)  # Select settings tab
    
    def show_logs(self):
        """Show application logs"""
        messagebox.showinfo("Logs", "Log viewer not implemented yet")
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
Gen2All - AI Backend Management System

Version: 1.0.0
Created with Python and tkinter

Features:
- Model Management
- Tokenizer Training
- Database Operations
- Customizable Configuration

© 2025 Gen2All Project
        """
        messagebox.showinfo("About Gen2All", about_text.strip())
    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        finally:
            # Cleanup
            if db_manager:
                db_manager.close()

def main():
    """Main function to run GUI"""
    app = MainGUI()
    app.run()

if __name__ == "__main__":
    main()
CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    version TEXT NOT NULL,
    description TEXT,
    parameters INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

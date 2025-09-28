CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT NOT NULL,
    description TEXT,
    source TEXT,
    record_count INTEGER,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

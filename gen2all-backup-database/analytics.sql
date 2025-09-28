CREATE TABLE usage_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    model_id INTEGER,
    prompt_count INTEGER DEFAULT 0,
    response_count INTEGER DEFAULT 0,
    last_used TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id),
    FOREIGN KEY(model_id) REFERENCES models(id)
);

CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER,
    metric_name TEXT,
    metric_value REAL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(model_id) REFERENCES models(id)
);

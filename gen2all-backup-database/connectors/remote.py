import sqlite3
from pathlib import Path

class DBConnector:
    def __init__(self, db_name):
        # path relative to this scripts parent directory
        self.db_path = Path(__file__).resolve().parent.parent / db_name
        self.connection = None

    def connect(self):
        try:
            self.connection = sqlite3.connect(self.db_path)
            print(f"Connected to {self.db_path.name}")
        except sqlite3.Error as e:
            print(f"Error connecting to {self.db_path.name}: {e}")
        return self.connection

    def close(self):
        if self.connection:
            self.connection.close()
            print(f"Closed connection to {self.db_path.name}")

    def execute_query(self, query, params=None):
        """Execute a query and return results (if SELECT)."""
        if not self.connection:
            self.connect()
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            if query.strip().upper().startswith("SELECT"):
                result = cursor.fetchall()
                return result
            else:
                self.connection.commit()
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
        finally:
            cursor.close()

dbs = {
    "users": DBConnector("users.sql"),
    "models": DBConnector("models.sql"),
    "prompts": DBConnector("prompts.sql"),
    "datasets": DBConnector("datasets.sql"),
    "logs": DBConnector("logs.sql"),
    "config": DBConnector("config.sql"),
    "analytics": DBConnector("analytics.sql")
}

if __name__ == "__main__":
    users_conn = dbs["users"].connect()
    result = dbs["users"].execute_query("SELECT name FROM sqlite_master WHERE type='table';")
    print("Tables in users.db:", result)
    dbs["users"].close()

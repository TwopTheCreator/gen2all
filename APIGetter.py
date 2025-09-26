import sqlite3
import secrets
import hashlib
import json
import time
import os
from typing import Optional, Dict, Any


class Gen2AllAPIKeyGenerator:
    def __init__(self, db_path: str = "gen2all_users.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    api_key TEXT UNIQUE NOT NULL,
                    quota_limit INTEGER DEFAULT -1,
                    quota_used INTEGER DEFAULT 0,
                    rate_limit INTEGER DEFAULT 10000,
                    created_at REAL,
                    last_login REAL,
                    is_active BOOLEAN DEFAULT 1,
                    tier TEXT DEFAULT 'unlimited',
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_username ON users(username)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_key ON users(api_key)
            """)
            conn.commit()
    
    def register_user(self, username: str, email: Optional[str] = None) -> Dict[str, Any]:
        user_id = f"usr_{int(time.time() * 1000000)}_{secrets.token_hex(8)}"
        api_key = f"gen2_{secrets.token_urlsafe(48)}"
        
        user_data = {
            'user_id': user_id,
            'username': username,
            'email': email,
            'api_key': api_key,
            'quota_limit': -1,
            'quota_used': 0,
            'rate_limit': 10000,
            'created_at': time.time(),
            'last_login': time.time(),
            'is_active': True,
            'tier': 'unlimited',
            'metadata': json.dumps({
                'registration_ip': 'local',
                'features': [
                    'unlimited_quota',
                    'high_rate_limit',
                    'premium_models',
                    'advanced_context',
                    'batch_processing',
                    'priority_queue'
                ]
            })
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO users 
                    (user_id, username, email, api_key, quota_limit, quota_used,
                     rate_limit, created_at, last_login, is_active, tier, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_data['user_id'], user_data['username'], user_data['email'],
                    user_data['api_key'], user_data['quota_limit'], user_data['quota_used'],
                    user_data['rate_limit'], user_data['created_at'], user_data['last_login'],
                    user_data['is_active'], user_data['tier'], user_data['metadata']
                ))
                conn.commit()
            
            return {
                'success': True,
                'user_id': user_data['user_id'],
                'username': user_data['username'],
                'api_key': user_data['api_key'],
                'tier': user_data['tier'],
                'quota_limit': user_data['quota_limit'],
                'rate_limit': user_data['rate_limit'],
                'features': json.loads(user_data['metadata'])['features']
            }
            
        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                return {'success': False, 'error': 'Username already exists'}
            elif 'email' in str(e):
                return {'success': False, 'error': 'Email already exists'}
            else:
                return {'success': False, 'error': 'Registration failed'}
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_id, username, email, api_key, quota_limit, quota_used,
                       rate_limit, created_at, last_login, is_active, tier, metadata
                FROM users 
                WHERE username = ?
            """, (username,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'user_id': result[0],
                    'username': result[1],
                    'email': result[2],
                    'api_key': result[3],
                    'quota_limit': result[4],
                    'quota_used': result[5],
                    'rate_limit': result[6],
                    'created_at': result[7],
                    'last_login': result[8],
                    'is_active': bool(result[9]),
                    'tier': result[10],
                    'metadata': json.loads(result[11])
                }
        
        return None
    
    def regenerate_api_key(self, username: str) -> Dict[str, Any]:
        user = self.get_user_by_username(username)
        if not user:
            return {'success': False, 'error': 'User not found'}
        
        new_api_key = f"gen2_{secrets.token_urlsafe(48)}"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE users 
                SET api_key = ?, last_login = ?
                WHERE username = ?
            """, (new_api_key, time.time(), username))
            conn.commit()
        
        return {
            'success': True,
            'username': username,
            'new_api_key': new_api_key,
            'message': 'API key regenerated successfully'
        }
    
    def list_users(self) -> Dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT username, email, tier, quota_used, rate_limit, 
                       created_at, last_login, is_active
                FROM users 
                ORDER BY created_at DESC
            """)
            
            users = []
            for row in cursor.fetchall():
                users.append({
                    'username': row[0],
                    'email': row[1],
                    'tier': row[2],
                    'quota_used': row[3],
                    'rate_limit': row[4],
                    'created_at': row[5],
                    'last_login': row[6],
                    'is_active': bool(row[7])
                })
            
            return {
                'success': True,
                'total_users': len(users),
                'users': users
            }


def main():
    print("=" * 60)
    print("🚀 GEN2ALL API KEY GENERATOR 🚀")
    print("=" * 60)
    print("Welcome to Gen2All - The Ultimate AI Platform")
    print("✨ Unlimited Quota | ⚡ High Performance | 🧠 Advanced AI")
    print("=" * 60)
    
    generator = Gen2AllAPIKeyGenerator()
    
    while True:
        print("\nChoose an option:")
        print("1. 🔑 Generate new API key")
        print("2. 👤 Get existing user info")
        print("3. 🔄 Regenerate API key")
        print("4. 📋 List all users")
        print("5. 🚪 Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            print("\n" + "─" * 40)
            print("📝 NEW USER REGISTRATION")
            print("─" * 40)
            
            username = input("Enter username: ").strip()
            if not username:
                print("❌ Username cannot be empty!")
                continue
            
            email = input("Enter email (optional): ").strip()
            email = email if email else None
            
            result = generator.register_user(username, email)
            
            if result['success']:
                print("\n🎉 SUCCESS! Your Gen2All account has been created!")
                print("─" * 50)
                print(f"👤 Username: {result['username']}")
                print(f"🆔 User ID: {result['user_id']}")
                print(f"🔑 API Key: {result['api_key']}")
                print(f"💎 Tier: {result['tier']}")
                print(f"📊 Quota: {'Unlimited' if result['quota_limit'] == -1 else result['quota_limit']}")
                print(f"⚡ Rate Limit: {result['rate_limit']} requests/hour")
                print("🎯 Features:")
                for feature in result['features']:
                    print(f"   ✓ {feature.replace('_', ' ').title()}")
                print("─" * 50)
                print("⚠️  IMPORTANT: Save your API key securely!")
                print("📝 You can now use the Gen2All Python client:")
                print("   from gen2all import Gen2AllClient")
                print(f"   client = Gen2AllClient('{result['api_key']}')")
                print("─" * 50)
            else:
                print(f"❌ Registration failed: {result['error']}")
        
        elif choice == '2':
            print("\n" + "─" * 40)
            print("👤 USER LOOKUP")
            print("─" * 40)
            
            username = input("Enter username: ").strip()
            if not username:
                print("❌ Username cannot be empty!")
                continue
            
            user = generator.get_user_by_username(username)
            
            if user:
                print(f"\n✅ User found!")
                print("─" * 40)
                print(f"👤 Username: {user['username']}")
                print(f"📧 Email: {user['email'] or 'Not provided'}")
                print(f"🆔 User ID: {user['user_id']}")
                print(f"🔑 API Key: {user['api_key']}")
                print(f"💎 Tier: {user['tier']}")
                print(f"📊 Quota Used: {user['quota_used']}")
                print(f"📈 Quota Limit: {'Unlimited' if user['quota_limit'] == -1 else user['quota_limit']}")
                print(f"⚡ Rate Limit: {user['rate_limit']} requests/hour")
                print(f"📅 Created: {time.ctime(user['created_at'])}")
                print(f"🔄 Last Login: {time.ctime(user['last_login'])}")
                print(f"🟢 Status: {'Active' if user['is_active'] else 'Inactive'}")
                print("─" * 40)
            else:
                print("❌ User not found!")
        
        elif choice == '3':
            print("\n" + "─" * 40)
            print("🔄 REGENERATE API KEY")
            print("─" * 40)
            
            username = input("Enter username: ").strip()
            if not username:
                print("❌ Username cannot be empty!")
                continue
            
            confirm = input(f"⚠️  Are you sure you want to regenerate API key for '{username}'? (yes/no): ").strip().lower()
            
            if confirm in ['yes', 'y']:
                result = generator.regenerate_api_key(username)
                
                if result['success']:
                    print(f"\n✅ API key regenerated successfully!")
                    print("─" * 40)
                    print(f"👤 Username: {result['username']}")
                    print(f"🔑 New API Key: {result['new_api_key']}")
                    print("─" * 40)
                    print("⚠️  Your old API key is now invalid!")
                    print("📝 Update your applications with the new key.")
                else:
                    print(f"❌ Failed to regenerate: {result['error']}")
            else:
                print("❌ Operation cancelled.")
        
        elif choice == '4':
            print("\n" + "─" * 40)
            print("📋 ALL USERS")
            print("─" * 40)
            
            result = generator.list_users()
            
            if result['success']:
                print(f"👥 Total Users: {result['total_users']}")
                print("─" * 40)
                
                for user in result['users']:
                    status = "🟢" if user['is_active'] else "🔴"
                    print(f"{status} {user['username']} ({user['tier']})")
                    print(f"   📧 {user['email'] or 'No email'}")
                    print(f"   📊 Used: {user['quota_used']} | Rate: {user['rate_limit']}/hr")
                    print(f"   📅 Created: {time.ctime(user['created_at'])}")
                    print()
                
                print("─" * 40)
            else:
                print("❌ Failed to retrieve users.")
        
        elif choice == '5':
            print("\n👋 Thanks for using Gen2All!")
            print("🚀 Build amazing AI applications with unlimited possibilities!")
            break
        
        else:
            print("❌ Invalid choice! Please select 1-5.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please try again or contact support.")
import os
import pymysql
from dotenv import load_dotenv
from argon2 import PasswordHasher
from urllib.parse import urlparse

load_dotenv()

password_hasher = PasswordHasher()

def get_password_hash(password):
    return password_hasher.hash(password)

def create_user(email: str, name: str, password: str):
    # Parse connection URL or use individual parameters
    db_url = os.getenv('DB_MARIADB_URL') or os.getenv('DB_MYSQL_URL')
    
    if db_url and (db_url.startswith('mysql://') or db_url.startswith('mariadb://')):
        parsed = urlparse(db_url)
        db_config = {
            'host': parsed.hostname or 'localhost',
            'port': parsed.port or 3306,
            'user': parsed.username,
            'password': parsed.password,
            'database': parsed.path.lstrip('/'),
            'charset': 'utf8mb4',
            'autocommit': False
        }
        connection = pymysql.connect(**db_config)
    else:
        # Use individual environment variables as fallback
        connection = pymysql.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', 3306)),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME'),
            charset='utf8mb4'
        )
    
    try:
        with connection.cursor() as cur:
            # Ensure users table exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    name VARCHAR(255),
                    hashed_password VARCHAR(255) NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            connection.commit()

            # Check if user already exists
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            existing_user = cur.fetchone()
            
            if existing_user:
                print(f"User {email} already exists!")
                return
            
            # Create new user
            hashed_password = get_password_hash(password)
            cur.execute(
                "INSERT INTO users (email, name, hashed_password) VALUES (%s, %s, %s)",
                (email, name, hashed_password)
            )
            user_id = cur.lastrowid
            connection.commit()
            print(f"User {email} created successfully with ID: {user_id}")
            
    except Exception as e:
        print(f"Error creating user: {e}")
    finally:
        connection.close()

if __name__ == "__main__":
    import sys
    import getpass
    
    if len(sys.argv) < 2:
        print("Usage: python create_user.py <email> [name]")
        print("Example: python create_user.py admin@example.com 'Admin User'")
        sys.exit(1)
    
    email = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else email.split('@')[0]  # Use part before @ as name if not provided
    
    # Securely prompt for password (hidden input)
    password = getpass.getpass("Enter password: ")
    password_confirm = getpass.getpass("Confirm password: ")
    
    if password != password_confirm:
        print("Passwords do not match!")
        sys.exit(1)
    
    if len(password) < 8:
        print("Password must be at least 8 characters long!")
        sys.exit(1)
    
    # Create the user
    create_user(email=email, name=name, password=password)
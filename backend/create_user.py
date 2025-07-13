import os
import psycopg
from dotenv import load_dotenv
from argon2 import PasswordHasher

load_dotenv()

password_hasher = PasswordHasher()

def get_password_hash(password):
    return password_hasher.hash(password)

def create_user(email: str, name: str, password: str):
    conn_info = os.getenv('DB_POSTGRES_URL')
    connection = psycopg.connect(conn_info)
    
    try:
        with connection.cursor() as cur:
            # Check if user already exists
            cur.execute("SELECT id FROM users WHERE email = %s", (email,))
            existing_user = cur.fetchone()
            
            if existing_user:
                print(f"User {email} already exists!")
                return
            
            # Create new user
            hashed_password = get_password_hash(password)
            cur.execute(
                "INSERT INTO users (email, name, hashed_password) VALUES (%s, %s, %s) RETURNING id",
                (email, name, hashed_password)
            )
            user_id = cur.fetchone()[0]
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
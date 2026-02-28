import sqlite3
import hashlib
import json
import os

DB_FILE = 'aura_wealth.db'

def hash_password(password: str) -> str:
    """Aplica hash SHA-256 a la contraseña."""
    return hashlib.sha256(password.encode()).hexdigest()

def init_db():
    """Inicializa las tablas de la base de datos si no existen."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Tabla de Usuarios
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Tabla de Carteras
    c.execute('''
        CREATE TABLE IF NOT EXISTS portfolios (
            username TEXT PRIMARY KEY,
            portfolio_json TEXT NOT NULL,
            FOREIGN KEY(username) REFERENCES users(username)
        )
    ''')
    conn.commit()
    conn.close()

def create_user(username, password) -> bool:
    """Crea un usuario nuevo. Retorna False si ya existe."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password) -> bool:
    """Verifica credenciales del usuario."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    if row and row[0] == hash_password(password):
        return True
    return False

def save_portfolio(username: str, portfolio_dict: dict):
    """Guarda o actualiza la cartera en formato JSON para un usuario."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    p_json = json.dumps(portfolio_dict)
    c.execute('''
        INSERT INTO portfolios (username, portfolio_json) 
        VALUES (?, ?) 
        ON CONFLICT(username) DO UPDATE SET portfolio_json=?
    ''', (username, p_json, p_json))
    conn.commit()
    conn.close()

def load_portfolio(username: str) -> dict:
    """Carga la cartera de un usuario. Retorna {} si no existe."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT portfolio_json FROM portfolios WHERE username = ?', (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return {}

# Inicializa el motor al arrancar
init_db()

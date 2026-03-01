import hashlib
import json
import os
import requests
from datetime import datetime

# API oficial de SheetDB conectada al Excel de Miguel
SHEETDB_URL = "https://sheetdb.io/api/v1/q8oe02avgnyth"

def hash_password(password: str) -> str:
    """Aplica hash SHA-256 a la contraseña por seguridad."""
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username: str, password: str, accept_terms: str = "Términos Aceptados") -> bool:
    """Crea un usuario nuevo en Google Sheets vía API. Retorna False si ya existe o hay error de red."""
    try:
        # Validación de existencia
        response = requests.get(f"{SHEETDB_URL}/search?Correo={username}")
        if response.status_code == 200 and len(response.json()) > 0:
            return False # El correo ya se registró

        # Registrar en Excel
        payload = {
            "data": [{
                "Correo": username,
                "Contrasena_Codificada": hash_password(password),
                "Fecha_Registro": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Acepta_Terminos": accept_terms
            }]
        }
        res = requests.post(SHEETDB_URL, json=payload)
        return res.status_code == 201
    except Exception as e:
        print(f"Error de base de datos en la nube: {e}")
        return False

def authenticate_user(username: str, password: str) -> bool:
    """Verifica credenciales del usuario en Google Sheets."""
    try:
        response = requests.get(f"{SHEETDB_URL}/search?Correo={username}")
        if response.status_code == 200:
            users = response.json()
            if len(users) > 0:
                user_data = users[0]
                if user_data.get("Contrasena_Codificada") == hash_password(password):
                    return True
        return False
    except Exception as e:
        print(f"Error de base de datos en la nube: {e}")
        return False

# Funciones desactivadas hasta versión Pro de portafolios (requerirían otra pestaña de Excel dedicada)
def save_portfolio(username: str, portfolio_dict: dict):
    pass

def load_portfolio(username: str) -> dict:
    return {}

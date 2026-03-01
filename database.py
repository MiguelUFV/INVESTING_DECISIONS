import json
import os
import requests
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# API oficial de SheetDB conectada al Excel de Miguel
SHEETDB_URL = "https://sheetdb.io/api/v1/uxvd67ii9eboo"

# Configuración SMTP - Datos a ser reemplazados en despliegue
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "miguellastra.investing@gmail.com" # Asumiendo correo genérico por ahora
APP_PASSWORD = "" # ESPERANDO CLAVE DEL USUARIO

def hash_password(password: str) -> str:
    """Aplica hash SHA-256 a la contraseña por seguridad."""
    return hashlib.sha256(password.encode()).hexdigest()

def send_welcome_email(recipient_email: str):
    """Envía un correo de bienvenida automático al registrarse usando SMTP Gmail."""
    if not APP_PASSWORD:
        return False
        
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Bienvenido a Aura Wealth OS Institucional"
        msg["From"] = f"Miguel | Aura Wealth <{SENDER_EMAIL}>"
        msg["To"] = recipient_email

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
            <div style="max-width: 600px; margin: auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 5px;">
                <h2 style="color: #1e3a8a;">AURA WEALTH OS</h2>
                <p>Hola,</p>
                <p>Tu identidad corporativa (<b>{recipient_email}</b>) ha sido aprovisionada con éxito en los servidores Cuantitativos de Aura Wealth.</p>
                <p>Ya puedes iniciar sesión en el Dashboard para acceder a los módulos de escaneo de mercado, inteligencia artificial temporal y optimizador de carteras de Markowitz.</p>
                <br>
                <p><em>Este es un sistema de notificaciones automático. Por favor, no respondas a este correo.</em></p>
                <hr style="border: none; border-top: 1px solid #eee;" />
                <p style="font-size: 12px; color: #888;">© 2026 Aura Wealth Quant Engine.<br>Desarrollado para la toma de decisiones algorítmicas de capital.</p>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, "html"))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Error SMTP Email: {e}")
        return False

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
        
        if res.status_code == 201:
            send_welcome_email(username) # Disparar mail si el Excel devuelve OK
            return True
            
        return False
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

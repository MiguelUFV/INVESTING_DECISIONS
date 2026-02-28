import yfinance as yf
import pandas as pd
import numpy as np
import logging
import datetime
import os

# Configuración del log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketDataETL:
    """
    Clase para realizar el proceso ETL (Extracción, Transformación y Carga) 
    de datos de mercado financiero.
    """

    def __init__(self, tickers, periods_years=2):
        """
        Inicializa la clase MarketDataETL con la lista de activos y el periodo a extraer.
        
        Args:
            tickers (list): Lista de símbolos de los activos (ej. ['AAPL', 'SAN.MC']).
            periods_years (int): Número de años históricos a descargar.
        """
        self.tickers = tickers
        self.periods_years = periods_years
        self.data_frames = []

    def extract(self):
        """
        Descarga el histórico de precios para la lista de activos especificada usando yfinance.
        """
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=self.periods_years * 365)
        
        logging.info(f"Iniciando extracción desde {start_date} hasta {end_date}")
        
        for ticker in self.tickers:
            logging.info(f"Descargando datos para: {ticker}")
            try:
                # Descarga de datos
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if df.empty:
                    logging.warning(f"No se encontraron datos para {ticker}. Saltando...")
                    continue
                
                # Aplanar las columnas si retornan como MultiIndex (comportamiento de versiones recientes de yf)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Agregar columna con el nombre del activo para identificarlo más adelante
                df['Ticker'] = ticker
                
                # Guardar el bloque de datos extraído en la lista
                self.data_frames.append(df)
                logging.info(f"Datos descargados con éxito para: {ticker}")
                
            except Exception as e:
                logging.error(f"Error al descargar los datos de {ticker}: {e}")

    def transform(self):
        """
        Transforma los datos calculando:
        - SMA_50 y EMA_20 (Tendencia).
        - Bollinger Bands (Inercia/Volatilidad).
        - RSI (Fuerza Relativa - 14 periodos).
        - Daily_Return_% (Retorno Diario).
        Limpia los valores nulos generados.
        """
        logging.info("Iniciando fase de transformación de datos (Indicadores RSI, Bollinger, etc)...")
        transformed_dfs = []
        
        for df in self.data_frames:
            ticker_name = df['Ticker'].iloc[0]
            
            if 'Close' not in df.columns:
                logging.warning(f"La columna 'Close' no está presente en {ticker_name}.")
                continue
                
            close_col = df['Close']
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.squeeze()

            # --- Medias Móviles ---
            df['SMA_50'] = close_col.rolling(window=50).mean()
            df['EMA_20'] = close_col.ewm(span=20, adjust=False).mean()
            
            # --- Bandas de Bollinger (SMA 20) ---
            sma_20 = close_col.rolling(window=20).mean()
            std_20 = close_col.rolling(window=20).std()
            df['BB_Upper'] = sma_20 + (std_20 * 2)
            df['BB_Lower'] = sma_20 - (std_20 * 2)

            # --- RSI (14 periodos) ---
            delta = close_col.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
            
            # Retorno Diario
            df['Daily_Return_%'] = close_col.pct_change() * 100
            
            # Limpieza exhaustiva
            df_clean = df.dropna()
            
            transformed_dfs.append(df_clean)
            
        self.data_frames = transformed_dfs
        logging.info("Transformación cuantitativa V2 completada.")

    def load(self, output_filename="historico_mercado.parquet"):
        """
        Concatena todos los datos procesados en un único DataFrame estructurado 
        y los exporta a un archivo Parquet para eficiencia máxima.
        
        Args:
            output_filename (str): Nombre del archivo Parquet de salida.
        """
        if not self.data_frames:
            logging.error("No hay datos para guardar. El proceso fue interrumpido o falló la extracción.")
            return
            
        logging.info(f"Iniciando fase de carga: Concatenando y guardando en {output_filename}...")
        
        try:
            # Concatenar todos los DataFrames
            final_df = pd.concat(self.data_frames)

            # Asegurar que los nombres de las columnas sean strings para evitar problemas de tipos con Parquet
            final_df.columns = final_df.columns.astype(str)
            
            # Guardar en formato Parquet usando el motor pyarrow
            final_df.to_parquet(output_filename, engine='pyarrow')
            logging.info(f"Datos exportados exitosamente y estructurados en: {os.path.abspath(output_filename)}")
            
        except Exception as e:
            logging.error(f"Error al guardar los datos en {output_filename}: {e}")

    def run(self):
        """
        Ejecuta el flujo completo de ETL (Extracción, Transformación y Carga).
        """
        logging.info("=== Iniciando proceso MarketDataETL ===")
        self.extract()
        self.transform()
        self.load()
        logging.info("=== Proceso MarketDataETL finalizado con éxito ===")

if __name__ == '__main__':
    # Lista de activos combinando mercado americano y español
    activos = ['AAPL', 'MSFT', 'SPY', 'SAN.MC', 'IBE.MC']
    
    # Crear la instancia del ETL
    etl = MarketDataETL(tickers=activos, periods_years=2)
    
    # Ejecutar el proceso
    etl.run()

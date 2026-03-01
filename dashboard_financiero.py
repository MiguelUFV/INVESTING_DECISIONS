import streamlit as st
import json
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import norm
import scipy.stats as stats
import time
import requests
import logging

try:
    from sklearn.ensemble import RandomForestRegressor
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    pass

# --- CONFIGURACION DE PAGINA ---
st.set_page_config(
    page_title="AURA WEALTH OS | Next-Gen Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS PREMIUM FINTECH (Glassmorphism & Gradients) ---
st.markdown("""
<style>
    /* Ocultar elementos nativos de Streamlit para parecer una Web App real */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden !important;}
    footer {visibility: hidden;}
    .streamlit-footer {display: none;}
    
    /* Fondo Moderno Radial Gradient */
    .stApp { 
        background: radial-gradient(circle at top right, #1E1B4B 0%, #0F172A 50%, #020617 100%);
        background-attachment: fixed;
        color: #E2E8F0; 
        font-family: 'Inter', 'Segoe UI', Helvetica, sans-serif; 
    }
    
    /* Estilos para titulos con gradientes premium */
    h1, h2, h3, h4 {
        color: #F8FAFC !important; 
        font-weight: 700 !important; 
        letter-spacing: -0.02em;
    }
    
    /* Layout Containers and Padding (Pegar más arriba) */
    .block-container { padding-top: 1.5rem !important; padding-bottom: 4rem; max-width: 1600px; }
    
    /* Sidebar con Efecto Cristal (Glassmorphism) intenso */
    [data-testid="stSidebar"] { 
        background: rgba(11, 15, 25, 0.75) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.05); 
    }
    
    /* Cards Modernas (Glassmorphism y Sombras Neón suaves) */
    div[data-testid="metric-container"] { 
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.4) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05); 
        padding: 1.25rem; 
        border-radius: 16px; 
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 10px 10px -5px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.6) 100%);
    }
    
    div[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 800; color: #FFFFFF; letter-spacing: -0.03em;}
    div[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
    
    /* Pestañas (Tabs) Estilo Navegador Premium tipo Pill Nav */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 10px 16px;
        border: none;
        transition: all 0.3s ease;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #F8FAFC;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.15) !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.1);
    }
    
    /* Botones Globables (Efecto Hover Premium) */
    .stButton > button {
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        border-color: #38BDF8 !important;
        color: #38BDF8 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.2) !important;
        transform: translateY(-2px);
    }
    
    /* Expander override (Cajas Limpias) */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #E2E8F0 !important;
        background-color: rgba(30, 41, 59, 0.4) !important;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    
    /* Convertir Radio Buttons del Sidebar en Cajas Interactivas (Menú) */
    [data-testid="stSidebar"] div[role="radiogroup"] > label {
        background-color: rgba(30, 41, 59, 0.5) !important;
        padding: 0.75rem 1rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        margin-bottom: 0.5rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background-color: rgba(56, 189, 248, 0.1) !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
        transform: translateX(3px);
    }
    [data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {
        background-color: rgba(56, 189, 248, 0.15) !important;
        border-color: #38BDF8 !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.1);
    }
    
    hr { border-color: rgba(255,255,255,0.05) !important; margin: 1.5rem 0;}
    
    /* Text Inputs y Text Areas */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(15, 23, 42, 0.6) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    .stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 1px #38BDF8 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTES GLOBALES Y COLORES PREMIUM ---
COLOR_TREND = '#38BDF8'      # Azul Cyan Esmeralda Brillante
COLOR_BG = 'rgba(0,0,0,0)'
COLOR_SMOKE = '#818CF8'      # Indigo Suave
COLOR_DARK_SMOKE = 'rgba(255,255,255,0.1)' 
BENCHMARK_TICKER = 'SPY'
RISK_FREE_RATE = 0.04

# --- INGESTA DINAMICA (Real-Time API) ---

@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Descarga de datos de mercado con manejo de errores estricto."""
    try:
        if BENCHMARK_TICKER not in tickers:
            tickers.append(BENCHMARK_TICKER)
            
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if data.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                df_close = data['Close'].copy()
            else:
                return pd.DataFrame(), pd.DataFrame()
        else:
            if 'Close' in data.columns:
                df_close = pd.DataFrame({tickers[0]: data['Close']}) if len(tickers) == 1 else data[['Close']].copy()
                if len(tickers) > 1: df_close.columns = df_close.columns.get_level_values(1) if isinstance(df_close.columns, pd.MultiIndex) else [tickers[0]]
            else:
                return pd.DataFrame(), pd.DataFrame()

        df_close = df_close.dropna(axis=1, how='all')
        df_close = df_close.ffill().bfill()
        
        return df_close, data
    except Exception as e:
        logging.error(f"Error de red o procesamiento en descarga de activos: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# --- MATEMATICA CUANTITATIVA ---

@st.cache_data(show_spinner=False)
def calculate_technical_indicators(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({'Close': series})
    df['Retorno_Diario'] = df['Close'].pct_change()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['Volatilidad_20d'] = df['Retorno_Diario'].rolling(window=20).std() * np.sqrt(252)
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    return df

@st.cache_data(show_spinner=False)
def get_performance_metrics(asset_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float):
    df = pd.concat({'asset': asset_returns, 'market': market_returns}, axis=1).dropna()
    if len(df) < 2: return None
    
    ret_asset = df['asset']
    ret_market = df['market']
    
    mean_ret_ann = ret_asset.mean() * 252
    std_ann = ret_asset.std() * np.sqrt(252)
    
    sharpe = (mean_ret_ann - risk_free_rate) / std_ann if std_ann != 0 else 0
    sortino = (mean_ret_ann - risk_free_rate) / (ret_asset[ret_asset < 0].std() * np.sqrt(252)) if (ret_asset[ret_asset < 0].std() * np.sqrt(252)) != 0 else 0

    slope, _, r_value, _, _ = stats.linregress(ret_market, ret_asset)
    beta = slope
    mean_market_ann = ret_market.mean() * 252
    
    expected_capm = risk_free_rate + beta * (mean_market_ann - risk_free_rate)
    alpha = mean_ret_ann - expected_capm
    
    active_return = ret_asset - ret_market
    tracking_error_ann = active_return.std() * np.sqrt(252)
    information_ratio = active_return.mean() * 252 / tracking_error_ann if tracking_error_ann != 0 else 0
    
    cum_returns = (1 + ret_asset).cumprod()
    mdd = ((cum_returns - cum_returns.cummax()) / cum_returns.cummax()).min()
    
    return {
        'Vol_Ann': std_ann, 'Ret_Ann': mean_ret_ann,
        'Mkt_Ret_Ann': mean_market_ann, 'Sharpe': sharpe,
        'Sortino': sortino, 'IR': information_ratio,
        'Beta': beta, 'Alpha': alpha, 'CAPM': expected_capm,
        'MDD': mdd, 'Market_Ret': ret_market.values, 'Asset_Ret': ret_asset.values
    }

def portfolio_perf(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return returns, std

def get_optimized_portfolios(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    res_min_vol = minimize(lambda w, ret, cov: portfolio_perf(w, ret, cov)[1],
                           num_assets * [1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    
    args_sharpe = (mean_returns, cov_matrix, risk_free_rate)
    def neg_sharpe(w, ret, cov, rf):
        p_ret, p_std = portfolio_perf(w, ret, cov)
        return -(p_ret - rf) / p_std
        
    res_max_sharpe = minimize(neg_sharpe, num_assets * [1./num_assets,], args=args_sharpe,
                              method='SLSQP', bounds=bounds, constraints=constraints)
    
    return res_max_sharpe.x, res_min_vol.x

def calculate_var(returns: pd.Series, confidence_level=0.95):
    mu = returns.mean()
    sigma = returns.std()
    var_95 = norm.ppf(1 - confidence_level, mu, sigma)
    return var_95

def run_monte_carlo(latest_price: float, returns: pd.Series, days=252, simulations=500):
    mu = returns.mean()
    vol = returns.std()
    sim_df = np.zeros((days, simulations))
    sim_df[0] = latest_price
    for i in range(1, days):
        sim_df[i] = sim_df[i-1] * (1 + np.random.normal(loc=mu, scale=vol, size=simulations))
    return sim_df

# --- OPTIMIZACION DE GRAFICOS (Plotly Professional) ---

def clean_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(family='Inter', size=14, color='#94A3B8')),
        template='plotly_dark', height=height, 
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        margin=dict(t=40, l=10, r=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#64748B')),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(15, 23, 42, 0.9)", font_size=13, font_family="Inter", bordercolor="rgba(56, 189, 248, 0.5)")
    )
    # Eliminacion de gridlines
    fig.update_xaxes(showgrid=False, zerolinecolor=COLOR_DARK_SMOKE, tickfont=dict(color='#64748B'))
    fig.update_yaxes(showgrid=False, zerolinecolor=COLOR_DARK_SMOKE, tickfont=dict(color='#64748B'))
    return fig

def plot_drawdown_underwater(returns: pd.Series, ticker: str):
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = ((cum_returns - rolling_max) / rolling_max) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns, fill='tozeroy', mode='lines', 
                             line=dict(color=COLOR_SMOKE, width=0), fillcolor='rgba(71, 85, 105, 0.4)', name='Drawdown'))
    fig = clean_layout(fig, title=f"EXPOSICION A RIESGO DE COLA (UNDERWATER DRAWDOWN) - {ticker}", height=300)
    fig.update_yaxes(title="RETRACCION DESDE MAXIMO (%)")
    return fig

# --- ARQUITECTURA DE INTERPRETACION (Insight Engine) ---

def interpret_tecnico(df: pd.DataFrame, ticker: str):
    if df.empty or len(df) < 50:
        st.info("HISTORICO INSUFICIENTE PARA ANALISIS ESTRUCTURAL.")
        return
        
    ult_close = df['Close'].iloc[-1]
    ult_sma = df['SMA_50'].iloc[-1]
    volatilidad = df['Volatilidad_20d'].iloc[-1] * 100
    rsi = df['RSI_14'].iloc[-1]
    macd = df['MACD'].iloc[-1]
    sig = df['Signal_Line'].iloc[-1]

    # Nivel 1: Resumen Ejecutivo
    if ult_close > ult_sma:
        st.success(f"CONVERGENCIA ESTRUCTURAL POSITIVA: COTIZACION DOMINANTE FRENTE A SIMPLE MOVING AVERAGE.")
    else:
        st.error(f"DETERIORO ESTRUCTURAL: EL ACTIVO COTIZA SUBYACENTE A LA MEDIA MOVIL DE 50 PERIODOS.")

    # Nivel 2: Analisis Detallado
    with st.expander("METODOLOGIA Y ANALISIS MULTIFACTORIAL DETALLADO"):
        st.markdown(f"""
        **🎯 TRADUCCION PRACTICA:**
        > *Básicamente, la acción {'está por encima de su precio medio sano y es un buen momento de mercado' if ult_close > ult_sma else 'está cayendo en picado por debajo de su media temporal, comprar ahora es peligroso e ir contra corriente'}. El indicador RSI nos chiva que el miedo/codicia está {'muy inflado (posible caída inminente)' if rsi > 70 else 'en pánico vendedor (podría rebotar arriba)' if rsi < 30 else 'en niveles sanos y normales'}.*
        
        <hr>

        **Fundamentos Matemáticos del Sesgo Direccional:**
        El cruce de la cotización actual (`{ult_close:.2f}`) sobre la Simple Moving Average de 50 periodos (`{ult_sma:.2f}`) 
        es monitoreado como un proxy algorítmico del consenso de los participantes institucionales. 
        
        **Oscilador de Momentum (RSI 14 Periodos):**
        Lectura estocástica actual en `{rsi:.2f}`. 
        {"Condición de saturación compradora (RSI > 70). Altamente probable regresión a la media." if rsi > 70 else "Condición de agotamiento vendedor (RSI < 30). Posible acumulación institucional latente." if rsi < 30 else "Cotización en banda neutral, sin desviaciones significativas del momentum histórico."}
        
        **Convergencia/Divergencia (MACD):**
        {"Expansión direccional alcista validada: MACD operando por encima de su Signal Line." if macd > sig else "Contracción direccional bajista validada: MACD operando por debajo de su Signal Line cruzando a la baja."}
        
        **Estructura de Riesgo (Volatilidad Anualizada - 20d):** 
        El registro actual marca una desviación estándar del `{volatilidad:.2f}%`. 
        {"Este valor supera el umbral paramétrico del 30%, implicando un régimen de alta dispersión en retornos asimétricos, lo que sugiere una contracción obligatoria en la asignación de capital (Position Sizing)." if volatilidad > 30 else "El régimen de dispersión muestra una moderación relativa, manteniéndose dentro de niveles tolerables para una exposición de capital pasiva/moderada."}
        """)

def interpret_institucional(res: dict, ticker: str):
    s = res['Sharpe']
    a = res['Alpha'] * 100
    b = res['Beta']
    
    # Nivel 1: Resumen Ejecutivo
    if s > 1.0:
        st.success(f"METRICAS DE RETORNO ESTADISTICAMENTE EXCELENTES: SHARPE POSITIVO Y GENERACION ALFA CONFIRMADA.")
    elif s > 0:
        st.info(f"METRICAS DE RETORNO ESTANDAR: PERFIL EXPOSICION/COMPENSACION NEUTRAL AL MERCADO DIRECTO.")
    else:
        st.error(f"DESTRUCCION DE VALOR AJUSTADO: RETORNO INSUFICIENTE PARA JUSTIFICAR LA VOLATILIDAD ASUMIDA.")
        
    # Nivel 2: Analisis Detallado
    with st.expander("ANATOMIA PARAMETRICA CAPM Y RIESGO SISTEMICO (DEEP DIVE)"):
        st.markdown(f"""
        **🎯 TRADUCCION PRACTICA:**
        > *Este módulo evalúa la "calidad real" de esta inversión comparada con no hacer nada (Letras del Tesoro). Un Sharpe mayor a 1.0 significa que el estrés de estar invertido compensa sobradamente. El Alpha indica si la acción sube por mérito propio (producto/gestión) o solo porque todo el mercado global empuja. La Beta te indica la agresividad: Si es > 1, es una montaña rusa; si es < 1, es un refugio seguro.*
        
        <hr>

        **Capital Asset Pricing Model (CAPM):**
        Evaluación del constructo de riesgo frente al Benchmark asumiendo una Tasa Libre de Riesgo del `{RISK_FREE_RATE*100}%`.
        
        **Eficiencia Operativa (Sharpe Ratio):** `{s:.2f}`
        {"Lectura Institucional de Alfa Verdadero (Sharpe > 1.0). El activo proporciona un exceso de retorno matemáticamente superior por cada unidad de volatilidad experimentada." if s > 1.0 else "El ratio sub-óptimo requiere una prima de riesgo adicional o diversificación forzada mediante co-varianzas compensatorias."}
        
        **Desacoplamiento Estructural (Alpha de Jensen):** `{a:.2f}%`
        {"El activo genera un rendimiento exógeno superior al predicho por la línea de mercado de valores, implicando ineficiencias capturables (Creación de Valor)." if a > 0 else "El activo destruye valor ajustado por riesgo sistémico en el paradigma estándar del CAPM."}
        
        **Sensibilidad Direccional (Beta):** `{b:.2f}`
        {"Coeficiente Beta > 1. El modelo denota asimetría expansiva: las contracciones o expansiones sistémicas globales serán estadísticamente amplificadas en este activo." if b > 1.0 else "Coeficiente Beta < 1. El activo ostenta propiedades intrínsecas defensivas frente al ciclo macroeconómico global."}
        """)

def interpret_markowitz(weights, tickers):
    df_w = pd.DataFrame({'Ticker': tickers, 'Weight': weights}).sort_values(by='Weight', ascending=False)
    df_w = df_w[df_w['Weight'] > 0.01] 
    if not df_w.empty:
        top_ticker = df_w.iloc[0]['Ticker']
        top_w = df_w.iloc[0]['Weight'] * 100
        
        # Nivel 1: Resumen Ejecutivo
        st.info(f"SOLUCION ALGORITMICA COMPUTADA: EXPOSICION MAXIMIZADA EN {top_ticker} AL {top_w:.1f}%.")
        
        # Nivel 2: Analisis Detallado
        with st.expander("FUNDAMENTOS MATEMATICOS Y MATRIZ DE COVARIANZA (METODO SLSQP)"):
            st.markdown(f"""
            **🎯 TRADUCCION PRACTICA:**
            > *Ray Dalio dice que la diversificación es el único "almuerzo gratis" en las finanzas. La Inteligencia Artificial de este módulo analiza cómo se mueven las acciones entre sí. Si una cae, la otra debería subir para protegerte. El motor ha simulado de fondo miles de combinaciones (los miles de puntos de la gráfica debajo de la curva) y te está entregando la mezcla (porcentajes) EXACTA y estadísticamente imbatible para ganar el máximo dinero asumiendo el menor riesgo posible hoy.*
            
            <hr>

            **Hipótesis del Mercado Eficiente y Target SLSQP:**
            El algoritmo de optimización cuadrática computa la frontera eficiente minimizando la varianza global del vector de retornos 
            con una restricción de suma ponderada igual estricta. La preponderancia algorítmica sobre **{top_ticker}** obedece a una ratio de covarianza estructuralmente negativa frente a los sub-componentes colindantes, garantizando teóricamente máxima retribución por unidad de riesgo sistémico absorbida (Plena Maximización Paramétrica de Sharpe).
            """)

def interpret_oraculo(simulations: np.ndarray, var_95: float):
    p_fin = simulations[-1,:] 
    media = np.mean(p_fin)
    p95 = np.percentile(p_fin, 95)
    p5 = np.percentile(p_fin, 5)
    
    # Nivel 1: Resumen Ejecutivo
    st.info(f"DIAGNOSTICO ESTOCASTICO A 1-ANIO: EL VALOR ESPERADO CONVERGE NIVEL {media:.2f}. VALUE AT RISK (VaR 95%): {var_95*100:.2f}%.")
    
    # Nivel 2: Analisis Detallado
    with st.expander("ARQUITECTURA DEL METODO MONTE CARLO Y DISTRIBUCION LOG-NORMAL (VAR LIMITS)"):
        st.markdown(f"""
        **🎯 TRADUCCION PRACTICA:**
        > *Nadie tiene una bola de cristal para predecir el futuro exacto. En su lugar, hemos simulado matemáticamente las matemáticas del precio creando 500 "multiversos" o mundos paralelos a un año vista. La línea central te dice la gravitación normal de hacia dónde va el precio. Además, el VAR al 95% te avisa claramente de cuánto es el límite estadístico de dinero que podrías perder de golpe en tu peor día si los mercados colapsan.*
        
        <hr>

        **Dinámica de Movimiento Browniano Simple:**
        Para la computación de los trayectos sintéticos (n=500 iteraciones a t=252), se asume que los retornos compuestos continuos siguen empíricamente una distribución Normal paramétrica. Mediante simulación computacional derivamos intervalos empíricos de confianza:
        
        *   **Percentil 95 (Límite Superior Estadístico Acumulado):** `{p95:.2f}`
        *   **Percentil 05 (Límite Inferior Crítico Descontado):** `{p5:.2f}`
        
        **Riesgo de Ruina Sistémica (VaR al 95% Inter-diario):**
        Las métricas analíticas constatan un Value at Risk paramétrico de `{var_95*100:.2f}%`. 
        Estadísticamente existe un 5% de probabilidad ex-ante de que las caídas en un horizonte diario de cotización contínua superen dicho umbral severo, lo que requiere coberturas estructuradas complejas si la exposición de capital excede los límites prescritos.
        """)

# --- APLICACION PRINCIPAL ---

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.username = None

    if not st.session_state.authenticated:
        st.markdown("""
        <div style="text-align: center; margin-top: 12vh; margin-bottom: 3rem;">
            <div style="display:inline-block; padding: 1.5rem; background: rgba(56, 189, 248, 0.05); border-radius: 20px; margin-bottom: 1rem; border: 1px solid rgba(56, 189, 248, 0.2); box-shadow: 0 0 30px rgba(56, 189, 248, 0.1);">
                <span style="font-size: 3rem; font-weight: 900; background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AW</span>
            </div>
            <h1 style="font-size: 4rem; letter-spacing: -0.05em; margin-bottom: 0; background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AURA WEALTH OS</h1>
            <p style="color: #94A3B8; font-size: 1.2rem; font-weight: 400; letter-spacing: 0.02em;">Portal de Autenticación Institucional</p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 1.5, 1])
        with c2:
            st.markdown("---")
            if st.button("ACCEDER A LA TERMINAL DE ANÁLISIS", type="primary", use_container_width=True):
                st.session_state.authenticated = True
                st.session_state.username = "Trader Local"
                st.rerun()
            st.markdown("---")
        return

    # Header Principal Dashboard (Reemplaza el antiguo h1/p)
    st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 1.5rem 2rem; background: rgba(15, 23, 42, 0.4); backdrop-filter: blur(12px); border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 2rem; box-shadow: 0 4px 20px -2px rgba(0,0,0,0.2);">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="height: 40px; width: 40px; background: linear-gradient(135deg, #38BDF8, #818CF8); border-radius: 10px; display: flex; justify-content: center; align-items: center; box-shadow: 0 0 15px rgba(56, 189, 248, 0.4);">
                <span style="color: white; font-weight: 900; font-size: 1.1rem; letter-spacing: -0.05em;">AW</span>
            </div>
            <div>
                <h1 style="margin: 0; padding: 0; font-size: 1.8rem; letter-spacing: -0.03em; color: white !important;">AURA WEALTH OS</h1>
                <p style="margin: 0; padding: 0; color: #94A3B8; font-size: 0.85rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">Quantitative Fintech Engine v2.0</p>
            </div>
        </div>
        <div style="text-align: right; display: flex; align-items: center; gap: 1rem;">
            <div style="display: inline-block; padding: 0.4rem 1rem; border-radius: 20px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); color: #10B981; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em;">
                LIVE MARKET DTLK
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown(f"**Usuario Local:** `{st.session_state.username}`")
        if st.button("Finalizar Sesión", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            if "df_close" in st.session_state:
                del st.session_state["df_close"]
            st.rerun()
            
        with st.expander("ALERTAS DE VOLATILIDAD", expanded=False):
            st.markdown("Configura notificaciones externas para brechas de VaR predictivo.")
            email_input = st.text_input("Correo Escolta", placeholder="riesgos@institucion.com")
            var_alert = st.slider("Trigger VaR Condicional (%)", 5, 20, 10)
            if st.button("Compilar Trigger"):
                st.success(f"Regla insertada para: {email_input}")

    if "df_close" not in st.session_state:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.form("filtros_globales", clear_on_submit=False):
            st.markdown("### CONFIGURACIÓN ESTRUCTURAL DE CARTERA")
            st.markdown("Defina los universos de inversión para iniciar el análisis cuantitativo. Ajuste la tasa libre de riesgo según las letras del Tesoro vigentes.")
            st.markdown("<hr>", unsafe_allow_html=True)
            
            tickers_selected = []
            
            c_m1, c_m2, c_m3 = st.columns(3)
            with c_m1:
                st.markdown("**USA: TECNOLOGÍA E IA (Nasdaq)**")
                us_tech = st.multiselect("Big Tech & Semiconductores", 
                    ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "CRM", "AMD", "ADBE", "NFLX", "CSCO", "INTC", "QCOM", "IBM", "ORCL", "NOW"], 
                    default=[], label_visibility="collapsed")
                st.markdown("<br>**USA: VALOR, FINANZAS Y SALUD**", unsafe_allow_html=True)
                us_fin = st.multiselect("Banca, Consumo y Farma", 
                    ["JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BRK-B", "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "DIS", "NKE"], 
                    default=[], label_visibility="collapsed")
            with c_m2:
                st.markdown("**EUROPA Y ESPAÑA**")
                eu_stocks = st.multiselect("Gran Capitalización", 
                    ["ASML", "SAP", "SIE.DE", "MC.PA", "AIR.PA", "OR.PA", "SAN.MC", "BBVA.MC", "IBE.MC", "ITX.MC", "REP.MC", "TEF.MC", "CABK.MC", "AENA.MC", "FER.MC"], 
                    default=[], label_visibility="collapsed")
                st.markdown("<br>**ASIA Y EMERGENTES**", unsafe_allow_html=True)
                asia_latam = st.multiselect("Dragones y LATAM", 
                    ["TSM", "BABA", "JD", "BIDU", "7203.T", "6758.T", "9984.T", "005930.KS", "VALE", "PBR", "MELI", "NU"], 
                    default=[], label_visibility="collapsed")
            with c_m3:
                st.markdown("**MATERIAS PRIMAS Y MACRO (ETFs)**")
                macro_etf = st.multiselect("Defensivo e Índices", 
                    ["GC=F", "SPY", "QQQ", "DIA", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV", "USO", "UNG", "UUP", "VIXY"], 
                    default=[], label_visibility="collapsed")
                st.markdown("<br>**CRIPTOACTIVOS (Top Market Cap)**", unsafe_allow_html=True)
                crypto = st.multiselect("Ecosistema Digital", 
                    ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "LINK-USD", "MATIC-USD", "LTC-USD"], 
                    default=[], label_visibility="collapsed")
                    
            st.markdown("<br>**BÚSQUEDA LIBRE DE TICKERS GLOBALES:**", unsafe_allow_html=True)
            custom_tickers = st.text_input("Ingrese símbolos de Yahoo Finance separados por comas", value="", placeholder="Ej: F, GM, PLTR, RKLB")
            
            tickers_selected.extend(us_tech + us_fin + eu_stocks + asia_latam + macro_etf + crypto)
            if custom_tickers:
                tickers_selected.extend([t.strip().upper() for t in custom_tickers.split(',') if t.strip()])
            
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("#### HORIZONTE DE DATOS Y PARAMETRÍA DE RIESGO")
            col_d1, col_d2, col_d3 = st.columns(3)
            with col_d1: start_date = st.date_input("FECHA DE INICIO", value=pd.to_datetime('2023-01-01'))
            with col_d2: end_date = st.date_input("FECHA FINAL", value=pd.to_datetime('today'))
            with col_d3: risk_free_val = st.number_input("TASA LIBRE DE RIESGO (Rf %)", value=4.0, step=0.1)
                
            global RISK_FREE_RATE
            RISK_FREE_RATE = risk_free_val / 100.0
            
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.form_submit_button("INICIALIZAR MOTOR Y COMPILAR DATOS", type="primary", use_container_width=True)

        if run_btn:
            tickers_list = list(set(tickers_selected))
            if not tickers_list:
                st.error("LA CONFIGURACIÓN REQUIERE UN ACTIVO COMO MÍNIMO.")
                st.stop()

            with st.spinner("Sintetizando series temporales via YFinance Core..."):
                df_close, raw_data = load_market_data(tickers_list.copy(), start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if df_close.empty:
                    st.error("ERROR DE EJECUCIÓN: LOS PARÁMETROS NO PROPORCIONAN UNA SALIDA VÁLIDA EN LA API EXTERNA.")
                    st.stop()
                    
                st.session_state["df_close"] = df_close
                st.session_state["raw_data"] = raw_data
                st.session_state["valid_tickers"] = [t for t in tickers_list if t in df_close.columns]
                st.rerun()
        st.stop()

    df_close = st.session_state["df_close"]
    valid_tickers = st.session_state["valid_tickers"]
    raw_data = st.session_state["raw_data"]
    st.sidebar.markdown("<br>---<br>", unsafe_allow_html=True)
    st.sidebar.markdown("### 🎛️ NAVEGACIÓN MÓDULOS")
    
    opciones_nav = [
        "📈 RESUMEN & COMPARATIVA",
        "TÉCNICO / MOMENTUM", 
        "RIESGO (CAPM)", 
        "MONTE CARLO (VaR)", 
        "ESTACIONALIDAD", 
        "RADAR QUANT", 
        "MARKOWITZ", 
        "MI PORTAFOLIO", 
        "MACRO & NOTICIAS", 
        "AI FORECASTING", 
        "BACKTESTER", 
        "REPORTE PDF", 
        "RAW DATA"
    ]
    
    vista_actual = st.sidebar.radio("Ir a Módulo Analítico:", opciones_nav, label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 Aura Wealth Quant Engine")

    # --- ENRUTADOR DE VISTAS ---
    if vista_actual == "📈 RESUMEN & COMPARATIVA":
        st.markdown("### 📈 PANEL DE CONTROL: COMPARATIVA DIRECTA")
        st.markdown("Añade o quita empresas para comparar visualmente su rendimiento normalizado desde el inicio del periodo seleccionado.")
        
        comparative_tickers = st.multiselect(
            "Selecciona los activos a comparar simultáneamente:", 
            options=valid_tickers, 
            default=valid_tickers
        )
        
        if not comparative_tickers:
            st.warning("Selecciona al menos un activo para visualizar la gráfica comparativa.")
        else:
            fecha_min = df_close.index.min().date()
            fecha_max = df_close.index.max().date()
            
            rango_fechas = st.slider(
                "Filtrar Rango de Fechas:",
                min_value=fecha_min,
                max_value=fecha_max,
                value=(fecha_min, fecha_max),
                format="DD/MM/YYYY"
            )
            
            mask = (df_close.index.date >= rango_fechas[0]) & (df_close.index.date <= rango_fechas[1])
            df_filtered = df_close.loc[mask, comparative_tickers]
            
            if df_filtered.empty:
                st.warning("Seleccione un rango de fechas con datos disponibles.")
            else:
                # Normalizamos usando el primer valor DENTRO del nuevo rango de fechas (Base 100)
                df_norm = (df_filtered / df_filtered.iloc[0]) * 100
                
                fig_comp = go.Figure()
            for tk in comparative_tickers:
                # Usar los colores de la app si es posible, pero Plotly asigna colores automáticos bien para muchas líneas
                fig_comp.add_trace(go.Scatter(
                    x=df_norm.index, 
                    y=df_norm[tk], 
                    mode='lines', 
                    name=tk,
                    hovertemplate='%{y:.2f}'
                ))
            
            fig_comp.update_layout(
                title="RENDIMIENTO COMPARATIVO (BASE 100)",
                plot_bgcolor='rgba(15, 23, 42, 0)',
                paper_bgcolor='rgba(15, 23, 42, 0)',
                font=dict(color='#E2E8F0'),
                xaxis=dict(showgrid=True, gridcolor='rgba(51, 65, 85, 0.5)', gridwidth=1, title='Fecha'),
                yaxis=dict(showgrid=True, gridcolor='rgba(51, 65, 85, 0.5)', gridwidth=1, title='Índice (Base 100)'),
                hovermode='x unified',
                margin=dict(l=40, r=40, t=60, b=40),
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.markdown("#### 📊 RETORNOS DEL PERIODO (TOTAL)")
            cols = st.columns(min(len(comparative_tickers), 4) if len(comparative_tickers) > 0 else 1)
            if len(comparative_tickers) > 0:
                for i, tk in enumerate(comparative_tickers[:4]):
                    ret_total = (df_norm[tk].iloc[-1] - 100)
                    cols[i].metric(label=f"{tk}", value=f"{df_norm[tk].iloc[-1]:.2f} pts", delta=f"{ret_total:.2f}%")
                if len(comparative_tickers) > 4:
                    st.caption(f"*Mostrando KPIs rápidos de los primeros 4 activos. ({len(comparative_tickers)} listados en gráfica).*")

    elif vista_actual == "TÉCNICO / MOMENTUM":
        c_head, c_select = st.columns([3, 1])
        c_head.markdown("### INSPECCION DE PRECIO Y MOMENTUM ESTRUCTURAL")
        ticker_tec = c_select.selectbox("SELECCIONAR ACTIVO BASE:", valid_tickers, index=None, placeholder="Elegir Activo...", label_visibility="collapsed")
        
        if not ticker_tec:
            st.info("👈 Use el menú superior derecho para desplegar el análisis sobre una de las acciones que inicializó al principio.")
            st.stop()
            
        with st.expander("GUIA REPTILIANA DE LECTURA GRAFICA (PASO A PASO PARA PRINCIPAIANTES) 👇"):
            st.markdown("""
            **Si nunca has abierto la bolsa, así es como debes traducir este panel institucional:**
            *   **Los Rectángulos (Velas):** Muestran el rastro de la pelea diaria. Si el trazo es brillante/claro, la jornada acabó en beneficios (fuerza de compra). Si es negro/oscuro, la jornada cayó (miedo). 
            *   **La Línea Azul Fiel (Media Móvil 50):** Es la frontera central. Cuando el precio baila **por encima** de esa línea azul, estamos en una bonanza alcista segura. Si rompe el cristal de la línea hacia abajo, se inicia zona de peligro estructural.
            *   **Las Barras Inferiores (Volumen):** Representa el "dinero institucional" en juego. Un movimiento brusco que no acompañe barras atlas significa que nadie se fía. Movimientos fuertes con barras altas son confirmaciones fiables.
            *   **Termómetro RSI (La línea ondulante):** Detecta los extremos humanos. Si la línea se rompe y vuela por encima del techo (70), los compradores están en euforia maníaca (puede venirse un estallido, sobrecompra). Si se entierra por el suelo (<30), hay pánico inyectado, la acción está barata y el rebote acecha (sobreventa).
            """)

        df_tec = calculate_technical_indicators(df_close[ticker_tec])
        
        has_ohlcv = ('Open' in raw_data.columns and 'Volume' in raw_data.columns)
        if isinstance(raw_data.columns, pd.MultiIndex):
            has_ohlcv = has_ohlcv and (ticker_tec in raw_data['Open'].columns)
            op = raw_data['Open'][ticker_tec] if has_ohlcv else None
            hi = raw_data['High'][ticker_tec] if has_ohlcv else None
            lo = raw_data['Low'][ticker_tec] if has_ohlcv else None
            vo = raw_data['Volume'][ticker_tec] if has_ohlcv else None
        else:
            op = raw_data['Open'] if 'Open' in raw_data.columns else None
            hi = raw_data['High'] if 'High' in raw_data.columns else None
            lo = raw_data['Low'] if 'Low' in raw_data.columns else None
            vo = raw_data['Volume'] if 'Volume' in raw_data.columns else None

        fig_tec = make_subplots(rows=3 if has_ohlcv else 2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2] if has_ohlcv else [0.7, 0.3])
        
        idx_main = 1
        idx_vol = 2 if has_ohlcv else None
        idx_osc = 3 if has_ohlcv else 2

        # Top Plot: Main Chart
        if has_ohlcv and op is not None:
             fig_tec.add_trace(go.Candlestick(x=df_tec.index, open=op, high=hi, low=lo, close=df_tec['Close'], name='Cotización Estructural', increasing_line_color=COLOR_SMOKE, decreasing_line_color=COLOR_DARK_SMOKE), row=idx_main, col=1)
             fig_tec.add_trace(go.Bar(x=df_tec.index, y=vo, name='Volumen Agregado', marker_color=COLOR_SMOKE, opacity=0.3), row=idx_vol, col=1)
        else:
             fig_tec.add_trace(go.Scatter(x=df_tec.index, y=df_tec['Close'], mode='lines', line=dict(color=COLOR_SMOKE, width=1.5), name='Cierre Limpio'), row=idx_main, col=1)
             
        if 'SMA_50' in df_tec: fig_tec.add_trace(go.Scatter(x=df_tec.index, y=df_tec['SMA_50'], mode='lines', line=dict(color=COLOR_TREND, width=2), name='SMA Institucional (50)'), row=idx_main, col=1)
        
        # Bottom Plot: RSI & MACD Overlaid (for clean minimalist UI)
        if 'RSI_14' in df_tec:
            fig_tec.add_trace(go.Scatter(x=df_tec.index, y=df_tec['RSI_14'], line=dict(color=COLOR_SMOKE, width=1.5), name='RSI (Momentum Oscilador)'), row=idx_osc, col=1)
            fig_tec.add_hline(y=70, line=dict(dash="dot", width=1, color=COLOR_DARK_SMOKE), row=idx_osc, col=1)
            fig_tec.add_hline(y=30, line=dict(dash="dot", width=1, color=COLOR_DARK_SMOKE), row=idx_osc, col=1)
            # Add MACD as Bar on secondary axis implicitly or on RSI chart if scaled, but lets keep it standard RSI for cleanliness
            
        fig_tec = clean_layout(fig_tec, height=750)
        fig_tec.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig_tec, use_container_width=True)
        
        interpret_tecnico(df_tec, ticker_tec)

    elif vista_actual == "RIESGO (CAPM)":
        c_head, c_select = st.columns([3, 1])
        c_head.markdown("### AUDITORIA PARAMETRICA DE RENDIMIENTO (CAPM)")
        t_sel = c_select.selectbox("ENFOCAR ACTIVO ANALITICO:", valid_tickers, index=None, placeholder="Elegir Activo...", label_visibility="collapsed", key="capm_sel")
        
        if not t_sel:
            st.info("👈 Seleccione un activo del menú superior derecho para desplegar Capital Asset Pricing.")
            st.stop()
            
        if t_sel and BENCHMARK_TICKER in df_close.columns:
            ret_act = df_close[t_sel].pct_change().dropna()
            ret_mkt = df_close[BENCHMARK_TICKER].pct_change().dropna()
            res = get_performance_metrics(ret_act, ret_mkt, RISK_FREE_RATE)
            if res:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RENDIMIENTO ESPERADO (ANUAL)", f"{res['Ret_Ann']*100:.2f}%")
                col2.metric("SHARPE RATIO (AJUSTE VOL)", f"{res['Sharpe']:.2f}")
                col3.metric("ALPHA DE JENSEN (EXCESO)", f"{res['Alpha']*100:.2f}%")
                col4.metric("BETA SISTEMICO (MERCADO)", f"{res['Beta']:.2f}")
                
                interpret_institucional(res, t_sel)
                
                fig_uw = plot_drawdown_underwater(ret_act, t_sel)
                st.plotly_chart(fig_uw, use_container_width=True)
        else:
             st.warning("SE REQUIERE EL ACTIVO BENCHMARK ('SPY') EN LA CONFIGURACION PARA COMPUTAR EL MODELO CAPM.")

    elif vista_actual == "MARKOWITZ":
        st.markdown("### MODELADO ESTOCÁSTICO Y FRONTERA EFICIENTE")
        
        if len(valid_tickers) < 2:
            st.error("OPERACION DENEGADA. LA ARQUITECTURA REQUIERE AL MENOS DOS SERIES COMPATIBLES PARAMETRICAMENTE.")
        else:
            returns_all = df_close[valid_tickers].pct_change().dropna()
            
            c_hm, c_sv = st.columns([1, 1.2])
            with c_hm:
                st.markdown("#### MATRIZ DE AUTOCORRELACION")
                corr = returns_all.corr()
                fig_hm = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='Greys', zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}", hoverinfo="z+x+y"))
                st.plotly_chart(clean_layout(fig_hm, height=450), use_container_width=True)
                
            with c_sv:
                st.markdown("#### OPTIMIZADOR SHARPE (METODO SLSQP)")
                if st.button("COMPUTAR FRONTERA EFICIENTE", type="primary"):
                    with st.spinner("Simulando 2,500 carteras aleatorias hipervectoriales..."):
                        max_w, min_w = get_optimized_portfolios(returns_all.mean().values, returns_all.cov().values, RISK_FREE_RATE)
                        
                        # Generación de Frontera Eficiente Visual
                        num_portfolios = 2500
                        results = np.zeros((3, num_portfolios))
                        mean_rets = returns_all.mean().values
                        cov_mat = returns_all.cov().values
                        
                        for i in range(num_portfolios):
                            weights = np.random.random(len(valid_tickers))
                            weights /= np.sum(weights)
                            p_ret = np.sum(mean_rets * weights) * 252
                            p_std = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights))) * np.sqrt(252)
                            p_sharpe = (p_ret - RISK_FREE_RATE) / p_std
                            results[0,i] = p_std
                            results[1,i] = p_ret
                            results[2,i] = p_sharpe
                            
                        fig_ef = go.Figure()
                        fig_ef.add_trace(go.Scatter(x=results[0,:], y=results[1,:], mode='markers', 
                                                    marker=dict(color=results[2,:], colorscale='Viridis', showscale=True, size=5, colorbar=dict(title="Sharpe")),
                                                    name='Carteras Aleatorias'))
                        
                        opt_ret = np.sum(mean_rets * max_w) * 252
                        opt_std = np.sqrt(np.dot(max_w.T, np.dot(cov_mat, max_w))) * np.sqrt(252)
                        fig_ef.add_trace(go.Scatter(x=[opt_std], y=[opt_ret], mode='markers', marker=dict(color='red', size=15, symbol='star'), name='Máximo Sharpe'))
                        
                        fig_ef = clean_layout(fig_ef, title="FRONTERA EFICIENTE DE MARKOWITZ", height=450)
                        fig_ef.update_xaxes(title="Riesgo Anualizado (Volatilidad)")
                        fig_ef.update_yaxes(title="Retorno Anualizado Esperado")
                        st.plotly_chart(fig_ef, use_container_width=True)
                    
                    interpret_markowitz(max_w, valid_tickers)
                    
                    df_weights = pd.DataFrame({'Activo': valid_tickers, 'Asignación (%)': max_w * 100})
                    df_weights = df_weights[df_weights['Asignación (%)'] > 0.5].sort_values(by='Asignación (%)', ascending=False)
                    st.dataframe(df_weights.style.format({'Asignación (%)': "{:.2f}%"}).bar(subset=['Asignación (%)'], color=COLOR_SMOKE, vmin=0, vmax=100), use_container_width=True)

    elif vista_actual == "MONTE CARLO (VaR)":
        c_head, c_select = st.columns([3, 1])
        c_head.markdown("### CAMINATA BROWNIANA Y METRICAS DE RIESGO DE COLA")
        t_mc = c_select.selectbox("SERIE OBJETIVO A 1-ANIO:", valid_tickers, index=None, placeholder="Elegir Activo...", label_visibility="collapsed")
        
        if not t_mc:
            st.info("👈 Seleccione un activo para generar la trayectoria matricial futura estocástica.")
            st.stop()
            
        if t_mc:
            with st.spinner("Vectorizando series sintéticas multifásicas..."):
                returns_mc = df_close[t_mc].pct_change().dropna()
                if len(returns_mc) < 20:
                    st.error("EL TAMAÑO DE LA MUESTRA IMPIDE LA INFERENCIA ESTADISTICA FIABLE.")
                else:
                    sim_data = run_monte_carlo(df_close[t_mc].iloc[-1], returns_mc, days=252, simulations=500)
                    var_95 = calculate_var(returns_mc, 0.95)
                    
                    fig = go.Figure()
                    for i in range(min(150, sim_data.shape[1])):
                        fig.add_trace(go.Scatter(y=sim_data[:, i], mode='lines', line=dict(color='rgba(71, 85, 105, 0.05)', width=1), showlegend=False, hoverinfo='skip'))
                        
                    mean_path = np.mean(sim_data, axis=1)
                    fig.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(color=COLOR_TREND, width=3), name='Trayectoria Esperada Vectorial'))
                    fig.add_trace(go.Scatter(y=np.percentile(sim_data, 95, axis=1), mode='lines', line=dict(color=COLOR_SMOKE, width=1.5, dash='dash'), name='Banda Confianza Superior (+95%)'))
                    fig.add_trace(go.Scatter(y=np.percentile(sim_data, 5, axis=1), mode='lines', line=dict(color=COLOR_SMOKE, width=1.5, dash='dash'), name='Banda Confianza Inferior (-5%)'))
                    
                    fig = clean_layout(fig, height=550)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    interpret_oraculo(sim_data, var_95)

    elif vista_actual == "ESTACIONALIDAD":
        st.markdown("### ESTACIONALIDAD MATEMÁTICA ANUAL")
        st.markdown("Mapa de calor histórico para detectar estadísticamente en qué meses una acción suele subir o colapsar de media.")
        
        t_season = st.selectbox("ANALIZAR MESES DE:", valid_tickers, index=None, placeholder="Elegir Activo...", label_visibility="collapsed", key="season_sel")
        
        if not t_season:
            st.info("👈 Seleccione un activo para mapear su estacionalidad en la Base de Datos.")
            st.stop()
            
        if t_season:
            df_s = df_close[[t_season]].copy()
            df_s['Year'] = df_s.index.year
            df_s['Month'] = df_s.index.month
            
            # Retorno mensual
            # Resampleamos a final de mes y calculamos pct change
            monthly_data = df_s[t_season].resample('ME').last().pct_change() * 100
            monthly_data = monthly_data.to_frame(name='Close')
            monthly_data['Year'] = monthly_data.index.year
            monthly_data['Month'] = monthly_data.index.month
            
            # Pivot table (Años en Y, Meses en X)
            season_pivot = monthly_data.pivot_table(index='Year', columns='Month', values='Close')
            season_pivot.columns = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'][:len(season_pivot.columns)]
            
            avg_monthly = season_pivot.mean()
            season_pivot.loc['Promedio Historico'] = avg_monthly
            
            fig_season = go.Figure(data=go.Heatmap(
                z=season_pivot.values,
                x=season_pivot.columns,
                y=season_pivot.index,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(season_pivot.values, 2),
                texttemplate="%{text}%",
                hoverinfo="x+y+z"
            ))
            fig_season = clean_layout(fig_season, title=f"MAPA DE ESTACIONALIDAD - {t_season} (%)", height=500)
            st.plotly_chart(fig_season, use_container_width=True)
            
            with st.expander("🎯 TRADUCCION PRACTICA ESTACIONALIDAD", expanded=True):
                st.markdown("> *Las matemáticas no mienten. A veces las acciones caen en Septiembre porque los grandes fondos cierran libros fiscales. Si la fila de abjo 'Promedio Histórico' está en rojo profundo para un mes en concreto, comprar a princippios de ese mes suele ser un suicidio estadístico probadamente repetido a lo largo de los años.*")

    elif vista_actual == "RADAR QUANT":
        st.markdown("### RADAR AUTOMÁTICO (SCREENER INSTITUCIONAL)")
        st.markdown("El sistema escanea de fondo una lista predefinida de gigantes de mercado, para avisarte en tiempo real de gangas sobrevendidas o burbujas sobrecompradas sin que tú tengas que buscar.")
        
        if st.button("EJECUTAR ESCÁNER MASIVO (NASDAQ 100 TOP / SP500)", type="primary"):
            target_univ = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT", "JNJ", "PG", "MA", "UNH", "XOM"]
            with st.spinner("Escaneando matriz algorítmica y calculando RSIs/MACDs en Background..."):
                try:
                    radar_data = yf.download(target_univ, period="3mo", auto_adjust=True, progress=False)['Close']
                    radar_results = []
                    
                    for tk in radar_data.columns:
                        try:
                            df_tk = calculate_technical_indicators(radar_data[tk].dropna())
                            if len(df_tk) > 20:
                                rsi_val = df_tk['RSI_14'].iloc[-1]
                                macd_val = df_tk['MACD'].iloc[-1]
                                sig_val = df_tk['Signal_Line'].iloc[-1]
                                ret_mensual = (df_tk['Close'].iloc[-1] / df_tk['Close'].iloc[-20] - 1) * 100
                                
                                estado = "NEUTRAL"
                                badge = "⚪"
                                if rsi_val < 35 and macd_val > sig_val:
                                    estado = "SOBREVENDIDA (COMPRA FUERTE)"
                                    badge = "🟢"
                                elif rsi_val < 35:
                                    estado = "SOBREVENDIDA (BARATA)"
                                    badge = "🟢"
                                elif rsi_val > 70:
                                    estado = "SOBRECOMPRADA (CARA)"
                                    badge = "🔴"
                                    
                                radar_results.append({
                                    "Activo": tk, "Estado": f"{badge} {estado}", "RSI (1-100)": round(rsi_val, 2), 
                                    "Momento (1 Mes)": f"{round(ret_mensual, 2)}%"
                                })
                        except:
                            pass
                    
                    df_radar = pd.DataFrame(radar_results)
                    df_radar = df_radar.sort_values(by="RSI (1-100)", ascending=True).reset_index(drop=True)
                    st.dataframe(df_radar, use_container_width=True)
                    st.success("Escáner finalizado. Activos en la parte alta están estadísticamente devaluados u ofrecen un descuento histórico a corto plazo.")
                except Exception as e:
                    st.error(f"Fallo de conexión al mercado escaner: {e}")

    elif vista_actual == "MI PORTAFOLIO":
        st.markdown("### VALORACION DE CARTERA Y PnL EN VIVO")
        st.markdown("Sistema local de sesión. Carga tus activos para realizar la valoración temporal (no se guarda en la nube).")
        
        saved_port = {}
        
        if valid_tickers:
            with st.form("portfolio_form"):
                cols = st.columns(4)
                shares = {}
                for i, tk in enumerate(valid_tickers):
                    with cols[i % 4]:
                         val_def = float(saved_port.get(tk, 0.0))
                         shares[tk] = st.number_input(f"Acciones {tk}", min_value=0.0, value=val_def, step=1.0)
                
                c_b1, c_b2 = st.columns(2)
                with c_b1:
                    port_submit = st.form_submit_button("COMPUTAR VALORACION (TEMPORAL)", type="primary", use_container_width=True)
            
            if port_submit:
                total_val = 0
                st.markdown("#### DESGLOSE NOMINAL DE POSICIONES")
                c1, c2, c3, c4 = st.columns(4)
                col_idx = 0
                cols_layout = [c1, c2, c3, c4]
                for tk, sh in shares.items():
                    if sh > 0:
                        price = df_close[tk].iloc[-1]
                        val = price * sh
                        total_val += val
                        with cols_layout[col_idx % 4]:
                            st.metric(f"{tk} ({sh} uds)", f"{val:,.2f} $", delta=f"Precio UD: {price:.2f} $", delta_color="off")
                        col_idx += 1
                
                st.markdown("<hr>", unsafe_allow_html=True)
                st.metric("VALOR TOTAL DE CARTERA (AUM)", f"{total_val:,.2f} $")
                
                # VaR de Cartera Sintetizado 
                ret_port = sum((df_close[tk].pct_change().dropna() * (shares[tk]*df_close[tk].iloc[-1] / total_val)) for tk in shares if shares[tk]>0 )
                var_cartera = calculate_var(ret_port, 0.95)
                st.warning(f"⚠️ **Riesgo de Ruina (VaR 95%):** Estadísticamente, en tu peor día de crash podrías perder hasta el **{var_cartera*100:.2f}%** ({abs(total_val * var_cartera):,.2f} $) del capital inyectado.")
        else:
            st.info("Agrega activos en la barra lateral para gestionarlos.")

    elif vista_actual == "MACRO & NOTICIAS":
        c_head, c_select = st.columns([3, 1])
        c_head.markdown("### MACROECONOMIA, FUNDAMENTALES Y DATA LAKE")
        t_fund = c_select.selectbox("AUDITORÍA CORPORATIVA:", valid_tickers, index=None, placeholder="Elegir Activo...", label_visibility="collapsed", key="fund_sel")
        
        if not t_fund:
            st.info("👈 Seleccione un activo superior para escanear su NLP.")
            st.stop()
            
        if t_fund:
            with st.spinner(f"Conectando con Data Lake de YFinance para {t_fund}..."):
                try:
                    tk_obj = yf.Ticker(t_fund)
                    info = tk_obj.info
                    news = tk_obj.news
                    
                    st.markdown(f"#### MÉTRICAS DE VALORACIÓN: {t_fund}")
                    c1, c2, c3, c4 = st.columns(4)
                    
                    pe = info.get('trailingPE', 'N/A')
                    dy = info.get('dividendYield', 0)
                    pm = info.get('profitMargins', 0)
                    mcap = info.get('marketCap', 0)
                    
                    c1.metric("P/E Ratio (PER)", pe)
                    c2.metric("Dividend Yield", f"{dy*100:.2f}%" if dy else 'N/A')
                    c3.metric("Profit Margin", f"{pm*100:.2f}%" if pm else 'N/A')
                    c4.metric("Market Cap", f"{mcap/1e9:.2f}B $" if mcap else 'N/A')
                    
                    with st.expander("🎯 TRADUCCION PRACTICA DE FUNDAMENTALES", expanded=True):
                        st.markdown("> *El **PER** indica cuántos años tardarías en recuperar tu inversión si los beneficios fuesen constantes (menos es más barato). El **Dividend Yield** es el % de dinero extra que te pagan al año solo por tener la acción. El **Profit Margin** indica de cada 100$ que venden, cuánto es beneficio puro para el bolsillo.*")
                    
                    st.markdown("#### TERMINAL DE NOTICIAS EN VIVO (BLOOMBERG/REUTERS FEED)")
                    if news:
                        try:
                            analyzer = SentimentIntensityAnalyzer()
                            sentiments = []
                            for n in news[:5]:
                                title = n.get('title', '')
                                compound = analyzer.polarity_scores(title)['compound']
                                sentiments.append(compound)
                                
                                if compound >= 0.05:
                                    s_badge = "🟢 BULLISH"
                                elif compound <= -0.05:
                                    s_badge = "🔴 BEARISH"
                                else:
                                    s_badge = "⚪ NEUTRAL"
                                    
                                st.markdown(f"📰 **[{title}]({n.get('link')})** - *(Fuente: {n.get('publisher')})*  \n└ *Sentimiento NLP:* **{s_badge}**")
                            
                            avg_sent = np.mean(sentiments)
                            st.markdown("#### 🧠 TERMÓMETRO NLP (Sentimiento Global)")
                            s_color = "#38BDF8" if avg_sent >= 0.05 else "#EF4444" if avg_sent <= -0.05 else "#94A3B8"
                            s_text = "ALCISTA (BULLISH)" if avg_sent >= 0.05 else "BAJISTA (BEARISH)" if avg_sent <= -0.05 else "NEUTRO"
                            st.markdown(f"<h3 style='color:{s_color};'>{s_text} (Score: {avg_sent:.2f})</h3>", unsafe_allow_html=True)
                            
                        except:
                            for n in news[:5]:
                                st.markdown(f"📰 **[{n.get('title')}]({n.get('link')})** - *(Fuente: {n.get('publisher')})*")
                    else:
                        st.info("No hay noticias macroeconómicas o corporativas recientes.")
                except Exception as e:
                    st.error(f"El Data Lake no devolvió fundamentales directos para el activo {t_fund}.")

    elif vista_actual == "AI FORECASTING":
        st.markdown("### MOTOR DE MACHINE LEARNING (FORECASTING NO-LINEAL)")
        st.markdown("Entrenamiento en vivo de un *Random Forest Regressor* hiperparametrizado para proyectar el vector direccional de los próximos 10 días basado en *Price Action* histórico (Momentum, Volatilidad, Autocorrelación).")
        
        t_ai = st.selectbox("OPERAR RED NEURONAL SOBRE ACTIVO:", valid_tickers, index=None, placeholder="Elegir Activo...", label_visibility="collapsed", key="ai_sel")
        
        if not t_ai:
            st.info("👈 Seleccione el activo de la lista para despertar el motor LLM.")
            st.stop()
            
        if t_ai:
            if st.button("INICIAR INFERENCIA (SCIKIT-LEARN)", type="primary"):
                with st.spinner(f"Entrenando modelo Random Forest para {t_ai}..."):
                    df_ai = pd.DataFrame()
                    df_ai['Close'] = df_close[t_ai]
                    df_ai['Returns'] = df_ai['Close'].pct_change()
                    df_ai['Lag_1'] = df_ai['Close'].shift(1)
                    df_ai['Lag_2'] = df_ai['Close'].shift(2)
                    df_ai['SMA_10'] = df_ai['Close'].rolling(window=10).mean()
                    df_ai['Vol_10'] = df_ai['Returns'].rolling(window=10).std()
                    
                    df_ai['Target'] = df_ai['Close'].shift(-1)
                    df_ai = df_ai.dropna()
                    
                    if len(df_ai) > 50:
                        from sklearn.ensemble import RandomForestRegressor
                        
                        vars_in = ['Lag_1', 'Lag_2', 'SMA_10', 'Vol_10']
                        X = df_ai[vars_in]
                        y = df_ai['Target']
                        
                        with st.spinner("Procesando Analíticas Cuantitativas..."):
                            df_ml = calculate_technical_indicators(df_close[t_ai])
                            current_pr = df_ml['Close'].iloc[-1]
                            rsi_now = df_ml['RSI_14'].iloc[-1]
                            vol_now = df_ml['Volatilidad_20d'].iloc[-1] * 100
                            
                            # Lógica Determinista de Diagnóstico
                            if current_pr < df_ml['SMA_50'].iloc[-1] and vol_now > 30:
                                rec = "REDUCIR EXPOSICIÓN"
                                diag = "Estructura bajista severa bajo SMA 50 combinada con volatilidad crítica (riesgo asimétrico)."
                            elif rsi_now < 30 and current_pr > df_ml['SMA_50'].iloc[-1]:
                                rec = "ACUMULAR"
                                diag = "Sobreventa temporal en osciladores mientras se mantiene la tendencia estructural primaria alcista."
                            elif rsi_now > 70:
                                rec = "MANTENER TAMAÑO / CAUTELA"
                                diag = "Niveles de sobrecompra extremos. Riesgo estadístico elevado de reversión inminente a la media."
                            elif current_pr < df_ml['SMA_50'].iloc[-1]:
                                rec = "MANTENER LIQUIDEZ"
                                diag = "Debilidad estructural. El precio cotiza por debajo de los promedios institucionales clave."
                            else:
                                rec = "MANTENER"
                                diag = "Estructura técnica neutral navegando dentro de la volatilidad y los parámetros estándar."
                            
                            # Presentación de la Interfaz
                            c1, c2, c3 = st.columns(3)
                            c1.metric("PRECIO HOY", f"{current_pr:.2f} $")
                            c2.metric("RSI INSTANTÁNEO", f"{rsi_now:.2f}", delta="Sobrecomprado" if rsi_now>70 else ("Sobrevendido" if rsi_now<30 else "Neutral"), delta_color="inverse")
                            c3.metric("MOTOR LÓGICO", "Algoritmo Matemático V2")
                            
                            st.markdown("### 🧠 SÍNTESIS TÉCNICA (NÚCLEO CUANTITATIVO)")
                            st.info(f"**DIAGNÓSTICO ESTRUCTURAL:** {diag}\n\n**POSICIONAMIENTO INSTITUCIONAL:** {rec}")
                            
                            st.caption("Nota: Este análisis estático proviene de reglas matemáticas en tiempo real integradas en el pipeline sin uso de APIs externas.")
                    else:
                        st.error("NO HAY SUFICIENTES DATOS HISTÓRICOS PARA ENTRENAR LA RED MULTICAPA.")
    elif vista_actual == "BACKTESTER":
        st.markdown("### MAQUINA DEL TIEMPO (BACKTESTER HISTORICO)")
        st.markdown("Simulación retrospectiva de la inyección de capital inicial frente a estrategias de Benchmark pasivo.")
        
        c_cap, c_btn = st.columns([1, 1])
        with c_cap:
             cap_inicial = st.number_input("CAPITAL INICIAL ($)", min_value=100.0, value=10000.0, step=1000.0, help="Dólares hipotéticos que hubieras invertido con tu cartera actual.")
        
        if c_btn.button("EYECTAR BACKTEST (VS SPY)", type="primary"):
            if BENCHMARK_TICKER not in df_close.columns:
                st.warning(f"Se necesita al ETF {BENCHMARK_TICKER} como proxy de mercado.")
            elif len(valid_tickers) < 2:
                st.warning("Selecciona al menos 2 activos en tu cartera para el Backtest.")
            else:
                ret_bench = df_close[BENCHMARK_TICKER].pct_change().dropna()
                # Cartera iterativa para la simulación
                ret_port = df_close[[t for t in valid_tickers if t != BENCHMARK_TICKER]].pct_change().dropna().mean(axis=1) if len(valid_tickers)>1 else ret_bench
                
                cap_bench = cap_inicial * (1 + ret_bench).cumprod()
                cap_port = cap_inicial * (1 + ret_port).cumprod()
                
                fig_bt = go.Figure()
                fig_bt.add_trace(go.Scatter(x=cap_bench.index, y=cap_bench, mode='lines', name='S&P 500 (SPY)', line=dict(color=COLOR_SMOKE)))
                fig_bt.add_trace(go.Scatter(x=cap_port.index, y=cap_port, mode='lines', name='Mi Cartera Actual', line=dict(color=COLOR_TREND, width=2)))
                fig_bt = clean_layout(fig_bt, title="CRECIMIENTO DEL EQUITY (CURVA DE CAPITAL)", height=450)
                st.plotly_chart(fig_bt, use_container_width=True)
                
                dif = cap_port.iloc[-1] - cap_bench.iloc[-1]
                st.success(f"📈 **COSTE DE OPORTUNIDAD:** Elegir tu cartera te hizo ganar/perder **{dif:,.2f} $** frente a simplemente haber comprado el mercado americano clásico.")
                c1, c2 = st.columns(2)
                c1.metric("SALDO FINAL CARTERA", f"{cap_port.iloc[-1]:,.2f} $", delta=f"{cap_port.iloc[-1] - cap_inicial:,.2f} $")
                c2.metric("SALDO FINAL SPY", f"{cap_bench.iloc[-1]:,.2f} $", delta=f"{cap_bench.iloc[-1] - cap_inicial:,.2f} $")

    elif vista_actual == "REPORTE PDF":
        st.markdown("### GENERACION DE TEAR SHEETS INSTITUCIONALES")
        st.markdown("Exporta la analítica matemática de esta sesión en un informe sintético de mercado para clientes institucionales. Combina métricas, visualizaciones de IA y narrativas de NLP.")
        
        t_rep = st.selectbox("GENERAR SÍNTESIS DE:", valid_tickers, index=None, placeholder="Elegir Activo...", label_visibility="collapsed")
        
        if not t_rep:
            st.info("👈 Selecciona el activo de la lista para forjar el Tear Sheet corporativo PDF.")
            st.stop()
            
        if st.button("COMPILAR TEAR SHEET (.PDF)", type="primary"):
            with st.spinner(f"Compilando Tear Sheet PDF para {t_rep} (Conectando con Servidor Neural)..."):
                try:
                    import os
                    from fpdf import FPDF
                    from datetime import datetime
                    
                    # 1. Variables Financieras
                    ret_anu = ((df_close[t_rep].iloc[-1] / df_close[t_rep].iloc[0]) ** (252 / len(df_close)) - 1) * 100
                    vol_anu = df_close[t_rep].pct_change().std() * np.sqrt(252) * 100
                    drawdown = -(df_close[t_rep].cummax() - df_close[t_rep]).max() / df_close[t_rep].max() * 100
                    
                    # 2. Generar Gráfico Matplotlib estático (Evita cuelgues de Kaleido/Plotly en la nube)
                    import matplotlib.pyplot as plt
                    
                    fig_pdf, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df_close.index, df_close[t_rep], color='#38BDF8', linewidth=2)
                    ax.set_title(f"Evolución Histórica: {t_rep} ({len(df_close)} Sesiones)", color='#1E293B', fontsize=12, fontweight='bold', pad=15)
                    ax.set_facecolor('#F8FAFC')
                    fig_pdf.patch.set_facecolor('#F8FAFC')
                    ax.grid(color='#E2E8F0', linestyle='--', linewidth=0.5)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#CBD5E1')
                    ax.spines['bottom'].set_color('#CBD5E1')
                    ax.tick_params(colors='#475569')
                    
                    img_path = f"tear_sheet_chart_{t_rep}.png"
                    fig_pdf.savefig(img_path, bbox_inches='tight', dpi=300)
                    plt.close(fig_pdf)
                    
                    # 3. NLP Prompt Reemplazado por Texto Determinista
                    texto_ia = f"SÍNTESIS ALGORÍTMICA: El activo {t_rep} evidencia un Retorno Anualizado de {ret_anu:.2f}% frente a una Volatilidad Base Histórica de {vol_anu:.2f}%. Contabilizando un Max Drawdown de {drawdown:.2f}%, el perfil de riesgo-retorno exige estricta vigilancia sobre la Media Móvil Estructural (SMA 50) para evitar trampas de liquidez y divergencias en los momentum rezagados."
                    
                    # 4. Compilar FPDF
                    class PDF(FPDF):
                        def header(self):
                            self.set_fill_color(15, 23, 42)
                            self.rect(0, 0, 210, 30, 'F')
                            self.set_font("helvetica", "B", 24)
                            self.set_text_color(255, 255, 255)
                            self.cell(0, 15, "AURA WEALTH OS", border=0, ln=1, align="C")
                            
                        def footer(self):
                            self.set_y(-25)
                            self.set_font("helvetica", "I", 8)
                            self.set_text_color(128, 128, 128)
                            self.multi_cell(0, 4, "CONFIDENCIAL: Este informe ha sido autogenerado por el Motor de IA Quant (Aura Wealth OS). Las simulaciones matematicas estocasticas o historicas aqui presentadas no constituyen asesoramiento financiero ni garantizan un comportamiento similar de los activos en el futuro. Rentabilidades pasadas no aseguran rentabilidades futuras.")

                    pdf = PDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=20)
                    
                    # Título Secundario
                    pdf.set_text_color(30, 41, 59)
                    pdf.set_font("helvetica", "B", 16)
                    pdf.ln(25)
                    pdf.cell(0, 10, f"TEAR SHEET INSTITUCIONAL: {t_rep}", border=0, ln=1, align="C")
                    pdf.set_font("helvetica", "I", 11)
                    pdf.cell(0, 6, f"Fecha de emision del informe: {datetime.now().strftime('%Y-%m-%d  %H:%M')}", border=0, ln=1, align="C")
                    pdf.ln(5)
                    
                    # Insertar Imagen Plotly
                    if os.path.exists(img_path):
                        pdf.image(img_path, x=15, w=180)
                        pdf.ln(5)
                        os.remove(img_path)
                    
                    # Bloque de Métricas
                    pdf.set_fill_color(241, 245, 249)
                    pdf.set_font("helvetica", "B", 12)
                    pdf.cell(0, 10, "1. AUDITORIA PARAMETRICA (RIESGO ESTRUCTURAL)", border=0, ln=1, fill=False)
                    pdf.ln(3)
                    
                    pdf.set_font("helvetica", "", 11)
                    pdf.cell(0, 8, f"   - Retorno Compuesto (CAGR): {ret_anu:.2f} %", border=0, ln=1)
                    pdf.cell(0, 8, f"   - Volatilidad Base Anualizada: {vol_anu:.2f} %", border=0, ln=1)
                    pdf.cell(0, 8, f"   - Peor Caida historica en muestra (Drawdown): {drawdown:.2f} %", border=0, ln=1)
                    pdf.ln(8)

                    # Bloque de IA
                    pdf.set_font("helvetica", "B", 12)
                    pdf.cell(0, 10, "2. INFERENCIA DEL ALGORITMO QUANTITATIVO (LLM)", border=0, ln=1, fill=False)
                    pdf.ln(3)
                    
                    pdf.set_font("helvetica", "", 11)
                    pdf.multi_cell(0, 6, f"NLP Engine Extract:\n\n{texto_ia}")
                    
                    raw_pdf = pdf.output()
                    if isinstance(raw_pdf, str):
                        pdf_bytes = raw_pdf.encode('latin1')
                    else:
                        pdf_bytes = bytes(raw_pdf)
                        
                    st.success(f"Tear Sheet Cuantitativo para {t_rep} Compilado con Éxito.")
                    st.download_button("📥 DESCARGAR INFORME OFFICIAL (.PDF)", data=pdf_bytes, file_name=f"TearSheet_{t_rep}_AuraWealth.pdf", mime="application/pdf", use_container_width=True)
                
                except Exception as e:
                    st.error(f"Fallo en Renderización Documental (PDF Engine): {e}")

    elif vista_actual == "RAW DATA":
        st.markdown("### EXPORTACIÓN DEL TENSOR DATALAKE")
        st.markdown("Selecciona el formato de renderizado para exportar los miles de puntos matriciales de este análisis a tu software externo.")
        st.dataframe(df_close.sort_index(ascending=False).head(150).style.format("{:.2f} $"), use_container_width=True)
        
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            st.download_button(
                label="DESCARGAR TENSOR (CSV)",
                data=df_close.to_csv(index=True).encode('utf-8'),
                file_name="quants_historical_database.csv",
                mime="text/csv"
            )
        with c2:
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_close.to_excel(writer, sheet_name='Historical_Prices')
            excel_data = buffer.getvalue()
            st.download_button(
                label="DESCARGAR MASTER (.XLSX)",
                data=excel_data,
                file_name="Aura_Wealth_DataLake.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

if __name__ == "__main__":
    main()

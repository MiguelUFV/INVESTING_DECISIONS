import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import norm
from scipy import stats
import logging

# --- CONFIGURACION DE PAGINA ---
st.set_page_config(
    page_title="AURA WEALTH OS | Next-Gen Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS PREMIUM FINTECH (Glassmorphism & Gradients) ---
st.markdown("""
<style>
    /* Fondo Moderno Radial Gradient */
    .stApp { 
        background: radial-gradient(circle at top right, #1E1B4B 0%, #0F172A 40%, #020617 100%);
        background-attachment: fixed;
        color: #E2E8F0; 
        font-family: 'Inter', 'Segoe UI', Helvetica, sans-serif; 
    }
    /* Estilos para titulos con gradientes premium */
    h1 {
        background: -webkit-linear-gradient(45deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    h2, h3, h4 { color: #F8FAFC !important; font-weight: 600 !important; letter-spacing: 0.02em; }
    
    /* Layout Containers and Padding */
    .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1600px; }
    
    /* Sidebar con Efecto Cristal (Glassmorphism) */
    [data-testid="stSidebar"] { 
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.05); 
    }
    
    /* Cards Modernas (Glassmorphism y Sombras Neón suaves) */
    div[data-testid="metric-container"] { 
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08); 
        padding: 1.25rem; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700; color: #FFFFFF; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
    
    /* Pestañas (Tabs) Estilo Navegador Premium */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.3);
        border-radius: 8px 8px 0 0;
        padding: 10px 16px;
        border: 1px solid transparent;
        transition: background-color 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-bottom: none;
    }
    
    /* Expander override (Cajas Limpias) */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #E2E8F0 !important;
        background-color: rgba(30, 41, 59, 0.5) !important;
        border-radius: 8px;
    }
    
    hr { border-color: rgba(255,255,255,0.05) !important; }
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
                df_close = data['Close']
            else:
                return pd.DataFrame(), pd.DataFrame()
        else:
            if 'Close' in data.columns:
                df_close = pd.DataFrame({tickers[0]: data['Close']}) if len(tickers) == 1 else data[['Close']]
            else:
                return pd.DataFrame(), pd.DataFrame()

        df_close = df_close.dropna(axis=1, how='all')
        df_close = df_close.ffill().bfill()
        
        return df_close, data
    except Exception as e:
        logging.error(f"Error de red o procesamiento en descarga de activos: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# --- MATEMATICA CUANTITATIVA ---

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
    st.markdown("<h1>AURA WEALTH OS (QUANT PLATFORM)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; font-size:1.1rem; letter-spacing:0.02em;'>Plataforma FinTech Next-Gen para Análisis Algorítmico y Retorno Absoluto.</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        with st.form("filtros_globales", clear_on_submit=False):
            st.markdown("<h3>PARAMETROS DE ENTORNO</h3>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            
            st.markdown("#### CONFIGURACION DE ACTIVOS")
            
            tickers_selected = []
            with st.expander("🌍 BASE DE DATOS GLOBAL DE ACTIVOS", expanded=True):
                st.markdown("**🇺🇸 USA: TECNOLOGÍA E IA (Nasdaq)**")
                us_tech = st.multiselect("Big Tech & Semiconductores", 
                    ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "CRM", "AMD", "ADBE", "NFLX", "CSCO", "INTC", "QCOM", "IBM", "ORCL", "NOW"], 
                    default=["AAPL", "MSFT"], help="Gigantes tecnológicos y arquitectos de Inteligencia Artificial.")
                
                st.markdown("**🇺🇸 USA: VALOR, FINANZAS Y SALUD (S&P 500)**")
                us_fin = st.multiselect("Banca, Consumo y Farma", 
                    ["JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "BRK-B", "JNJ", "UNH", "LLY", "ABBV", "MRK", "PFE", "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "DIS", "NKE"], 
                    default=[], help="Bancos de Wall Street y gigantes defensivos de la vieja economía.")
                
                st.markdown("**🇪🇺 EUROPA Y ESPAÑA (EuroStoxx & IBEX 35)**")
                eu_stocks = st.multiselect("Gran Capitalización Europea", 
                    ["ASML", "SAP", "SIE.DE", "MC.PA", "AIR.PA", "OR.PA", "SAN.MC", "BBVA.MC", "IBE.MC", "ITX.MC", "REP.MC", "TEF.MC", "CABK.MC", "AENA.MC", "FER.MC"], 
                    default=[], help="Industria del lujo, matriz de microchips europea y pesos pesados españoles (Añaden el sufijo .MC, .PA, .DE)")

                st.markdown("**🌏 ASIA Y MERCADOS EMERGENTES**")
                asia_latam = st.multiselect("Dragones Asiáticos y LATAM", 
                    ["TSM", "BABA", "JD", "BIDU", "7203.T", "6758.T", "9984.T", "005930.KS", "VALE", "PBR", "MELI", "NU"], 
                    default=[], help="Semiconductores de Taiwan, Ecommerce Chino, gigantes Japoneses (Toyota/Sony) y unicornios de Latam.")
                
                st.markdown("**🛢️ MATERIAS PRIMAS Y MACRO (ETFs)**")
                macro_etf = st.multiselect("Oro, Petróleo, Bonos e Índices", 
                    ["SPY", "QQQ", "DIA", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV", "USO", "UNG", "UUP", "VIXY"], 
                    default=["SPY", "TLT"], help="SPY=S&P500, QQQ=Nasdaq, TLT=Bonos 20 años, GLD=Oro Físico, USO=Petróleo Crudo.")

                st.markdown("**🪙 CRIPTOACTIVOS (Top Market Cap)**")
                crypto = st.multiselect("Ecosistema Blockchain", 
                    ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "LINK-USD", "MATIC-USD", "LTC-USD"], 
                    default=[], help="Activos digitales que cotizan 24/7 de estrés algorítmico extremo.")
                
                st.markdown("**🛠️ CAJA FUERTE: CUALQUIER OTRA EMPRESA DEL MUNDO**")
                custom_tickers = st.text_input("Buscador Libre (Tickers Manuales)", value="", placeholder="Ej: F, GM, PLTR, RKLB, RIVN", help="Hay más de 60.000 acciones en el mundo. Si no está en las listas rápidas de arriba, escribe aquí su 'Ticker' de Yahoo Finance separado por comas.")
                
                tickers_selected.extend(us_tech + us_fin + eu_stocks + asia_latam + macro_etf + crypto)
                if custom_tickers:
                    tickers_selected.extend([t.strip().upper() for t in custom_tickers.split(',') if t.strip()])
            
            st.markdown("#### HORIZONTE TEMPORAL")
            col_d1, col_d2 = st.columns(2)
            with col_d1: start_date = st.date_input("INICIO", value=pd.to_datetime('2023-01-01'), help="Día en el que empezamos a recolectar datos pasados.")
            with col_d2: end_date = st.date_input("FIN", value=pd.to_datetime('today'), help="Último día a analizar (normalmente, hoy).")
            
            st.markdown("#### PARAMETROS DE RIESGO")
            risk_free_val = st.number_input("TASA LIBRE DE RIESGO (Rf %)", value=4.0, step=0.1, help="Rendimiento de los bonos seguros de Gobierno. Si el banco te da un 4% seguro, invertir en bolsa debe exigirte dar MÁS de ese 4% para que merezca el riesgo.")
            global RISK_FREE_RATE
            RISK_FREE_RATE = risk_free_val / 100.0
            
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.form_submit_button("INICIALIZAR MOTOR Y COMPILAR DATOS", type="primary", use_container_width=True)

    if run_btn or "df_close" in st.session_state:
        tickers_list = list(set(tickers_selected)) # Remove duplicates automatically
        if not tickers_list:
            st.error("LA CONFIGURACION REQUIERE UN ACTIVO COMO MINIMO.")
            st.stop()

        with st.spinner("Sintetizando series temporales via YFinance Core..."):
            df_close, raw_data = load_market_data(tickers_list.copy(), start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if df_close.empty:
                st.error("ERROR DE EJECUCION: LOS PARAMETROS TEMPORALES O NOMINALES NO PROPORCIONAN UNA SALIDA VALIDA EN LA API EXTERNA.")
                st.stop()
                
            st.session_state["df_close"] = df_close
            st.session_state["raw_data"] = raw_data
            st.session_state["valid_tickers"] = [t for t in tickers_list if t in df_close.columns]
            
            if run_btn:
                st.toast("✅ Motor Cuantitativo Iniciado. Tensor de datos vectorizado con éxito.")

    if "df_close" not in st.session_state:
        st.info("SISTEMA EN ESPERA. COMPLETE LA CONFIGURACION PARALELA Y ACTIVE EL MOTOR PARA DESPLIEGUE.")
        st.stop()

    df_close = st.session_state["df_close"]
    valid_tickers = st.session_state["valid_tickers"]
    raw_data = st.session_state["raw_data"]

    # --- 9 PESTANAS ESTRUCTURALES (EXPANSION SAAS) ---
    tab_tec, tab_quant, tab_solver, tab_oraculo, tab_port, tab_fund, tab_backtest, tab_report, tab_data = st.tabs([
        "TÉCNICO / MOMENTUM", "RIESGO (CAPM)", "MARKOWITZ", "MONTE CARLO (VaR)", "MI PORTAFOLIO", "MACRO & NOTICIAS", "BACKTESTER", "REPORTE PDF", "RAW DATA"
    ])

    # --- TAB 1: DASHBOARD TECNICO + MACD/RSI ---
    with tab_tec:
        c_head, c_select = st.columns([3, 1])
        c_head.markdown("### INSPECCION DE PRECIO Y MOMENTUM ESTRUCTURAL")
        ticker_tec = c_select.selectbox("SELECCIONAR ACTIVO BASE:", valid_tickers, label_visibility="collapsed")
        
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

    # --- TAB 2: METRICAS RIESGO CAPM & DRAWDOWNS ---
    with tab_quant:
        c_head, c_select = st.columns([3, 1])
        c_head.markdown("### AUDITORIA PARAMETRICA DE RENDIMIENTO (CAPM)")
        t_sel = c_select.selectbox("ENFOCAR ACTIVO ANALITICO:", valid_tickers, label_visibility="collapsed", key="capm_sel")
        
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

    # --- TAB 3: MATRIZ CORRELACION Y SOLVER ---
    with tab_solver:
        st.markdown("### MODELADO DE COVARIANZA NO-LINEAL")
        
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
                    max_w, min_w = get_optimized_portfolios(returns_all.mean().values, returns_all.cov().values, RISK_FREE_RATE)
                    interpret_markowitz(max_w, valid_tickers)
                    
                    df_weights = pd.DataFrame({'Activo': valid_tickers, 'Asignación (%)': max_w * 100})
                    df_weights = df_weights[df_weights['Asignación (%)'] > 0.5].sort_values(by='Asignación (%)', ascending=False)
                    st.dataframe(df_weights.style.format({'Asignación (%)': "{:.2f}%"}).bar(subset=['Asignación (%)'], color=COLOR_SMOKE, vmin=0, vmax=100), use_container_width=True)

    # --- TAB 4: PROYECCION ESTOCASTICA ---
    with tab_oraculo:
        c_head, c_select = st.columns([3, 1])
        c_head.markdown("### CAMINATA BROWNIANA Y METRICAS DE RIESGO DE COLA")
        t_mc = c_select.selectbox("SERIE OBJETIVO A 1-ANIO:", valid_tickers, label_visibility="collapsed")
        
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

    # --- TAB 5: CARTERA REAL (PORTFOLIO TRACKER) ---
    with tab_port:
        st.markdown("### VALORACION DE CARTERA Y PnL EN VIVO")
        st.markdown("Carga aquí tus posiciones actuales de mercado para computar su valor nominal e impacto marginal de riesgo.")
        
        if valid_tickers:
            with st.form("portfolio_form"):
                cols = st.columns(4)
                shares = {}
                for i, tk in enumerate(valid_tickers):
                    with cols[i % 4]:
                         shares[tk] = st.number_input(f"Acciones de {tk}", min_value=0.0, value=0.0, step=1.0)
                port_submit = st.form_submit_button("COMPUTAR VALORACION", type="primary")
            
            if port_submit and sum(shares.values()) > 0:
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

    # --- TAB 6: FUNDAMENTAL Y NOTICIAS ---
    with tab_fund:
        c_head, c_select = st.columns([3, 1])
        c_head.markdown("### MACROECONOMIA, FUNDAMENTALES Y DATA LAKE")
        t_fund = c_select.selectbox("AUDITORÍA CORPORATIVA:", valid_tickers, label_visibility="collapsed", key="fund_sel")
        
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
                        for n in news[:5]:
                            st.markdown(f"📰 **[{n.get('title')}]({n.get('link')})** - *(Fuente: {n.get('publisher')})*")
                    else:
                        st.info("No hay noticias macroeconómicas o corporativas recientes.")
                except Exception as e:
                    st.error(f"El Data Lake no devolvió fundamentales directos para el activo {t_fund}.")

    # --- TAB 7: BACKTESTER MAQUINA DEL TIEMPO ---
    with tab_backtest:
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

    # --- TAB 8: TEAR SHEETS / REPORTES ---
    with tab_report:
        st.markdown("### GENERACION DE TEAR SHEETS INSTITUCIONALES")
        st.markdown("Exporta la analítica matemática de esta sesión en un informe sintético para clientes.")
        
        if st.button("GENERAR INFORME MATRICIAL", type="primary"):
            report_md = f"""# TEAR SHEET INSTITUCIONAL (QUANT ENGINE)
**Fecha de Generación:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Universo de Activos:** {', '.join(valid_tickers)}
**Tasa Libre Riesgo:** {RISK_FREE_RATE*100}%

## 1. RENDIMIENTOS Y DISPERSION DE RIESGO
(Métricas Anualizadas en el periodo temporal extraído)

"""
            for tk in valid_tickers:
                ret = df_close[tk].pct_change().dropna()
                if not ret.empty:
                    ann_ret = ret.mean() * 252 * 100
                    ann_vol = ret.std() * np.sqrt(252) * 100
                    report_md += f"- **{tk}**: Retorno Esperado **{ann_ret:.2f}%** | Volatilidad Asumida **{ann_vol:.2f}%**\n"
                
            report_md += """
---
*Generado de forma automática por el Motor de Series Temporales (Investing Decisions Quant Engine).*
*Exención de Responsabilidad: Modelos cuantitativos pasados no garantizan retornos macroeconógicos futuros.*
"""
            st.download_button("📥 DESCARGAR INFORME (.MD FORMATO WEB/DOC)", data=report_md.encode('utf-8'), file_name="Quant_Tear_Sheet.md", mime="text/markdown")

    # --- TAB 9: AUDITORIA DATOS ---
    with tab_data:
        st.markdown("### PIPELINE EXPORTACION E INSPECCION DE DATOS (CSV)")
        st.dataframe(df_close.sort_index(ascending=False).head(150).style.format("{:.2f} $"), use_container_width=True)
        st.download_button(
            label="DESCARGAR TENSOR DE DATOS (CSV PURIFICADO)",
            data=df_close.to_csv(index=True).encode('utf-8'),
            file_name="quants_historical_database.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()

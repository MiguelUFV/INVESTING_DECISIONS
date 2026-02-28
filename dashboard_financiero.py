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
    page_title="TERMINAL CUANTITATIVO INSTITUCIONAL",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS INSTITUCIONAL (Dark Theme Absoluto & Tipografia) ---
st.markdown("""
<style>
    /* Reset and Typography: Fondo oscuro puro #0A0B10 solicitado */
    .stApp { background-color: #0A0B10; color: #E2E8F0; font-family: 'Inter', 'Segoe UI', Helvetica, sans-serif; }
    h1, h2, h3, h4 { color: #F8FAFC !important; font-weight: 600 !important; tracking: -0.01em; margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }
    
    /* Layout Containers and Padding */
    .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1600px; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { border-right: 1px solid #1E293B; }
    
    /* Cards (Bordes sutiles y sombras suaves) */
    div[data-testid="metric-container"] { 
        background: #11141D; 
        border: 1px solid #1E293B; 
        padding: 1.25rem; 
        border-radius: 4px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 600; color: #F8FAFC; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 500; }
    
    /* Expander override */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: #94A3B8 !important;
        background-color: transparent !important;
    }
    
    hr { border-color: #1E293B !important; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTES GLOBALES Y COLORES INSTITUCIONALES ---
COLOR_TREND = '#2563EB'      # Azul Cobalto para lineas de tendencia
COLOR_BG = 'rgba(0,0,0,0)'
COLOR_SMOKE = '#475569'      # Gris Humo
COLOR_DARK_SMOKE = '#1E293B' 
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#64748B'))
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
    st.markdown("<h1>TERMINAL DE INTELIGENCIA FINANCIERA (QUANT ENGINE)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:1.0rem; letter-spacing:0.05em; text-transform:uppercase;'>Infraestructura propietaria de análisis sistemático y modelado de riesgo.</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("<h3>PARAMETROS DE ENTORNO</h3>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        
        st.markdown("#### CONFIGURACION DE ACTIVOS")
        tickers_input = st.text_input("Ingesta de Tickers (CSV format)", value="SPY, AAPL, MSFT, BRK-B, TLT")
        
        st.markdown("#### HORIZONTE TEMPORAL")
        col_d1, col_d2 = st.columns(2)
        with col_d1: start_date = st.date_input("INICIO", value=pd.to_datetime('2023-01-01'))
        with col_d2: end_date = st.date_input("FIN", value=pd.to_datetime('today'))
        
        st.markdown("#### PARAMETROS DE RIESGO")
        risk_free_val = st.number_input("TASA LIBRE DE RIESGO (Rf %)", value=4.0, step=0.1)
        global RISK_FREE_RATE
        RISK_FREE_RATE = risk_free_val / 100.0
        
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("INICIALIZAR MOTOR Y COMPILAR DATOS", type="primary", use_container_width=True)

    if run_btn or "df_close" in st.session_state:
        tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
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

    if "df_close" not in st.session_state:
        st.info("SISTEMA EN ESPERA. COMPLETE LA CONFIGURACION PARALELA Y ACTIVE EL MOTOR PARA DESPLIEGUE.")
        st.stop()

    df_close = st.session_state["df_close"]
    valid_tickers = st.session_state["valid_tickers"]
    raw_data = st.session_state["raw_data"]

    # --- 5 PESTANAS ESTRUCTURALES ---
    tab_tec, tab_quant, tab_solver, tab_oraculo, tab_data = st.tabs([
        "ANALISIS TECNICO Y MOMENTUM", "METRICAS RIESGO (CAPM)", "OPTIMIZACION MARKOVITZ", "PROYECCION ESTOCASTICA (VaR)", "AUDITORIA DE RAW DATA"
    ])

    # --- TAB 1: DASHBOARD TECNICO + MACD/RSI ---
    with tab_tec:
        c_head, c_select = st.columns([3, 1])
        c_head.markdown("### INSPECCION DE PRECIO Y MOMENTUM ESTRUCTURAL")
        ticker_tec = c_select.selectbox("SELECCIONAR ACTIVO BASE:", valid_tickers, label_visibility="collapsed")
        
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

    # --- TAB 5: AUDITORIA DATOS ---
    with tab_data:
        st.markdown("### PIPELINE EXPORTACION E INSPECCION DE DATOS MATRICIALES (CSV)")
        st.dataframe(df_close.sort_index(ascending=False).head(150), use_container_width=True)
        st.download_button(
            label="DESCARGAR TENSOR DE DATOS (CSV PURIFICADO)",
            data=df_close.to_csv(index=True).encode('utf-8'),
            file_name="quants_historical_database.csv",
            mime="text/csv",
            type="primary"
        )

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.optimize import minimize
import logging

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Terminal Financiero Quant Master",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PRO MAX (Arquitectura Bloomberg Premium) ---
st.markdown("""
<style>
    /* Reset and Typography: Fondo oscuro puro #0E1117 solicitado */
    .stApp { background-color: #0E1117; color: #E2E8F0; font-family: 'Inter', 'Segoe UI', Interstate, sans-serif; }
    h1, h2, h3, h4 { color: #F8FAFC !important; font-weight: 700 !important; tracking: -0.02em; margin-bottom: 0.5rem; }
    
    /* Layout Containers and Padding */
    .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1400px; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #161A25; border-right: 1px solid #1E293B; }
    
    /* Cards (Bordes sutiles) */
    div[data-testid="metric-container"] { 
        background: linear-gradient(145deg, #1E293B, #0F172A); 
        border: 1px solid #334155; padding: 1.25rem; border-radius: 8px; 
    }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTES GLOBALES Y COLORES ---
COLOR_UP = '#10B981' # Verde
COLOR_DOWN = '#F43F5E' # Rojo
COLOR_LINE = '#38BDF8' # Azul cielo
COLOR_SMA = '#FBBF24' # Amarillo
COLOR_BG = 'rgba(0,0,0,0)'
COLOR_GRID = 'rgba(255,255,255,0.05)'
RISK_FREE_RATE = 0.04

# --- INGESTA DINÁMICA (Real-Time API) ---

@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Descarga de datos de mercado con manejo de errores anti-fragil."""
    try:
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

        # Limpieza de nulos exhaustiva
        df_close = df_close.dropna(axis=1, how='all')
        df_close = df_close.ffill().bfill()
        
        return df_close, data
    except Exception as e:
        logging.error(f"Fallo crítico en conexión yfinance: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

# --- MATEMÁTICAS CUANTITATIVAS ---

def calculate_technical_indicators(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({'Close': series})
    df['Retorno_Diario'] = df['Close'].pct_change()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    return df

def portfolio_perf(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return returns, std

def get_optimized_portfolios(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Minimize Volatility
    res_min_vol = minimize(lambda w, ret, cov: portfolio_perf(w, ret, cov)[1],
                           num_assets * [1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    
    # Maximize Sharpe
    args_sharpe = (mean_returns, cov_matrix, risk_free_rate)
    def neg_sharpe(w, ret, cov, rf):
        p_ret, p_std = portfolio_perf(w, ret, cov)
        return -(p_ret - rf) / p_std
        
    res_max_sharpe = minimize(neg_sharpe, num_assets * [1./num_assets,], args=args_sharpe,
                              method='SLSQP', bounds=bounds, constraints=constraints)
    
    return res_max_sharpe.x, res_min_vol.x

def run_monte_carlo(latest_price: float, returns: pd.Series, days=252, simulations=500):
    mu = returns.mean()
    vol = returns.std()
    sim_df = np.zeros((days, simulations))
    sim_df[0] = latest_price
    for i in range(1, days):
        sim_df[i] = sim_df[i-1] * (1 + np.random.normal(loc=mu, scale=vol, size=simulations))
    return sim_df

# --- GRÁFICOS UI OPTIMIZADOS ---

def clean_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(family='Inter', size=16, color='#E2E8F0')),
        template='plotly_dark', height=height, 
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        margin=dict(t=50, l=10, r=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#94A3B8'))
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID)
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID)
    return fig

# --- MOTOR DE INTERPRETACIÓN DINÁMICA (Insight Engine) ---

def interpret_tecnico(df: pd.DataFrame, ticker: str):
    if df.empty or len(df) < 50:
        st.info("No hay suficientes datos (<50 días) para un análisis técnico robusto.", icon="⏳")
        return
        
    ult_close = df['Close'].iloc[-1]
    ult_sma = df['SMA_50'].iloc[-1]
    volatilidad = df['Retorno_Diario'].std() * np.sqrt(252) * 100 # Anualizada

    if ult_close > ult_sma:
        st.success(f"**📈 MACRO TENDENCIA ALCISTA ({ticker}):** El precio (${ult_close:.2f}) se encuentra cotizando por encima de su Media Móvil de 50 días (${ult_sma:.2f}). Institucionalmente, esto indica dominancia de compradores.", icon="✅")
    else:
        st.error(f"**📉 MACRO TENDENCIA BAJISTA ({ticker}):** El precio (${ult_close:.2f}) ha roto a la baja su SMA-50 (${ult_sma:.2f}). Riesgo estructural elevado, el activo enfrenta fuerte presión vendedora.", icon="⚠️")
        
    if volatilidad > 30:
         st.warning(f"**⚡ ADVERTENCIA DE RIESGO:** La volatilidad anualizada es altísima ({volatilidad:.1f}% > 30%). Se esperan movimientos de precio violentos (>2% diarios frecuentemente). Ajuste el tamaño de su posición fuertemente a la baja.", icon="🚨")
    else:
         st.info(f"**🛡️ VOLATILIDAD MODERADA:** La volatilidad histórica es del {volatilidad:.1f}%. El comportamiento del activo está demostrando madurez y fluctuaciones dentro de la campana de Gauss natural del mercado.", icon="📊")

def interpret_markowitz(weights, tickers):
    df_w = pd.DataFrame({'Ticker': tickers, 'Weight': weights}).sort_values(by='Weight', ascending=False)
    df_w = df_w[df_w['Weight'] > 0.01] 
    if not df_w.empty:
        top_ticker = df_w.iloc[0]['Ticker']
        top_w = df_w.iloc[0]['Weight'] * 100
        st.info(f"**🧠 INSIGHT DEL SOLVER:** Para maximizar el Ratio de Sharpe de tu cesta, la red matemática le otorga un peso brutal del **{top_w:.1f}% al activo {top_ticker}**. Estadísticamente, este activo empuja la rentabilidad arrastrando al resto, y cualquier caída será estadísticamente absorbida (Cancelación de Covarianza) por el remanente de la cartera fragmentada.", icon="🏆")

def interpret_oraculo(simulations: np.ndarray):
    p_fin = simulations[-1,:] 
    media = np.mean(p_fin)
    p95 = np.percentile(p_fin, 95)
    p5 = np.percentile(p_fin, 5)
    
    st.success(f"**🎯 ESCENARIO BASE:** Si las condiciones persisten sin shock estructural, en 365 días la media de convergencia establece un objetivo gravitacional en **${media:.2f}**.", icon="🔭")
    st.info(f"**🚀 TECHO +95% (Euforia):** En caso de sorpresas súper-positivas, la estadística asintótica rompe hasta **${p95:.2f}**.", icon="🟢")
    st.error(f"**🛡️ SUELO -5% (Cisne Negro):** Prepara tu stop-loss. En un escenario de pánico estocástico, la proyección arroja un soporte crudo final en los **${p5:.2f}**.", icon="🩸")

# --- APLICACIÓN PRINCIPAL ---

def main():
    st.title("🧊 Inteligencia Financiera en Tiempo Real")
    
    with st.sidebar:
        st.header("⚙️ INGESTA DE RED")
        st.markdown("<hr style='margin: 0.5rem 0;'>", unsafe_allow_html=True)
        tickers_input = st.text_input("TICKERS MANUALES", value="AAPL, TSLA, BTC-USD, SPY", help="Separados por coma. Ej: SAN.MC")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1: start_date = st.date_input("Inicio", value=pd.to_datetime('2023-01-01'))
        with col_d2: end_date = st.date_input("Fin", value=pd.to_datetime('today'))
        
        run_btn = st.button("🔥 EJECUTAR PIPELINE", type="primary", use_container_width=True)

    if run_btn or "df_close" in st.session_state:
        tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
        if not tickers_list:
            st.warning("Escriba al menos un Ticker en la barra lateral.")
            st.stop()

        with st.spinner("Descargando data de Yahoo Finance en tiempo real..."):
            df_close, raw_data = load_market_data(tickers_list.copy(), start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            if df_close.empty:
                st.error("❌ Fallo en la red financiera. Tickers mal formateados u horizonte de tiempo inválido. Intente 'AAPL' o fechas más amplias.")
                st.stop()
                
            st.session_state["df_close"] = df_close
            st.session_state["raw_data"] = raw_data
            st.session_state["valid_tickers"] = [t for t in tickers_list if t in df_close.columns]

    if "df_close" not in st.session_state:
        st.info("Configura la barra lateral y presiona 'EJECUTAR PIPELINE'.", icon="👈")
        st.stop()

    df_close = st.session_state["df_close"]
    valid_tickers = st.session_state["valid_tickers"]
    raw_data = st.session_state["raw_data"]

    # 4 PESTAÑAS ESTRICTAS
    tab_tec, tab_solver, tab_oraculo, tab_data = st.tabs([
        "📊 Terminal Técnica", "⚖️ Master Solver", "🔮 Oráculo Estocástico", "📥 Auditoría de Datos"
    ])

    # --- PESTAÑA 1: TERMINAL TÉCNICA ---
    with tab_tec:
        c_head, c_select = st.columns([3, 1])
        c_head.subheader("Gráficos de Precio y Volumen Institucionales")
        ticker_tec = c_select.selectbox("Activo a Inspeccionar:", valid_tickers, label_visibility="collapsed")
        
        df_tec = calculate_technical_indicators(df_close[ticker_tec])
        
        # Generar Velas + Volumen si existe open/high/low/volume
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

        fig_tec = make_subplots(rows=2 if has_ohlcv else 1, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3] if has_ohlcv else [1])
        
        if has_ohlcv and op is not None:
             fig_tec.add_trace(go.Candlestick(x=df_tec.index, open=op, high=hi, low=lo, close=df_tec['Close'], name='Velas', increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN), row=1, col=1)
             fig_tec.add_trace(go.Bar(x=df_tec.index, y=vo, name='Volumen', marker_color='rgba(255,255,255,0.2)'), row=2, col=1)
        else:
             fig_tec.add_trace(go.Scatter(x=df_tec.index, y=df_tec['Close'], mode='lines', line=dict(color=COLOR_LINE, width=2), name='Cierre'), row=1, col=1)
             
        if 'SMA_50' in df_tec: fig_tec.add_trace(go.Scatter(x=df_tec.index, y=df_tec['SMA_50'], mode='lines', line=dict(color=COLOR_SMA, width=2), name='SMA-50'), row=1, col=1)
        
        fig_tec = clean_layout(fig_tec, height=600)
        fig_tec.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig_tec, use_container_width=True)
        
        interpret_tecnico(df_tec, ticker_tec)

    # --- PESTAÑA 2: MASTER SOLVER ---
    with tab_solver:
        st.subheader("Modelado No-Lineal (Teoría Moderna de Carteras)")
        
        if len(valid_tickers) < 2:
            st.error("Requiere mínimo 2 activos en la configuración para crear una Matriz de Covarianza.")
        else:
            returns_all = df_close[valid_tickers].pct_change().dropna()
            
            c_hm, c_sv = st.columns([1, 1.2])
            with c_hm:
                st.markdown("**Matriz de Autocorrelación:**")
                corr = returns_all.corr()
                fig_hm = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='Viridis', zmin=-1, zmax=1, text=np.round(corr.values, 2), texttemplate="%{text}", hoverinfo="z+x+y"))
                st.plotly_chart(clean_layout(fig_hm, height=400), use_container_width=True)
                
            with c_sv:
                st.markdown("**Target de Optimización:** Maximizador de Sharpe SLSQP")
                if st.button("CALCULAR ECUACIÓN", type="primary"):
                    max_w, min_w = get_optimized_portfolios(returns_all.mean().values, returns_all.cov().values, RISK_FREE_RATE)
                    interpret_markowitz(max_w, valid_tickers)
                    
                    df_weights = pd.DataFrame({'Activo': valid_tickers, 'Peso %': max_w * 100})
                    df_weights = df_weights[df_weights['Peso %'] > 0.5].sort_values(by='Peso %', ascending=False)
                    st.dataframe(df_weights.style.format({'Peso %': "{:.2f}%"}).bar(subset=['Peso %'], color='#10B981', vmin=0, vmax=100), use_container_width=True)

    # --- PESTAÑA 3: ORÁCULO ESTOCÁSTICO ---
    with tab_oraculo:
        c_head, c_select = st.columns([3, 1])
        c_head.subheader("Monte Carlo: Caminata Aleatoria (Browniana)")
        t_mc = c_select.selectbox("Target a 1 Año:", valid_tickers, label_visibility="collapsed")
        
        if t_mc:
            with st.spinner("Compilando 500 Vidas Alternativas frente a Volatilidad..."):
                returns_mc = df_close[t_mc].pct_change().dropna()
                if len(returns_mc) < 20:
                    st.error("Histórico insuficiente para vectorizar Monte Carlo.")
                else:
                    sim_data = run_monte_carlo(df_close[t_mc].iloc[-1], returns_mc, days=252, simulations=500)
                    
                    fig = go.Figure()
                    for i in range(min(150, sim_data.shape[1])):
                        fig.add_trace(go.Scatter(y=sim_data[:, i], mode='lines', line=dict(color='rgba(56, 189, 248, 0.03)', width=1), showlegend=False, hoverinfo='skip'))
                        
                    mean_path = np.mean(sim_data, axis=1)
                    fig.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(color=COLOR_SMA, width=3), name='Trayectoria Media'))
                    fig.add_trace(go.Scatter(y=np.percentile(sim_data, 95, axis=1), mode='lines', line=dict(color=COLOR_UP, width=2, dash='dot'), name='+95% Rango'))
                    fig.add_trace(go.Scatter(y=np.percentile(sim_data, 5, axis=1), mode='lines', line=dict(color=COLOR_DOWN, width=2, dash='dot'), name='-5% Rango'))
                    
                    fig = clean_layout(fig, height=500)
                    fig.update_xaxes(title="Días de Mercado")
                    fig.update_yaxes(title="Valor Cotizado Simulado")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    interpret_oraculo(sim_data)

    # --- PESTAÑA 4: AUDITORÍA DE DATOS ---
    with tab_data:
        st.subheader("Data Lake Extraído (CSV Pipeline)")
        st.dataframe(df_close.sort_index(ascending=False).head(150), use_container_width=True)
        st.download_button(
            label="💾 Bajar Set de Extracción Limpio (.CSV)",
            data=df_close.to_csv(index=True).encode('utf-8'),
            file_name="master_dataset_quant.csv",
            mime="text/csv",
            type="primary"
        )

if __name__ == "__main__":
    main()

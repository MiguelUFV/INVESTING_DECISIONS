import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from scipy import stats
from scipy.optimize import minimize

    # --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Terminal Financiero Quant Master",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS PRO MAX (Alta Gama, Espaciado, Inter UI) ---
st.markdown("""
<style>
    /* Reset and Typography */
    .stApp { background-color: #0B0F19; color: #E2E8F0; font-family: 'Inter', 'Segoe UI', sans-serif; }
    h1, h2, h3, h4 { color: #F8FAFC !important; font-weight: 700 !important; tracking: -0.02em; margin-bottom: 0.5rem; }
    h1 { font-size: 2.2rem; } h2 { font-size: 1.6rem; } h3 { font-size: 1.25rem; }
    hr { border-color: #1E293B !important; margin: 2rem 0; }
    
    /* Layout Containers and Padding */
    .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1400px; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #0F172A; border-right: 1px solid #1E293B; }
    [data-testid="stSidebar"] .stMarkdown h2 { color: #94A3B8 !important; font-size: 1.1rem; }
    
    /* Tabs Redesign (Bloomberg Terminal feel) */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: 1px solid #1E293B; padding-bottom: 5px; }
    .stTabs [data-baseweb="tab"] { 
        height: 48px; background-color: transparent; border: none;
        color: #64748B; font-weight: 500; font-size: 1rem; padding: 0 1rem;
        transition: color 0.2sease, border-color 0.2sease;
    }
    .stTabs [aria-selected="true"] { 
        color: #38BDF8 !important; 
        border-bottom: 3px solid #38BDF8 !important; 
        background: radial-gradient(circle at bottom, rgba(56,189,248,0.1) 0%, transparent 70%); 
    }
    
    /* KPI Metrics Cards (Sleek Glassmorphism) */
    div[data-testid="metric-container"] { 
        background: linear-gradient(145deg, #1E293B, #0F172A); 
        border: 1px solid #334155; padding: 1.25rem; border-radius: 12px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
    }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700; color: #F8FAFC; }
    div[data-testid="stMetricLabel"] { font-size: 0.95rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
    
    /* Educational & Dynamic Boxes (Callouts) */
    .edu-box { 
        background-color: #0F172A; border: 1px solid #1E293B; border-left: 4px solid #8B5CF6; 
        padding: 1.5rem; border-radius: 8px; font-size: 0.95em; color: #CBD5E1; 
        line-height: 1.6; box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
    }
    .edu-title { 
        font-weight: 700; color: #A78BFA; margin-bottom: 0.75rem; font-size: 1.1em; 
        display: flex; align-items: center; gap: 0.5rem; 
    }
    
    /* Dynamic Analysis Box */
    .dynamic-analysis { 
        background-color: #1E293B; border: 1px solid #334155; padding: 1.25rem; 
        border-radius: 8px; font-size: 1em; color: #F1F5F9; margin-top: 1rem; 
    }
    .analysis-item { margin-bottom: 0.75rem; display: flex; align-items: flex-start; gap: 0.5rem; }
    
</style>
""", unsafe_allow_html=True)

# --- CONSTANTES GLOBALES Y COLORES ---
BENCHMARK_TICKER = 'SPY'
global RISK_FREE_RATE
RISK_FREE_RATE = 0.04

COLOR_UP = '#10B981' # Emerald Green
COLOR_DOWN = '#F43F5E' # Rose Red
COLOR_LINE = '#38BDF8' # Sky Blue
COLOR_SMA = '#FBBF24' # Amber
COLOR_RSI = '#A78BFA' # Violet
COLOR_BG = 'rgba(0,0,0,0)'
COLOR_GRID = 'rgba(255,255,255,0.05)'

# --- DESCARGA DE DATOS ---

@st.cache_data(show_spinner="📡 Sincronizando datos de mercado...", ttl=3600)
def download_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    if 'SPY' not in tickers:
        tickers.append('SPY')
        
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        df_close = data['Close']
    else:
        df_close = pd.DataFrame({tickers[0]: data['Close']}) if len(tickers) == 1 else data

    df_close = df_close.ffill().bfill()
    return df_close, data

# --- MATEMÁTICAS CUANTITATIVAS ---

def calculate_technical_indicators(series: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({'Close': series})
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = sma_20 + (std_20 * 2)
    df['BB_Lower'] = sma_20 - (std_20 * 2)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
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
    
    treynor = (mean_ret_ann - risk_free_rate) / beta if beta != 0 else 0
    
    active_return = ret_asset - ret_market
    tracking_error_ann = active_return.std() * np.sqrt(252)
    information_ratio = active_return.mean() * 252 / tracking_error_ann if tracking_error_ann != 0 else 0
    
    cum_returns = (1 + ret_asset).cumprod()
    mdd = ((cum_returns - cum_returns.cummax()) / cum_returns.cummax()).min()
    
    return {
        'Vol_Ann': std_ann, 'Ret_Ann': mean_ret_ann,
        'Mkt_Ret_Ann': mean_market_ann, 'Sharpe': sharpe,
        'Sortino': sortino, 'Treynor': treynor, 'IR': information_ratio,
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
                              
    min_v_ret, min_v_std = portfolio_perf(res_min_vol.x, mean_returns, cov_matrix)
    max_s_ret, max_s_std = portfolio_perf(res_max_sharpe.x, mean_returns, cov_matrix)
    
    num_ports = 3000
    results = np.zeros((3, num_ports))
    for i in range(num_ports):
        w = np.random.random(num_assets)
        w /= np.sum(w)
        p_ret, p_std = portfolio_perf(w, mean_returns, cov_matrix)
        results[0,i] = p_std
        results[1,i] = p_ret
        results[2,i] = (p_ret - risk_free_rate) / p_std
        
    return res_min_vol.x, res_max_sharpe.x, results, (min_v_ret, min_v_std), (max_s_ret, max_s_std)

def run_monte_carlo(latest_price: float, returns: pd.Series, days=252, simulations=500):
    mu = returns.mean()
    vol = returns.std()
    sim_df = np.zeros((days, simulations))
    sim_df[0] = latest_price
    for i in range(1, days):
        sim_df[i] = sim_df[i-1] * (1 + np.random.normal(loc=mu, scale=vol, size=simulations))
    return sim_df

# --- GRÁFICOS UI OPTIMIZADOS (Plotly Theme Clean) ---

def clean_layout(fig, title="", height=400):
    fig.update_layout(
        title=dict(text=title, font=dict(family='Inter', size=16, color='#E2E8F0')),
        template='plotly_dark', height=height, 
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        margin=dict(t=50, l=10, r=10, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color='#94A3B8')),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=True, gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, tickfont=dict(color='#64748B'))
    fig.update_yaxes(showgrid=True, gridcolor=COLOR_GRID, zerolinecolor=COLOR_GRID, tickfont=dict(color='#64748B'))
    return fig

def plot_drawdown_underwater(returns: pd.Series, ticker: str):
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdowns = ((cum_returns - rolling_max) / rolling_max) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drawdowns.index, y=drawdowns, fill='tozeroy', mode='lines', 
                             line=dict(color=COLOR_DOWN, width=0), fillcolor='rgba(244, 63, 94, 0.4)', name='Drawdown'))
    fig = clean_layout(fig, title=f"Agonía del Inversor (Underwater Drawdown) - {ticker}", height=300)
    fig.update_yaxes(title="Caída desde Pico (%)")
    return fig

def create_heatmap(returns_df: pd.DataFrame) -> go.Figure:
    corr = returns_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale=[[0.0, COLOR_UP], [0.5, '#0F172A'], [1.0, COLOR_DOWN]], zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}", hoverinfo="z+x+y", showscale=False
    ))
    fig = clean_layout(fig, title="Matriz de Autocorrelación Activos", height=450)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

def create_efficient_frontier_chart(results, min_v_metrics, max_s_metrics):
    fig = go.Figure()
    
    # Nube simulaciones aleatorias
    fig.add_trace(go.Scatter(
        x=results[0,:], y=results[1,:], mode='markers',
        marker=dict(color=results[2,:], colorscale='Viridis', showscale=True, 
                    size=4, opacity=0.4, colorbar=dict(title='Sharpe Ratio', outlinewidth=0, tickfont=dict(color='#94A3B8'))),
        name='Simulaciones Subyacentes', hoverinfo='none'
    ))
    
    # Max Sharpe
    fig.add_trace(go.Scatter(
        x=[max_s_metrics[1]], y=[max_s_metrics[0]], mode='markers+text',
        marker=dict(color=COLOR_UP, size=20, symbol='star', line=dict(color='white', width=1.5)),
        text=['🎯 MAX SHARPE'], textposition='top center', textfont=dict(color=COLOR_UP, weight='bold', size=12),
        name='Máximo Sharpe'
    ))
    
    # Min Volatility
    fig.add_trace(go.Scatter(
        x=[min_v_metrics[1]], y=[min_v_metrics[0]], mode='markers+text',
        marker=dict(color='#38BDF8', size=16, symbol='diamond', line=dict(color='white', width=1.5)),
        text=['🛡️ MIN RIESGO'], textposition='bottom center', textfont=dict(color='#38BDF8', weight='bold', size=12),
        name='Mínimo Riesgo'
    ))
    
    fig = clean_layout(fig, title="Gráfico Topológico: La Frontera Eficiente de Markowitz", height=450)
    fig.update_xaxes(title="Volatilidad Esperada Anualizada (Riesgo σ)", showgrid=True)
    fig.update_yaxes(title="Retorno Promedio Esperado Anual", showgrid=True)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5))
    return fig

def create_monte_carlo_chart(simulations_df: np.ndarray, ticker: str) -> go.Figure:
    fig = go.Figure()
    for i in range(min(150, simulations_df.shape[1])):
        fig.add_trace(go.Scatter(y=simulations_df[:, i], mode='lines', 
                                 line=dict(color='rgba(56, 189, 248, 0.02)', width=1), showlegend=False, hoverinfo='skip'))
        
    mean_path = np.mean(simulations_df, axis=1)
    p_5 = np.percentile(simulations_df, 5, axis=1)
    p_95 = np.percentile(simulations_df, 95, axis=1)
    
    fig.add_trace(go.Scatter(y=mean_path, mode='lines', line=dict(color=COLOR_SMA, width=3), name='Trayectoria Media'))
    fig.add_trace(go.Scatter(y=p_95, mode='lines', line=dict(color=COLOR_UP, width=2, dash='dot'), name='Escenario +95%'))
    fig.add_trace(go.Scatter(y=p_5, mode='lines', line=dict(color=COLOR_DOWN, width=2, dash='dot'), name='Escenario -95%'))
    
    fig = clean_layout(fig, title=f"Oráculo Estocástico (500 Vidas) - {ticker}", height=500)
    fig.update_xaxes(title="Días de Trading Estocásticos (Futuro)")
    fig.update_yaxes(title="Niveles de Cotización Simulada")
    return fig


# --- AVISOS Y TEXTOS UI ---

def render_education_box(title: str, content: str):
    st.markdown(f'<div class="edu-box"><div class="edu-title">📖 {title}</div>{content}</div>', unsafe_allow_html=True)

def interpret_technical(df: pd.DataFrame) -> str:
    ult = df.iloc[-1]
    msgs = ["<div class='dynamic-analysis'><h4 style='color:#38BDF8; margin-top:0;'>🧠 Análisis Técnico Generativo</h4>"]
    trend_color = COLOR_UP if ult['Close'] > ult['SMA_50'] else COLOR_DOWN
    trend_txt = "Fuerza Alcista" if ult['Close'] > ult['SMA_50'] else "Presión Bajista"
    msgs.append(f"<div class='analysis-item'><span style='color:{trend_color}; font-size:1.2em;'>●</span> <b>Tendencia Primaria ({trend_txt}):</b> Cotización actual alineada {'por encima' if ult['Close'] > ult['SMA_50'] else 'por debajo'} de la medía institucional (50 periodos).</div>")
    rsi = ult['RSI_14']
    rsi_icon = "⚠️" if rsi >= 70 else "🎯" if rsi <= 30 else "⚡"
    estado_rsi = "Sobrecompra Crítica (Riesgo Corrección)" if rsi >= 70 else "Sobreventa (Pánico / Soporte Potencial)" if rsi <= 30 else "Normalizado (Neutral)"
    msgs.append(f"<div class='analysis-item'><span>{rsi_icon}</span> <b>RSI a {rsi:.1f} ({estado_rsi}):</b> Lectura del indicador de momentum estocástico.</div>")
    macd = ult['MACD']
    sig = ult['Signal_Line']
    macd_color = COLOR_UP if (ult['MACD_Hist'] > 0 and macd > sig) else COLOR_DOWN
    macd_txt = "Señal de Compra (Expansión)" if (ult['MACD_Hist'] > 0 and macd > sig) else "Señal de Venta (Contracción)"
    msgs.append(f"<div class='analysis-item'><span style='color:{macd_color}; font-size:1.2em;'>●</span> <b> MACD ({macd_txt}):</b> Convergencia/Divergencia de Medias Móviles alineada con el sesgo direccional del momento.</div>")
    msgs.append("</div>")
    return "".join(msgs)

def interpret_institucional(res: dict, ticker: str) -> str:
     msgs = [f"<div class='dynamic-analysis'><h4 style='color:#A78BFA; margin-top:0;'>🔬 Veredicto Cuantitativo: {ticker} vs Benchmark</h4>"]
     b = res['Beta']
     b_desc = "Perfil Agresivo (Apariencia especulativa)" if b > 1.2 else "Perfil Defensivo (Refugio)" if b < 0.8 else "Perfil Pasivo (Sombra del Mdo)"
     msgs.append(f"<div class='analysis-item'><span>⚖️</span> <b>Riesgo Sistémico - Beta a {b:.2f} ({b_desc}):</b> Multiplicador de volatilidad esperada en impactos macro.</div>")
     a = res['Alpha'] * 100
     a_color = COLOR_UP if a > 0 else COLOR_DOWN
     a_desc = "Creación de Valor Neto" if a > 0 else "Destrucción de Valor"
     msgs.append(f"<div class='analysis-item'><span style='color:{a_color}; font-size:1.2em;'>●</span> <b>Alpha de Jensen a {a:.2f}% ({a_desc}):</b> Rendimiento intrínseco de la gestión corporativa tras descontar el riesgo provisto por la estructura del mercado.</div>")
     s = res['Sharpe']
     s_icon = "🏆" if s > 1.2 else "✅" if s > 0.6 else "🚮"
     msgs.append(f"<div class='analysis-item'><span>{s_icon}</span> <b>Eficacia de Riesgo - Sharpe a {s:.2f}:</b> Compensación estandarizada por cada unidad de volatilidad soportada (Umbral institucional de excelencia > 1.0).</div>")
     msgs.append("</div>")
     return "".join(msgs)

def interpret_markowitz(weights, tickers, strategy):
    df_w = pd.DataFrame({'Ticker': tickers, 'Weight': weights}).sort_values(by='Weight', ascending=False)
    df_w = df_w[df_w['Weight'] > 0.01] # Filtrar pesos insignificantes (<1%)
    
    if df_w.empty: return ""
    
    top_ticker = df_w.iloc[0]['Ticker']
    top_w = df_w.iloc[0]['Weight'] * 100
    
    msgs = ["<div class='dynamic-analysis'>"]
    if strategy == 'Max Sharpe':
        msgs.append(f"<h4 style='color:#10B981; margin-top:0;'>🧠 Análisis Dinámico: Estrategia Máximo Sharpe</h4>")
        msgs.append(f"<div class='analysis-item'><span>🚀</span> <b>Razón Asignación Fuerte:</b> El algoritmo ha priorizado maximizar el retorno por cada unidad de riesgo asumida. Observa que el motor le ha otorgado un peso hiper-dominante del <b>{top_w:.1f}% al activo {top_ticker}</b>. Estadísticamente, este activo empuja de la inercia alcista de la cartera y sus baches están siendo amortiguados por los porcentajes minúsculos secundarios del resto de la cesta.</div>")
    else:
         msgs.append(f"<h4 style='color:#38BDF8; margin-top:0;'>🧠 Análisis Dinámico: Estrategia Mínima Volatilidad</h4>")
         msgs.append(f"<div class='analysis-item'><span>🛡️</span> <b>Razón Asignación Fuerte (Defensiva):</b> Operación Refugio Activa. El ordenador ha forzado la ecuación para eliminar el pánico, buscando un estancamiento controlado. Ha inyectado nada menos que un <b>{top_w:.1f}% de tu patrimonio en {top_ticker}</b>. Seguramente su matriz de covarianza contra el resto actuaba cancelando sistemáticamente las fluctuaciones violentas. Has sacrificado rentabilidad para poder dormir.</div>")
         
    msgs.append("</div>")
    return "".join(msgs)

def interpret_monte_carlo(sim_df: np.ndarray, ticker: str) -> str:
     p_fin = sim_df[-1,:] 
     media = np.mean(p_fin)
     p95 = np.percentile(p_fin, 95)
     p5 = np.percentile(p_fin, 5)
     return f"""
     <div class='dynamic-analysis'>
         <h4 style='color:#FBBF24; margin-top:0;'>🔭 Proyección de Objetivos a 365 Días</h4>
         <div class='analysis-item'><span>🎯</span> <b>Escenario Base (Media Estocástica):</b> El modelo probabilístico sitúa el anclaje teórico gravitacional en los <b>${media:.2f}</b> si las variables estadísticas no sufren shocks estructurales.</div>
         <div class='analysis-item'><span style='color:{COLOR_UP}; font-size:1.2em;'>●</span> <b>Desviación Superior (+95%):</b> La banda alta (entorno de euforia o hipercrecimiento inesperado) proyecta picos de cotización hasta <b>${p95:.2f}</b>.</div>
         <div class='analysis-item'><span style='color:{COLOR_DOWN}; font-size:1.2em;'>●</span> <b>Cisne Negro / Corrección Severa (-5%):</b> El piso probabilístico, en caso de caídas sistémicas continuas, marca un soporte estructural en <b>${p5:.2f}</b>.</div>
     </div>
     """

# --- APLICACIÓN PRINCIPAL ---

def main():
    st.markdown("<h1>🧊 Terminal Cuantitativo <i>Prime</i></h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; font-size:1.1rem; margin-bottom: 2rem;'>Plataforma institucional de análisis exploratorio, simulación de riesgos y síntesis estocástica.</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h2>⚙️ CONSOLA DE OPERACIONES</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0.5rem 0 1.5rem 0;'>", unsafe_allow_html=True)
        with st.expander("🇺🇸 Renta Variable USA", expanded=True):
            us_tech = st.multiselect("Big Tech:", ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"], default=["AAPL", "MSFT"])
            us_fin = st.multiselect("Bancos / Value:", ["BRK-B", "JPM", "V", "JNJ", "HD"])
        with st.expander("🇪🇺 Renta Variable EU"):
            eu_stock = st.multiselect("Ibex & EuroStoxx:", ["SAN.MC", "IBE.MC", "ITX.MC", "ASML", "MC.PA"], default=["SAN.MC"])
        with st.expander("🌐 ETFs / Bonos / Cripto"):
            etf_idx = st.multiselect("Índices Principales:", ["SPY", "QQQ", "TLT", "GLD"])
            crypto = st.multiselect("Criptoactivos:", ["BTC-USD", "ETH-USD", "SOL-USD"], default=["BTC-USD"])
            
        st.markdown("<br>", unsafe_allow_html=True)
        custom_input = st.text_input("TICKERS PERSONALIZADOS", placeholder="Separados por coma: AMD, EURUSD=X")
        
        st.markdown("<br><b>HORIZONTE Y MACRO</b>", unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2)
        with col_d1: start_date = st.date_input("Inicio", value=pd.to_datetime('2023-01-01'))
        with col_d2: end_date = st.date_input("Fin", value=pd.to_datetime('today'))
        
        risk_free_val = st.number_input("Tasa Libre Riesgo (Rf %)", value=4.0, step=0.1)
        global RISK_FREE_RATE
        RISK_FREE_RATE = risk_free_val / 100.0

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("SINCRONIZAR TERMINAL", type="primary", use_container_width=True)

    if run_btn or "df_close" in st.session_state:
        tickers_list = us_tech + us_fin + eu_stock + etf_idx + crypto
        if custom_input.strip():
            tickers_list.extend([t.strip().upper() for t in custom_input.split(',') if t.strip()])
            
        tickers_list = list(dict.fromkeys(tickers_list))
        if not tickers_list:
            st.warning("Selección vacía. Configure el panel lateral.")
            st.stop()

        with st.spinner("Compilando vectores de precio e históricas..."):
            df_close, raw_data = download_data(tickers_list.copy(), start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            st.session_state["df_close"] = df_close
            st.session_state["raw_data"] = raw_data
            st.session_state["valid_tickers"] = [t for t in tickers_list if t in df_close.columns]

    if "df_close" not in st.session_state:
        st.info("💡 Por favor, configure el ecosistema de activos en la barra lateral izquierda y sincronice el terminal para comenzar.")
        st.stop()

    df_close = st.session_state["df_close"]
    valid_tickers = st.session_state["valid_tickers"]
    raw_data = st.session_state["raw_data"]

    tab_tec, tab_quant, tab_solver, tab_oraculo, tab_data = st.tabs([
        "  CHARTING  ", "  Q-RISK  ", "  SOLVER  ", "  PROPHET  ", "  RAW DATA  "
    ])

    with tab_tec:
        c_head, c_select = st.columns([3, 1])
        c_head.markdown("### Escaneo Técnico Multicapa")
        ticker_tec = c_select.selectbox("Seleccionar Lente:", valid_tickers, label_visibility="collapsed")
        df_tec = calculate_technical_indicators(df_close[ticker_tec])
        
        fig_tec = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
        
        has_ohlc = ('Open' in raw_data.columns)
        if isinstance(raw_data.columns, pd.MultiIndex):
            has_ohlc = has_ohlc and (ticker_tec in raw_data['Open'].columns)
            open_s = raw_data['Open'][ticker_tec] if has_ohlc else None
            high_s = raw_data['High'][ticker_tec] if has_ohlc else None
            low_s = raw_data['Low'][ticker_tec] if has_ohlc else None
        else:
            open_s = raw_data['Open'] if 'Open' in raw_data.columns else None
            high_s = raw_data['High'] if 'High' in raw_data.columns else None
            low_s = raw_data['Low'] if 'Low' in raw_data.columns else None

        if has_ohlc and open_s is not None:
             fig_tec.add_trace(go.Candlestick(x=df_tec.index, open=open_s, high=high_s, low=low_s, close=df_tec['Close'], name='Cotización', increasing_line_color=COLOR_UP, decreasing_line_color=COLOR_DOWN), row=1, col=1)
        else:
             fig_tec.add_trace(go.Scatter(x=df_tec.index, y=df_tec['Close'], mode='lines', line=dict(color=COLOR_LINE, width=2), name='Cierre'), row=1, col=1)
             
        if 'SMA_50' in df_tec: fig_tec.add_trace(go.Scatter(x=df_tec.index, y=df_tec['SMA_50'], mode='lines', line=dict(color=COLOR_SMA, width=2), name='SMA-50'), row=1, col=1)
        if 'BB_Upper' in df_tec:
            fig_tec.add_trace(go.Scatter(x=df_tec.index, y=df_tec['BB_Upper'], mode='lines', line=dict(color='rgba(56, 189, 248, 0.2)', width=1), name='Banda Top'), row=1, col=1)
            fig_tec.add_trace(go.Scatter(x=df_tec.index, y=df_tec['BB_Lower'], mode='lines', line=dict(color='rgba(56, 189, 248, 0.2)', width=1), fill='tonexty', fillcolor='rgba(56, 189, 248, 0.05)', name='Banda Base'), row=1, col=1)
        
        if 'MACD' in df_tec:
            fig_tec.add_trace(go.Scatter(x=df_tec.index, y=df_tec['MACD'], line=dict(color=COLOR_LINE, width=1.5), name='MACD'), row=2, col=1)
            fig_tec.add_trace(go.Scatter(x=df_tec.index, y=df_tec['Signal_Line'], line=dict(color=COLOR_SMA, width=1.5), name='Señal'), row=2, col=1)
            colors_hist = [COLOR_UP if val > 0 else COLOR_DOWN for val in df_tec['MACD_Hist']]
            fig_tec.add_trace(go.Bar(x=df_tec.index, y=df_tec['MACD_Hist'], marker_color=colors_hist, name='Momentum'), row=2, col=1)
            
        if 'RSI_14' in df_tec:
            fig_tec.add_trace(go.Scatter(x=df_tec.index, y=df_tec['RSI_14'], line=dict(color=COLOR_RSI, width=1.5), name='RSI'), row=3, col=1)
            fig_tec.add_hline(y=70, line=dict(dash="dot", width=1, color=COLOR_DOWN), row=3, col=1)
            fig_tec.add_hline(y=30, line=dict(dash="dot", width=1, color=COLOR_UP), row=3, col=1)
            
        fig_tec = clean_layout(fig_tec, title="", height=800)
        fig_tec.update_xaxes(rangeslider_visible=False, title="")
        st.plotly_chart(fig_tec, use_container_width=True)
        st.markdown(interpret_technical(df_tec), unsafe_allow_html=True)

    with tab_quant:
        c_head, c_select = st.columns([3, 1])
        c_head.markdown("### Auditoría de Rendimiento Ajustado (CAPM)")
        t_sel = c_select.selectbox("Enfocar Activo:", valid_tickers, label_visibility="collapsed", key="q_sel")
        
        if t_sel and BENCHMARK_TICKER in df_close.columns:
            ret_act = df_close[t_sel].pct_change().dropna()
            ret_mkt = df_close[BENCHMARK_TICKER].pct_change().dropna()
            res = get_performance_metrics(ret_act, ret_mkt, RISK_FREE_RATE)
            if res:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Retorno Histórico", f"{res['Ret_Ann']*100:.2f}%")
                col2.metric("Rating de Sharpe", f"{res['Sharpe']:.2f}")
                col3.metric("Alpha Generado", f"{res['Alpha']*100:.2f}%")
                col4.metric("Estrés Máximo (Drawdown)", f"{res['MDD']*100:.2f}%")
                st.markdown(interpret_institucional(res, t_sel), unsafe_allow_html=True)
                g1, g2 = st.columns(2)
                with g1:
                    fig_hist = px.histogram(ret_act*100, nbins=50, color_discrete_sequence=[COLOR_RSI], marginal='box')
                    fig_hist = clean_layout(fig_hist, title="Perfil de Distribución Promedio", height=350)
                    fig_hist.update_layout(showlegend=False)
                    st.plotly_chart(fig_hist, use_container_width=True)
                with g2:
                     fig_uw = plot_drawdown_underwater(ret_act, t_sel)
                     st.plotly_chart(fig_uw, use_container_width=True)

    with tab_solver:
        st.markdown("### Ecosistema Multi-Activo (Solver Markowitz Avanzado)")
        
        if len(valid_tickers) < 2:
            st.warning("⚠️ El solver necesita materia prima. Introduzca 2 o más activos globalmente en la configuración.")
        else:
            returns_all = df_close[valid_tickers].pct_change().dropna()
            
            # Sub-Tab Navigation for logical grouping
            st_solver1, st_solver2 = st.tabs(["1) Laboratorio de Rebalanceo", "2) Malla de Correlaciones"])
            
            with st_solver1:
                # Top Selectors
                s_col1, s_col2 = st.columns([3, 2])
                with s_col1:
                    port_assets = st.multiselect("Definir Componentes del Índice Personalizado:", valid_tickers, default=valid_tickers)
                with s_col2:
                     st.write("") # Spacer
                     strategy = st.radio("📡 Target de Optimización Estocástica:", 
                                         ["🏆 Máximo Ratio Sharpe", "🛡️ Mínima Volatilidad Absoluta"])
                
                if len(port_assets) >= 2:
                    if st.button("🚀 EJECUTAR MOTOR DE OPTIMIZACIÓN MULTIDIMENSIONAL", type="primary", use_container_width=True):
                        with st.spinner("Procesando Tensores de Covarianza & Resolviendo Frontera de Markowitz (SLSQP)..."):
                            ret_p = returns_all[port_assets]
                            min_w, max_w, sim_r, min_v_mets, max_s_mets = get_optimized_portfolios(ret_p.mean().values, ret_p.cov().values, RISK_FREE_RATE)
                            
                            target_w = max_w if "Sharpe" in strategy else min_w
                            target_mets = max_s_mets if "Sharpe" in strategy else min_v_mets
                            
                            st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
                            
                            # RESULTADOS EN DOS COLUMNAS
                            res_c1, res_c2 = st.columns([1.5, 1])
                            
                            # Left Column: Efficient Frontier
                            with res_c1:
                                fig_frontier = create_efficient_frontier_chart(sim_r, min_v_mets, max_s_mets)
                                st.plotly_chart(fig_frontier, use_container_width=True)
                                
                                with st.expander("¿Qué demonios es este gráfico curvado?"):
                                    render_education_box("La Frontera Eficiente", """
                                    El gráfico te muestra TODAS las combinaciones posibles que podrías armar con el porcentaje de tu dinero. 
                                    Los miles de puntitos son <b>Carteras Mediocres simuladas al azar</b>. La curva externa (el borde superior) es la <b>"Frontera Eficiente"</b>. El Solver ha detectado con precisión milimétrica donde está tu portfolio deseado flotando en esa frontera, marcándolo con un icono gigante para que veas empíricamente que has derrotado matemáticamente a cualquier otra combinación al azar posible.
                                    """)

                            # Right Column: Assigned Weights & Analysis
                            with res_c2:
                                st.markdown("#### Configuración de Patrimonio")
                                
                                # Visual Table of Weights
                                df_weights = pd.DataFrame({'Activo (Ticker)': port_assets, 'Peso Asignado (%)': target_w * 100})
                                df_weights = df_weights[df_weights['Peso Asignado (%)'] >= 0.5] # Eliminar microscopicos
                                df_weights = df_weights.sort_values(by='Peso Asignado (%)', ascending=False).reset_index(drop=True)
                                
                                # Bar chart styling for visual impact
                                st.dataframe(df_weights.style.format({'Peso Asignado (%)': "{:.2f}%"}).bar(subset=['Peso Asignado (%)'], color='#38BDF8', vmin=0, vmax=100), use_container_width=True)
                                
                                st.markdown(interpret_markowitz(target_w, port_assets, "Max Sharpe" if "Sharpe" in strategy else "Min Volatilidad"), unsafe_allow_html=True)
                else:
                    st.info("Añade al menos 2 activos en el multiselect superior.")

            with st_solver2:
                st.markdown("### Mapa de Termodinámica Conjunta (Correlaciones)")
                st.plotly_chart(create_heatmap(returns_all), use_container_width=True)
                st.caption("Estructura de Riesgo Cruzado: Tonos Verdes marcan Activos Defensivos Mutuos. Los Rojos sufrirán catástrofes de manera idéntica juntos.")

    with tab_oraculo:
        c_head, c_select = st.columns([3, 1])
        c_head.markdown("### Estocástica Predictiva: Convergencia Simulada")
        t_mc = c_select.selectbox("Target Simulación:", valid_tickers, label_visibility="collapsed")
        if t_mc:
            with st.spinner("Procesando Tensores de Probabilidad (500 Iteraciones)..."):
                returns_mc = df_close[t_mc].pct_change().dropna()
                sim_data = run_monte_carlo(df_close[t_mc].iloc[-1], returns_mc, days=252, simulations=500)
                st.plotly_chart(create_monte_carlo_chart(sim_data, t_mc), use_container_width=True)
                st.markdown(interpret_monte_carlo(sim_data, t_mc), unsafe_allow_html=True)

    with tab_data:
        st.markdown("### Data Lake Extraído a Terminal")
        st.dataframe(df_close.sort_index(ascending=False).head(100), use_container_width=True)
        csv_buffer = df_close.to_csv(index=True)
        st.download_button(
            label="Descargar Dataset Limpio (.CSV)",
            data=csv_buffer.encode('utf-8'),
            file_name="master_dataset_quant.csv",
            mime="text/csv",
            type="primary"
        )

if __name__ == "__main__":
    main()

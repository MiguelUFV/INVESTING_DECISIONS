# 🌍 Terminal Cuantitativo *Prime* (INVESTING_DECISIONS)

Plataforma institucional de análisis exploratorio, simulación de riesgos y síntesis estocástica basada en Teoría Moderna de Carteras (Harry Markowitz) y visualización avanzada de datos (Plotly).

## 🚀 Características Principales

1. **Dashboard Técnico Multicapa:** Análisis interactivo del precio usando bandas de Bollinger, medias móviles (SMA-50), RSI, MACD e histogramas de aceleración de momento.
2. **Q-Risk Analytics (CAPM):** Cálculo y comparación en tiempo real de métricas profesionales contra Benchmark (SPY). Generación dinámica de Alpha, Beta, Sharpe Ratio y Drawdowns sumergidos.
3. **Solver Multidimensional de Markowitz:** Optimizador estadístico puro iterativo con restricciones reales. Malla de correlaciones visual y generación en vivo de la **Frontera Eficiente**. 
4. **Oráculo Predictivo:** Motores de simulación estocástica continua (Monte Carlo: Trayectoria de Movimiento Browniano Simple) calculando probabilidades a 1 año bajo varianza algorítmica.
5. **Reportes IA:** Interpretación textual instantánea en base a los cálculos logrados tras la ingesta de las series.

## 🛠️ Tecnologías y Librerías Utilizadas

*   **Aplicación y UI Front-end:** `Streamlit`, `Markdown CSS`
*   **Gestión Estructural y Numérica:** `Pandas`, `NumPy`, `SciPy` (Solver SLSQP y Optimización Bayesiana paramétrica)
*   **Visualización Renderizada GL:** `Plotly` (Graph Objects y Express Line)
*   **Ingesta de Red Externa:** `yfinance` (APIs bursátiles latentes)
*   **Motores de I/O Temporales:** `pyarrow`, `fastparquet`

## 📦 Despliegue en Local

Clona el repositorio, asegúrate de tener Python instalado y arranca el entorno:
```bash
pip install -r requirements.txt
streamlit run dashboard_financiero.py
```

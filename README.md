# TERMINAL DE INTELIGENCIA FINANCIERA (QUANT ENGINE)

Plataforma institucional de análisis exploratorio, simulación estocástica de riesgos y modelado algorítmico basado en Teoría Moderna de Carteras (Harry Markowitz) y matemáticas financieras cuantitativas (CAPM, Value At Risk).

## ARQUITECTURA DEL SISTEMA 

El entorno ha sido estrictamente diseñado bajo el paradigma de interfaces de fondos de cobertura y terminales profesionales (arquitectura *Dark Form*, Blue Cobalt / Smoke aesthetics), priorizando la ingesta de datos en tiempo real (Real-Time API) y la purificación estadística del ruido del mercado mediante módulos interactivos de *Click-to-Expand Insight Engine*.

### MODULOS ESTRUCTURALES

1. **ANALISIS TECNICO Y MOMENTUM:** Análisis estructural interactivo del precio superponiendo directrices móviles críticas (SMA-50) e indicadores rezagados de comportamiento oscilante (RSI 14 y divergencias MACD) bajo esquemas lógicos sin ruido visual de fondo.
2. **METRICAS DE RIESGO DE MERCADO (CAPM):** Auditoría paramétrica frente al Benchmark (SPY) desplegando dinámicamente el rendimiento anualizado ajustado (Sharpe Ratio), coeficientes de asimetría direccional (Beta Sistémica) y extracción algorítmica de valor exógeno (Alpha de Jensen). Analítica complementada con medición estricta de exposiciones pasadas (*Underwater Drawdown*).
3. **OPTIMIZACION MATRICIAL DE MARKOWITZ:** Construcción topológica en tiempo real de la frontera eficiente por medio del optimizador cuadrático SLSQP iterativo sobre la Matriz de Autocorrelación de los retornos (Covarianzas Negativas) para despejar el vector de pesos que maximiza paramétricamente la compensación estadística Riesgo-Beneficio.
4. **PROYECCION ESTOCASTICA Y CONVERGENCIA (VAR):** Compilación sintética masiva usando simulación de Monte Carlo (Caminata Browniana Aleatoria, 500 n / 252 t) sobre distribución log-normal asintótica, parametrizando dinámicamente las bandas de confianza de los percentiles p95 y p05, y extrayendo el *Value at Risk (VaR)* algorítmico frente a desintegración inopinada del capital.
5. **PIPELINE DE EXTRACCION PURIFICADA:** Extracción íntegra y renderizado de la base de datos descargada libre de lagunas temporales para exportación final mediante CSV crudo a gestores de portafolio o ingesta algorítmica externa.

## REQUIREMENTOS DE INFRAESTRUCTURA

El motor corre enteramente sobre Python procesando cálculos optimizados de vectorización en local a partir de la API de *YFinance*.

*   `streamlit==1.36.0` (Motor de Interfaz Reactiva y Estado)
*   `pandas` / `numpy` (Manipulación Tensorial Matricial Lineal)
*   `scipy.optimize` (Optimizador Paramétrico SLSQP)
*   `plotly` (Renderizado de Motores Gráficos en GPU del navegador, sin *Gridlines*)
*   `yfinance` (Gateway a la Ingesta Temporal Externa)

## DESPLIEGUE Y COMPILACION

```bash
#  Asegurar entorno (virtual environment / global) e inyectar dependencias 
pip install -r requirements.txt

# Disparar compilación del Servidor Interactivo de Datos
streamlit run dashboard_financiero.py
```
*NOTA: Se requiere conectividad web pasante sin restricciones de firewall corporativo que bloqueen puertos 443 a los servidores raw históricos de Yahoo Finance.*

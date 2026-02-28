# 🌐 AURA WEALTH OS (Quantitative Terminal)

**Aura Wealth OS** es una infraestructura SaaS *Next-Gen* de análisis algorítmico, gestión de carteras y proyecciones de riesgo asimétrico. Diseñada con estándares de grado institucional (Hedge Funds) para inversores que requieren una aproximación táctica, visual y matemáticamente estricta a los mercados financieros globales.

---

## 🏛️ Arquitectura del Sistema

La plataforma está diseñada íntegramente en Python utilizando el paradigma de arquitecturas monolíticas reactivas de datos:

- **Frontend / Motor UI:** Interfaz construida sobre **Streamlit** modificada visualmente con CSS inyectado puro (Glassmorphism, Radial Gradients). Formularios asíncronos para evitar recargas excesivas y gestión de renderizado de alto contraste.
- **Data Lake Connector (ETL en Vivo):** Conector Web Scraper y API directa contra `yfinance` para extracción de Series Temporales intradiarias, Fundamentales Corporativos (Márgenes, PER, Cap) y Feed de Noticias propietarias de **Reuters/Bloomberg**.
- **Backend Cuantitativo:** Pipeline de tensores matemáticos y vectorizados (`NumPy` + `Pandas`) para la rápida síntesis de matrices de covarianza cruzada en menos de <200ms de latencia.
- **Pipeline de Reportes:** Generador dinámico en formato Markdown (`.md`) para Tear Sheets Institucionales y exportador en memoria I/O hacia `.xlsx` (Excel) con parseo `openpyxl`.

---

## 🧮 Modelos Matemáticos

Aura implementa funciones financieras bajo los tres grandes marcos teóricos del *Quantitative Finance*:

### 1. Modelo de Valoración de Activos (CAPM)
La plataforma mide el Factor de Riesgo inherente de cada posición iterado contra un Benchmark global (S&P 500).
- **Ratio de Sharpe:** Rentabilidad excedentaria asumiendo la Tasa Libre de Riesgo (Rf) penalizada por la Volatilidad Histórica (Desviación Estándar Anual).
- **Alpha de Jensen y Beta:** Diferenciación entre el Retorno del Mercado (Exposición Pasiva Sistémica) y el Valor Pila Absoluto (Habilidad del portfolio o activo para batir al mercado con menor riesgo direccional).

### 2. Teoría Moderna de Carteras (Harry Markowitz)
Implementación nativa del solucionador no lineal de `SciPy Minimize` (Método SLSQP) para encontrar el vértice absoluto de la **Frontera Eficiente**.
La plataforma calcula la matriz matemática de *Varianza-Covarianza* y dictamina la ponderación percentil teórica exacta que cada acción debe tener en el portafolio total para maximizar retornos destruyendo la volatilidad cruzada (correlación).

### 3. Proyecciones Estocásticas (Monte Carlo & VaR)
Proyección de caminos aleatorios que usan derivadas de dispersión para simular la degradación o crecimiento a 12 meses vista.
- **Value at Risk (VaR 95%):** El algoritmo advierte directamente del riesgo de *ruina* o máxima pérdida estadística probable en términos de porcentaje de capital para el siguiente impacto de peor caso en el mercado.

---

## 🚀 Instrucciones de Despliegue (Nube y Local)

La aplicación está completamente aislada de la máquina anfitriona y está lista para despliegues Continuos (CI/CD) tanto en infraestructuras Cloud ligeras (Streamlit Cloud) como mediante contenedores absolutos (Render, Railway, AWS ECS).

### A. Despliegue en Render (Recomendado vía Docker)
El repositorio cuenta con un `Dockerfile` optimizado (Python 3.10-slim) de muy bajo peso de RAM para instancias gratuitas o Micro-Instancias.
1. Haz **Fork** o clon de este repositorio en GitHub.
2. Inicia sesión en **Render.com** > Nuevo *Web Service*.
3. Enlaza tu GitHub y elige este repositorio.
4. Renderizará automáticamente detectando el `Dockerfile`. 
   *(Nota técnica: el Dockerfile ya ignora paquetes conflictivos Debian y define un `ENTRYPOINT` absoluto contra el `dashboard_financiero.py` exponiendo el puerto 8501).*
5. Espera al Build y pulsa el Botón **Live**. Listo.

### B. Despliegue Instantáneo en Streamlit Community Cloud
Si no deseas manejar contenedores, Streamlit Community Cloud es nativo:
1. Dirígete a [share.streamlit.io](https://share.streamlit.io).
2. Haz "New App" y selecciona la rama principal (`main`) de este repositorio de GitHub.
3. Rellena *"Main file path"* con `dashboard_financiero.py`.
4. El sistema autodetectará el archivo `requirements.txt` y lo levantará en línea.

### C. Instalación Local / Pruebas
Si deseas trastear con los algoritmos y testear la respuesta en tu máquina personal:
```bash
git clone https://github.com/migue/terminal_financiero.git
cd terminal_financiero
pip install -r requirements.txt
streamlit run dashboard_financiero.py
```
*(El navegador se abrirá en localhost:8501 automáticamente).*

## QuantifiLib

QuantifiLib is a research-oriented Python library for quantitative finance, developed by Quantifi Sogang.  
It provides a modular and extensible framework for systematic trading research, including data loading, event-based labeling, signal generation, backtesting, portfolio optimization, time series modeling, causal inference, and synthetic data generation.  

QuantifiLib is designed to serve as a full-stack infrastructure for academic finance research, integrating both traditional econometric models and modern machine learning techniques.

Whether you're building event-driven strategies, training machine learning models with purged cross-validation, or simulating asset prices with GANs, QuantifiLib offers a unified environment to streamline your research process.

### 📁 Project Structure

<pre lang="markdown">

quantifilib/
└── data/
    ├── data_loader/
    │   ├── yfinance_loader.py
    │   ├── fred_loader.py
    │   └── naver_loader.py
    └── stock_universe/
    │   └── wikipedia.py
    ├── features/
    │   ├── bar_sampling/
    │   │   ├── bar_feature.py
    │   │   ├── base_bars.py
    │   │   ├── core.py
    │   │   ├── imbalance_data_structures.py
    │   │   ├── microstructure.py
    │   │   ├── run_data_structures.py
    │   │   ├── standard_data_structures.py
    │   │   └── time_data_structures.py
    ├── metrics/
    │   ├── liquidity/
    │   │   ├── corwin_schultz.py
    │   │   ├── lambda.py
    │   │   ├── pin.py
    │   │   └── roll_models.py
    │   ├── risk/
    │   │   ├── market.py
    │   │   └── strategy.py
    ├── utils/
    │   └── fast_ewma.py
    
</pre>

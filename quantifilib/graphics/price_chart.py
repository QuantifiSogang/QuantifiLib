import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_candle_chart(
        data: pd.DataFrame,
        body_width: float = 2,
        tail_width: float = 0.5,
        white_soldier_color: str = 'tab:green',
        black_crow_color: str = 'tab:red',
        figsize=(6, 4),
        volume_chart: bool = False
    ) -> tuple:
    """
    Candlestick chart with optional volume bar chart below.

    Parameters:
    - data: DataFrame with ['Open', 'High', 'Low', 'Close'] (and 'Volume' if volume_chart=True)
    - body_width: Width of the candle body lines
    - tail_width: Width of the wick lines
    - white_soldier_color: Color for bullish candles
    - black_crow_color: Color for bearish candles
    - figsize: Tuple for figure size
    - volume_chart: If True, plots volume as a subplot

    Returns:
    - fig, ax: matplotlib figure and main axes
    """
    required_cols = ['Open', 'High', 'Low', 'Close']
    if volume_chart:
        required_cols.append('Volume')
    assert all(col in data.columns for col in required_cols), f"Data must contain columns: {required_cols}"

    data = data.copy()
    data.index = pd.to_datetime(data.index)

    if volume_chart:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw={'height_ratios': [3, 1]},
                                       sharex=True, dpi=100)
    else:
        fig, ax1 = plt.subplots(figsize=figsize, dpi=100)
        ax2 = None

    width = (data.index[1] - data.index[0]).days * 0.6
    width = max(width, 0.4)

    for idx, row in data.iterrows():
        open_, high, low, close = row['Open'], row['High'], row['Low'], row['Close']
        color = white_soldier_color if close >= open_ else black_crow_color
        x = mdates.date2num(idx)

        # Wick
        ax1.plot([x, x], [low, high], color='black', linewidth=tail_width)
        # Body
        ax1.plot([x, x], [open_, close], color=color, linewidth=body_width)

        # Volume chart
        if volume_chart:
            ax2.bar(x, row['Volume'], color=color, width=width, alpha=0.6)

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.tight_layout()
    return fig, ax1 if not volume_chart else (fig, (ax1, ax2))
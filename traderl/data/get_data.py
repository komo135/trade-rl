import pandas as pd
import pandas_datareader as pdr


def get_forex_data(symbol: str, timeframe: str):
    """
    :param symbol: AUDJPY, AUDUSD, EURCHF, EURGBP, EURJPY, EURUSD, GBPJPY, GBPUSD, USDCAD, USDCHF, USDJPY, XAUUSD
    :param timeframe: m15, m30, h1, h4, d1
    :return: pandas DataFrame
    """
    symbol = symbol.upper()

    url = "https://raw.githubusercontent.com/komo135/forex-historical-data/main/"
    url += symbol + "/" + symbol + timeframe.lower() + ".csv"

    return pd.read_csv(url)


def get_stock_data(symbol, start=None, end=None):
    return pdr.get_data_yahoo(symbol, strt=start, end=end)


__all__ = ["get_stock_data", "get_forex_data"]
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import pytz
from datetime import date, timedelta, datetime


def yahoo_finance_1d():
    df = pd.read_csv('tickers.csv', header=None, encoding='utf-8')
    tickers = df[0].to_list()
    names = df[1].to_list()

    # Get data from last 3000 days, enough since we only have crypto data since 2017
    ed = date.today()
    sd = datetime.strptime('2017-08-15', '%Y-%m-%d').date()
    dates_list = [sd + timedelta(days=x) for x in range((ed - sd).days + 1)]

    # Use UTC timezone, same as binance
    tz = pytz.timezone('Europe/London')
    ed_tz = tz.localize(pd.Timestamp(ed))
    sd_tz = tz.localize(pd.Timestamp(sd))

    result = pd.DataFrame()
    result['Dates'] = dates_list
    result = result.set_index('Dates')

    tickers_por_archivo = 100
    actual_file = 1

    # Download ticker data
    for i, ticker in enumerate(tqdm(tickers, desc="Downloading data"), start=1):
        try:
            data = yf.download(tickers=ticker, start=sd_tz, end=ed_tz, interval="1d")
            data['tic'] = ticker
            data['Dates'] = data.index
            result = pd.merge(result, data['Close'].rename(names[i-1]), how='outer', left_index=True, right_index=True)
        except Exception as e:
            print(f"Error downloading data from {ticker}: {str(e)}")
            continue

        # Store data
        if i % tickers_por_archivo == 0 or i == len(tickers):
            result = result.fillna(method='ffill')
            result = result.iloc[1:, :]
            result.to_csv(f'../binance_trading_environment/datasets/data/financial_data.csv', index=True, index_label='Dates')
            result = pd.DataFrame()
            actual_file += 1




yahoo_finance_1d()

import datetime
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import os

THREAD_COUNT = 500
PERIOD = 365


@st.cache()
def run(_):
    print("getting data")
    start = datetime.datetime.today().date() - datetime.timedelta(days=PERIOD)
    end = datetime.datetime.today().date() + datetime.timedelta(days=1)

    def clearNA(df):
        return (
            df.dropna(axis=1, how="all")).dropna(axis=0, how="all")

    def getStockData(tickers: list) -> pd.DataFrame:
        stocks = yf.download(tickers=tickers, start=start,
                             end=end, threads=THREAD_COUNT)['Close']
        if len(tickers) == 1:
            df = pd.DataFrame()
            df[tickers[0]] = stocks
            return df
        return clearNA(pd.DataFrame(stocks))

    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    scrape = table[0]
    sp500 = [i.replace(".", "-") for i in scrape["Symbol"].to_list()]
    table = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    scrape = table[3]
    nas100 = [i.replace(".", "-") for i in scrape["Ticker"].to_list()]

    tickerList = list(set(nas100 + sp500))

    if "data.csv" in os.listdir():
        dateMod = datetime.datetime.fromtimestamp(os.path.getmtime("data.csv"))
        if dateMod != end:
            copy = getStockData(tickerList)
            copy.to_csv("data.csv")
        else:
            copy = pd.read_csv("data.csv", parse_dates=['Date'])
            copy = copy.set_index("Date")
    else:
        copy = getStockData(tickerList)
        copy.to_csv("data.csv")

    copy["zeros"] = np.zeros(len(copy))

    return tickerList, copy


@st.cache
def getDf(df, SMA_PERIOD, name):
    print("getting sma")
    for i in df.columns:
        if i != "zeros":
            tempDf = pd.DataFrame(index=df.index)
            rollingAvg = df[i].rolling(SMA_PERIOD).mean()
            if i == name:
                tempDf[f"{i} {SMA_PERIOD}"] = rollingAvg
            tempDf[f"{i} difference"] = (df[i] - rollingAvg) * 100 / copy[i]

            df = pd.concat([df, tempDf], axis=1)

    df = df.dropna(axis=0, how="all")
    return df


@st.cache
def getDic(df):
    dic = {}
    for i in range(len(df.iloc[-1])):
        if "difference" in df.iloc[-1].index[i]:
            dic[df.iloc[-1].index[i]] = abs(df.iloc[-1][i])
    dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1])}
    return dic


def graph(ticker: str, days: int):
    if days > len(copy):
        raise ValueError("Days out of range")
    msa = f"{ticker} {SMA_PERIOD}"
    df = copy[[ticker, msa]].iloc[-1 * days:]
    return df


def graphDiff(ticker: str):
    df = copy[[f"{ticker} difference", "zeros"]]
    return df


t = datetime.datetime.now().date()
print(t)
tickerList, copy = run(t)

option = st.sidebar.selectbox('Ticker', sorted(tickerList))

SMA_PERIOD = st.slider("SMA PERIOD", 1, 100, 50)
copy = getDf(copy, SMA_PERIOD, option)
dic = getDic(copy)

top10diff = [f"{i} - {round(dic[i], 3)}%" for i in list(dic.keys())[-10:]]
low10diff = [f"{i} - {round(dic[i], 3)}%" for i in list(dic.keys())[:10]]
st.sidebar.subheader(f"Top 10 most volatile")
for i in top10diff:
    st.sidebar.text(i)
st.sidebar.subheader(f"10 least volatile")
for i in low10diff:
    st.sidebar.text(i)

dayRange = st.slider("Period", 1, len(copy) - 1, 75)
st.line_chart(graph(option, dayRange))
st.line_chart(graphDiff(option))

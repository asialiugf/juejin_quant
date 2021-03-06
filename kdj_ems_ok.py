#!/usr/bin/env python
# encoding: utf-8
import pandas as pd
import tushare as ts
import time
import datetime
import matplotlib.pyplot as plt
import pylab
from matplotlib.dates import MONDAY, SATURDAY
from matplotlib.dates import date2num
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from gmsdk import md
import tquant as tt
from tquant.Formula import MA
from tquant.Formula import KDJ
from tquant.Formula import EMA

# import matplotlib.dates as mdates
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, WeekdayLocator, DateFormatter

'''
ss = ts.get_hist_data('600848')#一次性获取全部日k线数据
print(ss)
'''


def _candlestick(ax, df, width=1000, colorup='k', colordown='r',
                 alpha=1.0):
    """
    Plot the time, open, high, low, close as a vertical line ranging
    from low to high.  Use a rectangular bar to represent the
    open-close span.  If close >= open, use colorup to color the bar,
    otherwise use colordown

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    df : pandas data from tushare
    width : float
        fraction of a day for the rectangle width
    colorup : color
        the color of the rectangle where close >= open
    colordown : color
         the color of the rectangle where close <  open
    alpha : float
        the rectangle alpha level
    ochl: bool
        argument to select between ochl and ohlc ordering of quotes

    Returns
    -------
    ret : tuple
        returns (lines, patches) where lines is a list of lines
        added and patches is a list of the rectangle patches added

    """

    OFFSET = width / 2.0

    lines = []
    patches = []
    t = 0
    for date_string, row in df.iterrows():
        # date_time = datetime.datetime.strptime(date_string,'%Y-%m-%d')
        # t = date2num(date_time)
        t = t + 10
        # open, high, low, close= row[:4]
        open = row[5]
        high = row[3]
        low = row[4]
        close = row[1]
        # print("open:",open)
        # print("high:",high)
        # print("low:",low)
        # print("close:",close)
        if close >= open:
            color = colorup
            lower = open
            height = close - open
        else:
            color = colordown
            lower = close
            height = open - close

        vline = Line2D(
            xdata=(t, t), ydata=(low, high),
            color=color,
            linewidth=0.5,
            antialiased=True,
        )

        rect = Rectangle(
            # xy=(t - OFFSET, lower),
            xy=(t - OFFSET, lower),
            width=width,
            height=height,
            # facecolor=color,
            edgecolor=color,
        )
        # rect.set_alpha(alpha)

        lines.append(vline)
        patches.append(rect)
        ax.add_line(vline)
        ax.add_patch(rect)
    ax.autoscale_view()

    return lines, patches


def drawPic(df, code, name):
    mondays = WeekdayLocator(MONDAY)  # 主要刻度
    alldays = DayLocator()  # 次要刻度
    # weekFormatter = DateFormatter('%b %d')     # 如：Jan 12
    mondayFormatter = DateFormatter('%m-%d-%Y')  # 如：2-29-2015
    dayFormatter = DateFormatter('%d')  # 如：12

    plt.figure(2)  # 创建图表2
    ax1 = plt.subplot(211)  # 在图表2中创建子图1
    ax2 = plt.subplot(212)  # 在图表2中创建子图2
    plt.sca(ax1)
    ax1.xaxis.set_major_locator(mondays)
    ax1.xaxis.set_minor_locator(alldays)
    ax1.xaxis.set_major_formatter(mondayFormatter)
    _candlestick(ax1, df, width=0.6, colorup='r', colordown='g')
    ax1.xaxis_date()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    # ax.grid(True)

    # print("kdj2222222222222222222222222222222222222222222:",kdj2)
    plt.sca(ax2)

    kdj = KDJ(df, 45, 15, 15)
    kdj2 = KDJ(df, 45 * 5, 15 * 5, 15 * 5)
    kdj3 = KDJ(df, 45 * 5 * 5, 15 * 5 * 5, 15 * 5 * 5)
    kdj4 = KDJ(df, 45 * 5 * 5 * 5, 15 * 5 * 5 * 5, 15 * 5 * 5 * 5)

    #JJ = pd.DataFrame({'KDJ_J': kdj['KDJ_J']})
    #JJ2 = pd.DataFrame({'KDJ_J': kdj2['KDJ_J']})
    #JJ3 = pd.DataFrame({'KDJ_J': kdj3['KDJ_J']})
    #JJ4 = pd.DataFrame({'KDJ_J': kdj4['KDJ_J']})

    JJ = kdj['KDJ_J']
    JJ2 = kdj2['KDJ_J']
    JJ3 = kdj3['KDJ_J']
    JJ4 = kdj4['KDJ_J']

    ema = EMA(JJ, 5)
    ema2 = EMA(JJ2, 5 * 5)
    ema3 = EMA(JJ3, 5 * 5 * 5)
    ema4 = EMA(JJ4, 5 * 5 * 5 * 5)

    print("111111111 df:")
    print(df)
    print("222222222 kdj:")
    print(kdj)
    print("3333333333 JJ:")
    print(JJ)
    print("4444444444 ema:")
    print(ema)

    # print('ema:',ema)
    # print("KDJ ==== ", kdj)
    xx = []
    t = 0
    for DAT, DD in kdj.iterrows():
        t = t + 10
        xx.append(t)

    # print(xx)
    plt.sca(ax2)

    plt.plot(xx, ema, linewidth=0.5)
    plt.plot(xx, kdj['KDJ_J'], linewidth=0.5)

    plt.plot(xx, ema2, linewidth=0.5)
    plt.plot(xx, kdj2['KDJ_J'], linewidth=0.5)

    plt.plot(xx, ema3, linewidth=0.5)
    plt.plot(xx, kdj3['KDJ_J'], linewidth=0.5)

    plt.plot(xx, ema4, linewidth=0.5)
    plt.plot(xx, kdj4['KDJ_J'], linewidth=0.5)

    print("ema44444444444444444444444444444444444:", ema4)
    print("len:of df:", len(df))

    plt.show()


def makePicture(code, name):
    # df = ts.get_hist_data(code, start=begin_time, end=end_time)
    # df = ts.get_hist_data('600848')
    ret = md.init("13601380996", "it@iZ23psatkqsZ")
    df = tt.get_bars("SHFE.ru1709", 300, "2017-04-05 09:00:00", "2017-06-06 09:30:00")
    # print("kkkkkkkkkkkkkkkkkkkkkkkkkkkk!!!")
    # print("df:",df)
    # df = df.sort_index(0)
    #    df.plot()

    C = df['close']  # 切片收盘价
    C30 = C.tail(30)  # 取最后30个收盘价数据
    MA30 = C30.mean()  # 均值
    # print("MA30:",MA30)
    # xx = MA(C,5)
    # print("MA30 ==== ",xx)
    # kdj = KDJ(df,9,3,3)
    # print("KDJ ==== ", kdj)

    drawPic(df, code, name)


if __name__ == '__main__':
    begin_time = "20170101"
    end_time = "20170501"
    makePicture("600001", "kkk")

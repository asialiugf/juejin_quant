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

import numpy as np
import seequant as sq

# import matplotlib.dates as mdates
from matplotlib.dates import YearLocator, MonthLocator, DayLocator, WeekdayLocator, DateFormatter

'''
ss = ts.get_hist_data('600848')#一次性获取全部日k线数据
print(ss)
'''


def get_data( begin_time, end_time, period ):
    """
    :param from_time:
    :param to_time:
    :return:
    """
    # df = ts.get_hist_data(code, start=begin_time, end=end_time)
    # df = ts.get_hist_data('600848')
    #ret = md.init( "13601380996", "it@iZ23psatkqsZ" )
    """
    下面的tt.get_bars() 可以不用 md.init();
    在myquant.py中，已经有  md.init();
    """
    # df4 = tt.get_bars( "SHFE.ru1709", 60, "2016-09-05 09:00:00", "2016-11-04 23:30:00" )
    # df5 = tt.get_bars( "SHFE.ru1709", 60, "2016-11-05 09:00:00", "2017-01-04 23:30:00" )
    # df6 = tt.get_bars( "SHFE.ru1709", 60, "2017-01-05 09:00:00", "2017-03-04 23:30:00" )
    # df7 = tt.get_bars( "SHFE.ru1709", 60, "2017-03-05 09:00:00", "2017-05-04 23:30:00" )
    # df8 = tt.get_bars( "SHFE.ru1709", 60, "2017-05-25 09:00:00", "2017-07-06 23:30:00" )
    df8 = tt.get_bars( "SHFE.ru1709", 5*60, "2017-02-25 09:00:00", "2017-07-06 23:30:00" )

    # df8 = tt.get_bars( "CZCE.CF709", 600, "2017-03-25 09:00:00", "2017-07-06 23:30:00" )
    #df8 = tt.get_bars( "CZCE.ZC709", 15*60, "2017-02-25 09:00:00", "2017-07-06 23:30:00" )
    #df8 = tt.get_bars( "CZCE.OI709", 3*60, "2017-02-25 09:00:00", "2017-07-06 23:30:00" )

    # df = pd.concat( [df4, df5, df6, df7, df8] )

    df = tt.get_last_n_dailybars( "CZCE.ZC709", 20000 )

    df = pd.concat( [df8] )
    print( len( df ) )
    return df


"""
trading_data只记录 buy, sell, close_buy, close_sell, 当时的值
"""
trading_data = {
    "init_cash": 1000000,
    "curr_cash": 1000000,
    "H": 0.8,                       # hua dian
    "B": 0,
    "S": 0,
    "OB": 0,                        # on buy
    "OS": 0,                        # on sell
    "OC": 1,                        # on close
    "R": 1,                         # rate  收益率
    "BR": 1,                        # base rate  计算下一波收益率的起始收益率 相当于追加
    "P": -10000,                    # base price

    "close_rate": 1                 # 平仓的比例
}


def my_buy( td, C ):
    print( "RRRR", td["R"] )
    close_S( td, C )
    td["B"] = 1
    td["S"] = 0
    td["OS"] = 0
    td["OC"] = 0
    if (td["OB"] == 0):
        td["P"] = C+td["H"]
        td["OB"] = 1
        print( "R1", td["R"] )
    elif (td["OB"] == 1):
        td["R"] = td["BR"]*(1+(C-td["P"])/td["P"])
        print( "R2", td["R"] )

    print( "rate: from my_buy: ", td["R"] )
    return


def my_sell( td, C ):
    print( "RRRR", td["R"] )
    close_B( td, C )
    td["B"] = 0
    td["S"] = 1
    td["OB"] = 0
    td["OC"] = 0
    if (td["OS"] == 0):
        td["P"] = C-td["H"]
        td["OS"] = 1
        print( "sell 1:", td["R"] )
    elif (td["OS"] == 1):
        td["R"] = td["BR"]*(1+(td["P"]-C)/td["P"])
        print( "sell 2:", td["R"] )
    print( "rate: from my_sell: ", td["R"] )
    return


def close_B( td, C ):
    if (td["B"] == 1 or td["OB"] == 1):
        td["R"] = td["BR"]*(1+(C-td["P"])/td["P"])
        td["BR"] = td["R"]
    td["B"] = 0
    td["OB"] = 0
    return


def close_S( td, C ):
    if (td["S"] == 1 or td["OS"] == 1):
        td["R"] = td["BR"]*(1+(td["P"]-C)/td["P"])
        td["BR"] = td["R"]
    td["S"] = 0
    td["OS"] = 0
    return


def strategy01( df ):
    # initial_cash = 1000000          # 初始化资金
    # transaction_ratio = 1           # 委托量成交比率，默认 = 1（每个委托100%成交）
    # commission_ratio = 0            # 手续费率，默认 = 0（不计算手续费）
    # slippage_ratio = 0              # 滑点比率，默认 = 0（无滑点）
    # price_type = 1                  # 行情复权模式, 0 = 不复权, 1 = 前复权

    buy = 0
    sell = 0
    on_buy = 0
    on_sell = 0
    on_close = 0

    df['base'] = np.nan
    df['R'] = np.nan
    df['BR'] = np.nan
    df['buy'] = np.nan
    df['sell'] = np.nan
    df['BP'] = np.nan

    x1 = df['diff'].values
    x2 = df['x8610'].values

    for i in range( 610, len( df ) ):
        print( "i-------------------------------:", i )
        if (x2[i]>0):
            my_buy( trading_data, df['close'].values[i] )
            df['buy'].iat[i] = 1
            df['R'].iat[i] = trading_data["R"]
            df['BR'].iat[i] = trading_data["BR"]

        if (x2[i]<0):
            my_sell( trading_data, df['close'].values[i] )
            # float(df['close'].values[i])
            df['buy'].iat[i] = -1
            df['R'].iat[i] = trading_data["R"]
            df['BR'].iat[i] = trading_data["BR"]

        print( "trading data:", trading_data["R"] )
        print( "df rate:", df['R'].values[i] )

        df['BP'].iat[i] = trading_data["P"]

    return df




def _candlestick( ax, df, width = 1000, colorup = 'k', colordown = 'r',
                  alpha = 1.0 ):
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

    OFFSET = width/2.0

    lines = []
    patches = []
    t = 0
    for date_string, row in df.iterrows():
        # date_time = datetime.datetime.strptime(date_string,'%Y-%m-%d')
        # t = date2num(date_time)
        t = t+10
        # open, high, low, close= row[:4]
        open = row[5]
        high = row[3]
        low = row[4]
        close = row[1]
        # print("open:",open)
        # print("high:",high)
        # print("low:",low)
        # print("close:",close)
        if close>=open:
            color = colorup
            lower = open
            height = close-open
        else:
            color = colordown
            lower = close
            height = open-close

        vline = Line2D(
            xdata = (t, t), ydata = (low, high),
            color = color,
            linewidth = 0.5,
            antialiased = True,
        )

        rect = Rectangle(
            # xy=(t - OFFSET, lower),
            xy = (t-OFFSET, lower),
            width = width,
            height = height,
            # facecolor=color,
            edgecolor = color,
        )
        # rect.set_alpha(alpha)

        lines.append( vline )
        patches.append( rect )
        ax.add_line( vline )
        ax.add_patch( rect )
    ax.autoscale_view()

    return lines, patches


def drawPic( df, code, name ):
    mondays = WeekdayLocator( MONDAY )  # 主要刻度
    alldays = DayLocator()  # 次要刻度
    # weekFormatter = DateFormatter('%b %d')     # 如：Jan 12
    mondayFormatter = DateFormatter( '%m-%d-%Y' )  # 如：2-29-2015
    dayFormatter = DateFormatter( '%d' )  # 如：12

    xx = []
    t = 0
    for DAT, DD in df.iterrows():
        t = t+1
        xx.append( t )

    # --------------------------------------------------------------------
    plt.figure( 4 )           # 创建图表2
    ax1 = plt.subplot( 411 )  # 在图表2中创建子图1
    ax2 = plt.subplot( 412 )  # 在图表2中创建子图2
    ax3 = plt.subplot( 413 )  # 在图表2中创建子图2
    ax4 = plt.subplot( 414 )  # 在图表2中创建子图2

    # --------------------------------------------(子图表1)
    plt.sca( ax1 )
    ax1.xaxis.set_major_locator( mondays )
    ax1.xaxis.set_minor_locator( alldays )
    ax1.xaxis.set_major_formatter( mondayFormatter )
    _candlestick( ax1, df, width = 0.6, colorup = 'r', colordown = 'g' )
    ax1.xaxis_date()
    plt.setp( plt.gca().get_xticklabels(), rotation = 45, horizontalalignment = 'right' )

    """
    plt.plot( xx, df['ma5'], linewidth = 0.5 )
    plt.plot( xx, df['ma10'], linewidth = 0.5 )
    plt.plot( xx, df['ma20'], linewidth = 0.5 )
    plt.plot( xx, df['ma30'], linewidth = 0.5 )
    plt.plot( xx, df['ma40'], linewidth = 0.5 )
    plt.plot( xx, df['ma60'], linewidth = 0.5 )
    plt.plot( xx, df['ma89'], linewidth = 0.5 )
    plt.plot( xx, df['ma144'], linewidth = 0.5 )
    plt.plot( xx, df['ma233'], linewidth = 0.5 )
    plt.plot( xx, df['ma377'], linewidth = 0.5 )
    plt.plot( xx, df['ma610'], linewidth = 0.5 )
    """
    # ax.grid(True)
    # --------------------------------------------(子图表2)
    plt.sca( ax2 )
    # plt.plot( xx[len(xx)-500:len(xx)], df['j1'].tail(500), linewidth = 0.5 )
    plt.plot( xx, df['j1'], linewidth = 0.5 )
    plt.plot( xx, df['e1'], linewidth = 0.5 )

    plt.plot( xx, df['j2'], linewidth = 0.5 )
    plt.plot( xx, df['e2'], linewidth = 0.5 )

    plt.plot( xx, df['j3'], linewidth = 0.5 )
    plt.plot( xx, df['e3'], linewidth = 0.5 )

    plt.plot( xx, df['j4'], linewidth = 0.5 )
    plt.plot( xx, df['e4'], linewidth = 0.5 )

    plt.sca( ax3 )
    # plt.plot( xx, df['x1'], linewidth = 0.5 )
    # plt.plot( xx, df['x2']*7, linewidth = 0.5 )
    # plt.plot( xx, df['x3']*30, linewidth = 0.5 )
    # plt.plot( xx, df['x4']*80, linewidth = 0.5 )
    plt.plot( xx, df['DIFF1'], linewidth = 0.5 )
    plt.plot( xx, df['DEA1'], linewidth = 0.5 )
    plt.plot( xx, df['DIFF2'], linewidth = 0.5 )
    plt.plot( xx, df['DEA2'], linewidth = 0.5 )
    plt.plot( xx, df['DIFF3'], linewidth = 0.5 )
    plt.plot( xx, df['DEA3'], linewidth = 0.5 )

    plt.sca( ax4 )
    # plt.plot( xx, df['gain'], linewidth = 0.5 )
    # plt.plot( xx, df['diff'], linewidth = 0.5 )
    # plt.plot( xx, df['x8610'], linewidth = 0.5 )
    print("00000000000000000000000000000000000000000000000000000000000",df['R'])
    plt.plot( xx, df['R'], linewidth = 0.5 )

    plt.show()


def makePicture( code, name ):
    df = get_data( "2016-09-05 09:00:00", "2016-11-04 23:30:00", 60 )
    print( "df len:", len( df ) )

    # aa = calc_kdj_ma_ema( df )
    aa = sq.calc_ma( df )
    aa = sq.calc_kdj( df )
    aa = sq.calc_macd( df )
    aa = strategy01( df )
    kk = df[['close', 'BP', 'BR', 'R', 'buy',
             'dea1pos','dea1upd',
             'dea2pos','dea2upd',
             'dea3pos','dea3upd',
             'dx1pos', 'dx1upd',
             'dx2pos', 'dx2upd',
             'dx3pos', 'dx3upd',
             'macd1pos','macd1upd',
             'macd2pos','macd2upd',
             'macd3pos','macd3upd']]
    kk.to_csv( "dddddd.csv" )
    print( df['R'] )

    # print( aa.head( 2 ) )
    # print( aa.tail( 5 ) )

    # print("kkkkkkkkkkkkkkkkkkkkkkkkkkkk!!!")
    # print("df:",df)
    # df = df.sort_index(0)
    # df.plot()

    # C = df['close']  # 切片收盘价
    # C30 = C.tail( 30 )  # 取最后30个收盘价数据
    # MA30 = C30.mean()  # 均值
    # print("MA30:",MA30)
    # xx = MA(C,5)
    # print("MA30 ==== ",xx)
    # kdj = KDJ(df,9,3,3)
    # print("KDJ ==== ", kdj)

    drawPic( df, code, name )

if __name__ == '__main__':
    begin_time = "20170101"
    end_time = "20170501"
    makePicture( "600001", "kkk" )

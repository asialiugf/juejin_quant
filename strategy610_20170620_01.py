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
from tquant.Formula import MACD
import numpy as np

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
    ret = md.init( "13601380996", "it@iZ23psatkqsZ" )
    # df4 = tt.get_bars( "SHFE.ru1709", 60, "2016-09-05 09:00:00", "2016-11-04 23:30:00" )
    # df5 = tt.get_bars( "SHFE.ru1709", 60, "2016-11-05 09:00:00", "2017-01-04 23:30:00" )
    # df6 = tt.get_bars( "SHFE.ru1709", 60, "2017-01-05 09:00:00", "2017-03-04 23:30:00" )
    # df7 = tt.get_bars( "SHFE.ru1709", 60, "2017-03-05 09:00:00", "2017-05-04 23:30:00" )
    # df8 = tt.get_bars( "SHFE.ru1709", 60, "2017-05-25 09:00:00", "2017-07-06 23:30:00" )
    # df8 = tt.get_bars( "CZCE.CF709", 600, "2017-03-25 09:00:00", "2017-07-06 23:30:00" )
    df8 = tt.get_bars( "CZCE.ZC709", 900, "2017-02-25 09:00:00", "2017-07-06 23:30:00" )
    # df = pd.concat( [df4, df5, df6, df7, df8] )
    df = pd.concat( [df8] )
    print( len( df ) )
    return df


"""
trading_data只记录 buy, sell, close_buy, close_sell, 当时的值
"""
trading_data = {
    "init_cash": 1000000,
    "curr_cash": 1000000,
    "huadian": 5,
    "B": 0,
    "S": 0,
    "OB": 0, # on buy
    "OS": 0, # on sell
    "OC": 1, # on close
    "R": 1, # rate
    "BR": 1, # base rate
    "P": -10000, # base price
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
        td["P"] = C
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
        td["P"] = C
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
    initial_cash = 1000000          # 初始化资金
    transaction_ratio = 1           # 委托量成交比率，默认 = 1（每个委托100%成交）
    commission_ratio = 0            # 手续费率，默认 = 0（不计算手续费）
    slippage_ratio = 0              # 滑点比率，默认 = 0（无滑点）
    price_type = 1                  # 行情复权模式, 0 = 不复权, 1 = 前复权

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

    x1 = df['diff'].values
    x2 = df['x8610'].values

    for i in range( 610, len( df ) ):
        print( "i-------------------------------:", i )
        if (x2[i]>0):
            my_buy( trading_data, df['close'].values[i] )
            df['buy'].iat[i] = 1
            df['R'].iat[i] = trading_data["R"]
            df['BR'].iat[i] = trading_data["BR"]
            print( "trading data:", trading_data["R"] )
            print( "df rate:", df['R'].values[i] )

        if (x2[i]<0):
            my_sell( trading_data, df['close'].values[i] )
            # float(df['close'].values[i])
            df['buy'].iat[i] = -1
            df['R'].iat[i] = trading_data["R"]
            df['BR'].iat[i] = trading_data["BR"]
            print( "trading data:", trading_data["R"] )
            print( "df rate:", df['R'].values[i] )

    return df


def calc_ma( df ):
    df['ma5'] = MA( df['close'], 5 )
    df['ma8'] = MA( df['close'], 8 )
    df['ma10'] = MA( df['close'], 10 )
    df['ma13'] = MA( df['close'], 13 )
    df['ma20'] = MA( df['close'], 20 )
    df['ma21'] = MA( df['close'], 21 )
    df['ma30'] = MA( df['close'], 30 )
    df['ma34'] = MA( df['close'], 34 )
    df['ma40'] = MA( df['close'], 40 )
    df['ma55'] = MA( df['close'], 55 )
    df['ma60'] = MA( df['close'], 60 )
    df['ma89'] = MA( df['close'], 89 )
    df['ma144'] = MA( df['close'], 144 )
    df['ma233'] = MA( df['close'], 233 )
    df['ma377'] = MA( df['close'], 377 )
    df['ma610'] = MA( df['close'], 610 )

    df['diff'] = abs( df['ma5']-df['ma8'] )+abs( df['ma5']-df['ma13'] ) \
                 +abs( df['ma5']-df['ma21'] )+abs( df['ma5']-df['ma34'] ) \
                 +abs( df['ma5']-df['ma55'] )+abs( df['ma5']-df['ma89'] ) \
                 +abs( df['ma5']-df['ma144'] )+abs( df['ma5']-df['ma233'] ) \
                 +abs( df['ma5']-df['ma377'] )+abs( df['ma5']-df['ma610'] ) \
                 +abs( df['ma8']-df['ma13'] ) \
                 +abs( df['ma8']-df['ma21'] )+abs( df['ma8']-df['ma34'] ) \
                 +abs( df['ma8']-df['ma55'] )+abs( df['ma8']-df['ma89'] ) \
                 +abs( df['ma8']-df['ma144'] )+abs( df['ma8']-df['ma233'] ) \
                 +abs( df['ma8']-df['ma377'] )+abs( df['ma8']-df['ma610'] ) \
                 +abs( df['ma13']-df['ma21'] )+abs( df['ma13']-df['ma34'] ) \
                 +abs( df['ma13']-df['ma55'] )+abs( df['ma13']-df['ma89'] ) \
                 +abs( df['ma13']-df['ma144'] )+abs( df['ma13']-df['ma233'] ) \
                 +abs( df['ma13']-df['ma377'] )+abs( df['ma13']-df['ma610'] ) \
                 +abs( df['ma21']-df['ma34'] ) \
                 +abs( df['ma21']-df['ma55'] )+abs( df['ma21']-df['ma89'] ) \
                 +abs( df['ma21']-df['ma144'] )+abs( df['ma21']-df['ma233'] ) \
                 +abs( df['ma21']-df['ma377'] )+abs( df['ma21']-df['ma610'] ) \
                 +abs( df['ma34']-df['ma55'] )+abs( df['ma34']-df['ma89'] ) \
                 +abs( df['ma34']-df['ma144'] )+abs( df['ma34']-df['ma233'] ) \
                 +abs( df['ma34']-df['ma377'] )+abs( df['ma34']-df['ma610'] ) \
                 +abs( df['ma55']-df['ma89'] ) \
                 +abs( df['ma55']-df['ma144'] )+abs( df['ma55']-df['ma233'] ) \
                 +abs( df['ma55']-df['ma377'] )+abs( df['ma55']-df['ma610'] ) \
                 +abs( df['ma89']-df['ma144'] )+abs( df['ma89']-df['ma233'] ) \
                 +abs( df['ma89']-df['ma377'] )+abs( df['ma89']-df['ma610'] ) \
                 +abs( df['ma144']-df['ma233'] ) \
                 +abs( df['ma144']-df['ma377'] )+abs( df['ma144']-df['ma610'] ) \
                 +abs( df['ma233']-df['ma377'] )+abs( df['ma233']-df['ma610'] ) \
                 +abs( df['ma377']-df['ma610'] )

    df['x8610'] = df['ma8']-df['ma610']
    return df


def calc_kdj( df ):
    kdj1 = KDJ( df, 45, 15, 15 )
    kdj2 = KDJ( df, 45*5, 15*5, 15*5 )
    kdj3 = KDJ( df, 45*5*5, 15*5*5, 15*5*5 )
    kdj4 = KDJ( df, 45*5*5*5, 15*5*5*5, 15*5*5*5 )

    df['k1'] = kdj1['KDJ_K']
    df['d1'] = kdj1['KDJ_D']
    df['j1'] = kdj1['KDJ_J']
    df['e1'] = EMA( kdj1['KDJ_J'], 5 )

    df['k2'] = kdj2['KDJ_K']
    df['d2'] = kdj2['KDJ_D']
    df['j2'] = kdj2['KDJ_J']
    df['e2'] = EMA( kdj2['KDJ_J'], 5*5 )

    df['k3'] = kdj3['KDJ_K']
    df['d3'] = kdj3['KDJ_D']
    df['j3'] = kdj3['KDJ_J']
    df['e3'] = EMA( kdj3['KDJ_J'], 5*5*5 )

    df['k4'] = kdj4['KDJ_K']
    df['d4'] = kdj4['KDJ_D']
    df['j4'] = kdj4['KDJ_J']
    df['e4'] = EMA( kdj4['KDJ_J'], 5*5*5*5 )

    # 计算KDJ值的斜率 ------------ begin -----------------------------------
    """
    df['x1'] = np.nan
    df['x2'] = np.nan
    df['x3'] = np.nan
    df['x4'] = np.nan
    n = 6
    x = np.arange( n )
    yy = df['e1'].values
    for i in range( n-1, len( df ) ):
        y1 = df['e1'][i-n+1:i+1].values
        y2 = df['e2'][i-n+1:i+1].values
        y3 = df['e3'][i-n+1:i+1].values
        y4 = df['e4'][i-n+1:i+1].values
        # print(x)
        # print(y)
        z1 = np.polyfit( x, y1, 3 )
        z2 = np.polyfit( x, y2, 3 )
        z3 = np.polyfit( x, y3, 3 )
        z4 = np.polyfit( x, y4, 3 )
        # print('zzzzzz',z)
        r1 = z1[0]*x[n-1]*x[n-1]+z1[1]*x[n-1]+z1[2]
        r2 = z2[0]*x[n-1]*x[n-1]+z2[1]*x[n-1]+z2[2]
        r3 = z3[0]*x[n-1]*x[n-1]+z3[1]*x[n-1]+z3[2]
        r4 = z4[0]*x[n-1]*x[n-1]+z4[1]*x[n-1]+z4[2]
        df['x1'].iat[i] = r1
        df['x2'].iat[i] = r2
        df['x3'].iat[i] = r3
        df['x4'].iat[i] = r4
    """
    # 计算KDJ值的斜率 ------------- end -----------------------------------
    return df


def calc_macd( df ):
    macd1 = MACD( df, 12, 26, 9 )
    macd2 = MACD( df, 12*5, 26*5, 9*5 )
    macd3 = MACD( df, 12*5*5, 26*5*5, 9*5*5 )
    macd4 = MACD( df, 12*5*5*5, 26*5*5*5, 9*5*5*5 )
    df['DIFF1'] = macd1['DIFF']
    df['DEA1'] = macd1['DEA']
    df['MACD1'] = macd1['MACD']

    df['DIFF2'] = macd2['DIFF']
    df['DEA2'] = macd2['DEA']
    df['MACD2'] = macd2['MACD']

    df['DIFF3'] = macd3['DIFF']
    df['DEA3'] = macd3['DEA']
    df['MACD3'] = macd3['MACD']

    df['DIFF4'] = macd4['DIFF']
    df['DEA4'] = macd4['DEA']
    df['MACD4'] = macd4['MACD']
    return df


def calc_kdj_ma_ema( df ):
    """
    # KDJ1
    df['k1'] = np.nan
    df['d1'] = np.nan
    df['j1'] = np.nan
    df['e1'] = np.nan
    df['x1'] = np.nan

    # KDJ2
    df['k2'] = np.nan
    df['d2'] = np.nan
    df['j2'] = np.nan
    df['e2'] = np.nan
    df['x2'] = np.nan

    # KDJ3
    df['k3'] = np.nan
    df['d3'] = np.nan
    df['j3'] = np.nan
    df['e3'] = np.nan
    df['x3'] = np.nan

    # KDJ4
    df['k4'] = np.nan
    df['d4'] = np.nan
    df['j4'] = np.nan
    df['e4'] = np.nan
    df['x4'] = np.nan

    # MA

    df['ma5'] = np.nan
    df['ma10'] = np.nan
    df['ma20'] = np.nan
    df['ma30'] = np.nan
    df['ma40'] = np.nan
    df['ma60'] = np.nan
    df['ma89'] = np.nan
    df['ma144'] = np.nan
    df['ma233'] = np.nan
    df['ma377'] = np.nan
    df['ma610'] = np.nan
    """
    # ------------------------------------------------------------
    df['ma5'] = MA( df['close'], 5 )
    df['ma8'] = MA( df['close'], 8 )
    df['ma10'] = MA( df['close'], 10 )
    df['ma13'] = MA( df['close'], 13 )
    df['ma20'] = MA( df['close'], 20 )
    df['ma21'] = MA( df['close'], 21 )
    df['ma30'] = MA( df['close'], 30 )
    df['ma34'] = MA( df['close'], 34 )
    df['ma40'] = MA( df['close'], 40 )
    df['ma55'] = MA( df['close'], 55 )
    df['ma60'] = MA( df['close'], 60 )
    df['ma89'] = MA( df['close'], 89 )
    df['ma144'] = MA( df['close'], 144 )
    df['ma233'] = MA( df['close'], 233 )
    df['ma377'] = MA( df['close'], 377 )
    df['ma610'] = MA( df['close'], 610 )

    df['diff'] = abs( df['ma5']-df['ma8'] )+abs( df['ma5']-df['ma13'] ) \
                 +abs( df['ma5']-df['ma21'] )+abs( df['ma5']-df['ma34'] ) \
                 +abs( df['ma5']-df['ma55'] )+abs( df['ma5']-df['ma89'] ) \
                 +abs( df['ma5']-df['ma144'] )+abs( df['ma5']-df['ma233'] ) \
                 +abs( df['ma5']-df['ma377'] )+abs( df['ma5']-df['ma610'] ) \
                 +abs( df['ma8']-df['ma13'] ) \
                 +abs( df['ma8']-df['ma21'] )+abs( df['ma8']-df['ma34'] ) \
                 +abs( df['ma8']-df['ma55'] )+abs( df['ma8']-df['ma89'] ) \
                 +abs( df['ma8']-df['ma144'] )+abs( df['ma8']-df['ma233'] ) \
                 +abs( df['ma8']-df['ma377'] )+abs( df['ma8']-df['ma610'] ) \
                 +abs( df['ma13']-df['ma21'] )+abs( df['ma13']-df['ma34'] ) \
                 +abs( df['ma13']-df['ma55'] )+abs( df['ma13']-df['ma89'] ) \
                 +abs( df['ma13']-df['ma144'] )+abs( df['ma13']-df['ma233'] ) \
                 +abs( df['ma13']-df['ma377'] )+abs( df['ma13']-df['ma610'] ) \
                 +abs( df['ma21']-df['ma34'] ) \
                 +abs( df['ma21']-df['ma55'] )+abs( df['ma21']-df['ma89'] ) \
                 +abs( df['ma21']-df['ma144'] )+abs( df['ma21']-df['ma233'] ) \
                 +abs( df['ma21']-df['ma377'] )+abs( df['ma21']-df['ma610'] ) \
                 +abs( df['ma34']-df['ma55'] )+abs( df['ma34']-df['ma89'] ) \
                 +abs( df['ma34']-df['ma144'] )+abs( df['ma34']-df['ma233'] ) \
                 +abs( df['ma34']-df['ma377'] )+abs( df['ma34']-df['ma610'] ) \
                 +abs( df['ma55']-df['ma89'] ) \
                 +abs( df['ma55']-df['ma144'] )+abs( df['ma55']-df['ma233'] ) \
                 +abs( df['ma55']-df['ma377'] )+abs( df['ma55']-df['ma610'] ) \
                 +abs( df['ma89']-df['ma144'] )+abs( df['ma89']-df['ma233'] ) \
                 +abs( df['ma89']-df['ma377'] )+abs( df['ma89']-df['ma610'] ) \
                 +abs( df['ma144']-df['ma233'] ) \
                 +abs( df['ma144']-df['ma377'] )+abs( df['ma144']-df['ma610'] ) \
                 +abs( df['ma233']-df['ma377'] )+abs( df['ma233']-df['ma610'] ) \
                 +abs( df['ma377']-df['ma610'] )

    df['x8610'] = df['ma8']-df['ma610']

    kdj1 = KDJ( df, 45, 15, 15 )
    kdj2 = KDJ( df, 45*5, 15*5, 15*5 )
    kdj3 = KDJ( df, 45*5*5, 15*5*5, 15*5*5 )
    kdj4 = KDJ( df, 45*5*5*5, 15*5*5*5, 15*5*5*5 )

    df['k1'] = kdj1['KDJ_K']
    df['d1'] = kdj1['KDJ_D']
    df['j1'] = kdj1['KDJ_J']
    df['e1'] = EMA( kdj1['KDJ_J'], 5 )

    df['k2'] = kdj2['KDJ_K']
    df['d2'] = kdj2['KDJ_D']
    df['j2'] = kdj2['KDJ_J']
    df['e2'] = EMA( kdj2['KDJ_J'], 5*5 )

    df['k3'] = kdj3['KDJ_K']
    df['d3'] = kdj3['KDJ_D']
    df['j3'] = kdj3['KDJ_J']
    df['e3'] = EMA( kdj3['KDJ_J'], 5*5*5 )

    df['k4'] = kdj4['KDJ_K']
    df['d4'] = kdj4['KDJ_D']
    df['j4'] = kdj4['KDJ_J']
    df['e4'] = EMA( kdj4['KDJ_J'], 5*5*5*5 )

    # 计算KDJ值的斜率 ------------ begin -----------------------------------
    """
    df['x1'] = np.nan
    df['x2'] = np.nan
    df['x3'] = np.nan
    df['x4'] = np.nan
    n = 6
    x = np.arange( n )
    yy = df['e1'].values
    for i in range( n-1, len( df ) ):
        y1 = df['e1'][i-n+1:i+1].values
        y2 = df['e2'][i-n+1:i+1].values
        y3 = df['e3'][i-n+1:i+1].values
        y4 = df['e4'][i-n+1:i+1].values
        # print(x)
        # print(y)
        z1 = np.polyfit( x, y1, 3 )
        z2 = np.polyfit( x, y2, 3 )
        z3 = np.polyfit( x, y3, 3 )
        z4 = np.polyfit( x, y4, 3 )
        # print('zzzzzz',z)
        r1 = z1[0]*x[n-1]*x[n-1]+z1[1]*x[n-1]+z1[2]
        r2 = z2[0]*x[n-1]*x[n-1]+z2[1]*x[n-1]+z2[2]
        r3 = z3[0]*x[n-1]*x[n-1]+z3[1]*x[n-1]+z3[2]
        r4 = z4[0]*x[n-1]*x[n-1]+z4[1]*x[n-1]+z4[2]
        df['x1'].iat[i] = r1
        df['x2'].iat[i] = r2
        df['x3'].iat[i] = r3
        df['x4'].iat[i] = r4
    """
    # 计算KDJ值的斜率 ------------- end -----------------------------------

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
        t = t+10
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
    plt.plot( xx, df['R'], linewidth = 0.5 )

    plt.show()


def makePicture( code, name ):
    df = get_data( "2016-09-05 09:00:00", "2016-11-04 23:30:00", 60 )
    print( "df len:", len( df ) )

    # aa = calc_kdj_ma_ema( df )
    aa = calc_ma( df )
    aa = calc_kdj( df )
    aa = calc_macd( df )
    aa = strategy01( df )
    kk = df[['close', 'BR', 'R', 'buy']]
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

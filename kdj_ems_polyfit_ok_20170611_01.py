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
    df8 = tt.get_bars( "SHFE.ru1709", 60, "2017-05-05 09:00:00", "2017-07-06 23:30:00" )

    # df = pd.concat( [df4, df5, df6, df7, df8] )
    df = pd.concat( [df8] )
    print( len( df ) )

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
    df['ma10'] = MA( df['close'], 10 )
    df['ma20'] = MA( df['close'], 20 )
    df['ma30'] = MA( df['close'], 30 )
    df['ma40'] = MA( df['close'], 40 )
    df['ma60'] = MA( df['close'], 60 )
    df['ma89'] = MA( df['close'], 89 )
    df['ma144'] = MA( df['close'], 144 )
    df['ma233'] = MA( df['close'], 233 )
    df['ma377'] = MA( df['close'], 377 )
    df['ma610'] = MA( df['close'], 610 )

    kdj1 = KDJ( df, 45, 15, 15 )
    kdj2 = KDJ( df, 45*5, 15*5, 15*5 )
    kdj3 = KDJ( df, 45*5*5, 15*5*5, 15*5*5 )
    kdj4 = KDJ( df, 45*5*5*5, 15*5*5*5, 15*5*5*5 )
    """
    #ema1 = EMA( kdj1['KDJ_J'], 5 )
    #ema2 = EMA( kdj2['KDJ_J'], 5*5 )
    #ema3 = EMA( kdj3['KDJ_J'], 5*5*5 )
    #ema4 = EMA( kdj4['KDJ_J'], 5*5*5*5 )
    """

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

    df['x1'] = np.nan
    df['x2'] = np.nan
    df['x3'] = np.nan
    df['x4'] = np.nan

    """
    for i in range( 0, n ):
        df['x1'].iat[i] = 0
        df['x2'].iat[i] = 0
        df['x3'].iat[i] = 0
        df['x4'].iat[i] = 0
    """


    n = 6
    x = np.arange( n )

    yy = df['e1'].values
    for i in range( n-1, len( df ) ):
        y1 = df['e1'][i-n+1:i+1].values
        y2 = df['e2'][i-n+1:i+1].values
        y3 = df['e3'][i-n+1:i+1].values
        y4 = df['e4'][i-n+1:i+1].values
        #print(x)
        #print(y)
        z1 = np.polyfit( x, y1, 3 )
        z2 = np.polyfit( x, y2, 3 )
        z3 = np.polyfit( x, y3, 3 )
        z4 = np.polyfit( x, y4, 3 )

        #print('zzzzzz',z)

        r1 = z1[0]*x[n-1]*x[n-1] + z1[1]*x[n-1] + z1[2]
        r2 = z2[0]*x[n-1]*x[n-1] + z2[1]*x[n-1] + z2[2]
        r3 = z3[0]*x[n-1]*x[n-1] + z3[1]*x[n-1] + z3[2]
        r4 = z4[0]*x[n-1]*x[n-1] + z4[1]*x[n-1] + z4[2]

        df['x1'].iat[i] = r1
        df['x2'].iat[i] = r2
        df['x3'].iat[i] = r3
        df['x4'].iat[i] = r4

        #print('nnnnnn:',i,r1)

    print(df[['x1','x2','x3','x4']])



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
    plt.figure( 3 )           # 创建图表2
    ax1 = plt.subplot( 311 )  # 在图表2中创建子图1
    ax2 = plt.subplot( 312 )  # 在图表2中创建子图2
    ax3 = plt.subplot( 313 )  # 在图表2中创建子图2

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
    # print("kdj2222222222222222222222222222222222222222222:",kdj2)

    # --------------------------------------------(子图表2)
    plt.sca( ax2 )

    plt.plot( xx, df['j2'], linewidth = 0.5 )
    plt.plot( xx, df['e2'], linewidth = 0.5 )

    plt.plot( xx, df['j3'], linewidth = 0.5 )
    plt.plot( xx, df['e3'], linewidth = 0.5 )


    plt.sca( ax3 )
    plt.plot( xx, df['x3']*3, linewidth = 0.5 )
    plt.plot( xx, df['x2'], linewidth = 0.5 )

    """
    plt.plot( xx, ema, linewidth = 0.5 )
    plt.plot( xx, kdj['KDJ_J'], linewidth = 0.5 )

    plt.plot( xx, ema2, linewidth = 0.5 )
    plt.plot( xx, kdj2['KDJ_J'], linewidth = 0.5 )

    plt.plot( xx, ema3, linewidth = 0.5 )
    plt.plot( xx, kdj3['KDJ_J'], linewidth = 0.5 )

    plt.plot( xx, ema4, linewidth = 0.5 )
    plt.plot( xx, kdj4['KDJ_J'], linewidth = 0.5 )

    print( "ema44444444444444444444444444444444444:", ema4 )
    print( "len:of df:", len( df ) )
    
    """
    plt.show()


def makePicture( code, name ):
    df = get_data( "2016-09-05 09:00:00", "2016-11-04 23:30:00", 60 )
    print( "df len:", len( df ) )

    aa = calc_kdj_ma_ema( df )

    #print( aa.head( 2 ) )
    #print( aa.tail( 5 ) )

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

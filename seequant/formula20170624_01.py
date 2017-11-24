# -*- coding:utf-8 -*-
"""
交易数据接口
Created on 2014/07/31
@author: Asialine Liu
@group : seequant
@contact: asialiugf@sina.com
"""
from __future__ import division
import numpy as np
import tquant as tt
from tquant.Formula import MA
from tquant.Formula import KDJ
from tquant.Formula import EMA
from tquant.Formula import MACD


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

    # df['x8610'] = df['ma5']-df['ma610']
    df['x8610'] = df['close']-df['ma20']
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

    # df['DIFF4'] = macd4['DIFF']
    # df['DEA4'] = macd4['DEA']
    # df['MACD4'] = macd4['MACD']

    DIFF1 = df['DIFF1'].values
    DEA1 = df['DEA1'].values
    MACD1 = df['MACD1'].values

    DIFF2 = df['DIFF2'].values
    DEA2 = df['DEA2'].values
    MACD2 = df['MACD2'].values

    DIFF3 = df['DIFF3'].values
    DEA3 = df['DEA3'].values
    MACD3 = df['MACD3'].values

    # DIFF3 = df['DIFF4'].values
    # DEA3 = df['DEA4'].values
    # MACD3 = df['MACD4'].values

    df['dea1pos'] = np.nan
    df['dea1upd'] = np.nan
    df['dea2pos'] = np.nan
    df['dea2upd'] = np.nan
    df['dea3pos'] = np.nan
    df['dea3upd'] = np.nan
    # df['dea4pos'] = np.nan
    # df['dea4upd'] = np.nan

    df['macd1pos'] = np.nan
    df['macd1upd'] = np.nan
    df['macd2pos'] = np.nan
    df['macd2upd'] = np.nan
    df['macd3pos'] = np.nan
    df['macd3upd'] = np.nan
    # df['macd4pos'] = np.nan
    # df['macd4upd'] = np.nan

    df['dx1pos'] = np.nan
    df['dx1upd'] = np.nan
    df['dx2pos'] = np.nan
    df['dx2upd'] = np.nan
    df['dx3pos'] = np.nan
    df['dx3upd'] = np.nan
    # df['dx4pos'] = np.nan
    # df['dx4upd'] = np.nan


    df['dx1pos'].iat[0] = 0
    df['dx1upd'].iat[0] = 1
    df['macd1pos'].iat[0] = 0
    df['macd1upd'].iat[0] = 1

    df['dx2pos'].iat[0] = 0
    df['dx2upd'].iat[0] = 1
    df['macd2pos'].iat[0] = 0
    df['macd2upd'].iat[0] = 1

    df['dx3pos'].iat[0] = 0
    df['dx3upd'].iat[0] = 1
    df['macd3pos'].iat[0] = 0
    df['macd3upd'].iat[0] = 1

    # df['dx4pos'].iat[0] = 0
    # df['dx4upd'].iat[0] = 1
    # df['macd4pos'].iat[0] = 0
    # df['macd4upd'].iat[0] = 1

    df['dx1pos'].iat[0] = 0
    df['dx1upd'].iat[0] = 1
    df['dx2pos'].iat[0] = 0
    df['dx2upd'].iat[0] = 1
    df['dx3pos'].iat[0] = 0
    df['dx3upd'].iat[0] = 1
    # df['dx4pos'].iat[0] = 0
    # df['dx4upd'].iat[0] = 1

    ddxx1 = np.nan
    ddxx0 = np.nan
    for i in range( 1, len( df ) ):
        df['dx1pos'].iat[i] = i
        df['dx1upd'].iat[i] = 0
        df['macd1pos'].iat[i] = i
        df['macd1upd'].iat[i] = 0

        df['dx2pos'].iat[i] = i
        df['dx2upd'].iat[i] = 0
        df['macd2pos'].iat[i] = i
        df['macd2upd'].iat[i] = 0

        df['dx3pos'].iat[i] = i
        df['dx3upd'].iat[i] = 0
        df['macd3pos'].iat[i] = i
        df['macd3upd'].iat[i] = 0

        # df['dx4pos'].iat[i] = i
        # df['dx4upd'].iat[i] = 0
        # df['macd4pos'].iat[i] = i
        # df['macd4upd'].iat[i] = 0

        df['dx1pos'].iat[i] = i
        df['dx1upd'].iat[i] = 0
        df['dx2pos'].iat[i] = i
        df['dx2upd'].iat[i] = 0
        df['dx3pos'].iat[i] = i
        df['dx3upd'].iat[i] = 0
        # df['dx4pos'].iat[i] = i
        # df['dx4upd'].iat[i] = 0

        # 1111111111111111111111111111111111111111111111111111111111111111111
        if (DEA1[i]>0):
            df['dx1upd'].iat[i] = 1
            if (DEA1[i-1]>0):
                df['dx1pos'].iat[i] = df['dx1pos'].iat[i-1]
            elif (DEA1[i-1]<0):
                df['dx1pos'].iat[i] = i
        elif (DEA1[i]<0):
            df['dx1upd'].iat[i] = -1
            if (DEA1[i-1]>0):
                df['dx1pos'].iat[i] = i
            elif (DEA1[i-1]<0):
                df['dx1pos'].iat[i] = df['dx1pos'].iat[i-1]

        if (MACD1[i]>0):
            df['macd1upd'].iat[i] = 1
            if (MACD1[i-1]>0):
                df['macd1pos'].iat[i] = df['macd1pos'].iat[i-1]
            elif (MACD1[i-1]<0):
                df['macd1pos'].iat[i] = i
        elif (MACD1[i]<0):
            df['macd1upd'].iat[i] = -1
            if (MACD1[i-1]>0):
                df['macd1pos'].iat[i] = i
            elif (MACD1[i-1]<0):
                df['macd1pos'].iat[i] = df['macd1pos'].iat[i-1]

        ddxx1 = DIFF1[i]-DIFF1[i-1]
        ddxx0 = DIFF1[i-1]-DIFF1[i-2]
        if (ddxx1>0):
            df['dx1upd'].iat[i] = 1
            if (ddxx0>0):
                df['dx1pos'].iat[i] = df['dx1pos'].iat[i-1]
            elif (ddxx0<0):
                df['dx1pos'].iat[i] = i
        elif (ddxx1<0):
            df['dx1upd'].iat[i] = -1
            if (ddxx0>0):
                df['dx1pos'].iat[i] = i
            elif (ddxx0<0):
                df['dx1pos'].iat[i] = df['dx1pos'].iat[i-1]

        # 22222222222222222222222222222222222222222222222222222222222222222222
        if (DEA2[i]>0):
            df['dea2upd'].iat[i] = 1
            if (DEA2[i-1]>0):
                df['dea2pos'].iat[i] = df['dea2pos'].iat[i-1]
            elif (DEA2[i-1]<0):
                df['dea2pos'].iat[i] = i
        elif (DEA2[i]<0):
            df['dea2upd'].iat[i] = -1
            if (DEA2[i-1]>0):
                df['dea2pos'].iat[i] = i
            elif (DEA2[i-1]<0):
                df['dea2pos'].iat[i] = df['dea2pos'].iat[i-1]

        if (MACD2[i]>0):
            df['macd2upd'].iat[i] = 1
            if (MACD2[i-1]>0):
                df['macd2pos'].iat[i] = df['macd2pos'].iat[i-1]
            elif (MACD2[i-1]<0):
                df['macd2pos'].iat[i] = i
        elif (MACD2[i]<0):
            df['macd2upd'].iat[i] = -1
            if (MACD2[i-1]>0):
                df['macd2pos'].iat[i] = i
            elif (MACD2[i-1]<0):
                df['macd2pos'].iat[i] = df['macd2pos'].iat[i-1]

        ddxx1 = DIFF2[i]-DIFF2[i-1]
        ddxx0 = DIFF2[i-1]-DIFF2[i-2]
        if (ddxx1>0):
            df['dx2upd'].iat[i] = 1
            if (ddxx0>0):
                df['dx2pos'].iat[i] = df['dx2pos'].iat[i-1]
            elif (ddxx0<0):
                df['dx2pos'].iat[i] = i
        elif (ddxx1<0):
            df['dx2upd'].iat[i] = -1
            if (ddxx0>0):
                df['dx2pos'].iat[i] = i
            elif (ddxx0<0):
                df['dx2pos'].iat[i] = df['dx2pos'].iat[i-1]

        # 333333333333333333333333333333333333333333333333333333333333333333333
        if (DEA3[i]>0):
            df['dea3upd'].iat[i] = 1
            if (DEA3[i-1]>0):
                df['dea3pos'].iat[i] = df['dea3pos'].iat[i-1]
            elif (DEA3[i-1]<0):
                df['dea3pos'].iat[i] = i
        elif (DEA3[i]<0):
            df['dea3upd'].iat[i] = -1
            if (DEA3[i-1]>0):
                df['dea3pos'].iat[i] = i
            elif (DEA3[i-1]<0):
                df['dea3pos'].iat[i] = df['dea3pos'].iat[i-1]

        if (MACD3[i]>0):
            df['macd3upd'].iat[i] = 1
            if (MACD3[i-1]>0):
                df['macd3pos'].iat[i] = df['macd3pos'].iat[i-1]
            elif (MACD3[i-1]<0):
                df['macd3pos'].iat[i] = i
        elif (MACD3[i]<0):
            df['macd3upd'].iat[i] = -1
            if (MACD3[i-1]>0):
                df['macd3pos'].iat[i] = i
            elif (MACD3[i-1]<0):
                df['macd3pos'].iat[i] = df['macd3pos'].iat[i-1]

        ddxx1 = DIFF3[i]-DIFF3[i-1]
        ddxx0 = DIFF3[i-1]-DIFF3[i-2]
        if (ddxx1>0):
            df['dx3upd'].iat[i] = 1
            if (ddxx0>0):
                df['dx3pos'].iat[i] = df['dx3pos'].iat[i-1]
            elif (ddxx0<0):
                df['dx3pos'].iat[i] = i
        elif (ddxx1<0):
            df['dx3upd'].iat[i] = -1
            if (ddxx0>0):
                df['dx3pos'].iat[i] = i
            elif (ddxx0<0):
                df['dx3pos'].iat[i] = df['dx3pos'].iat[i-1]

        """ 
        # 44444444444444444444444444444444444444444444444444444444444444444444
        if (DEA4[i]>0):
            df['dea4upd'].iat[i] = 1
            if(DEA4[i-1]>0):
                df['dea4pos'].iat[i] = df['dea4pos'].iat[i-1]
            elif(DEA4[i-1]<0):
                df['dea4pos'].iat[i] = i
        elif(DEA4[i]<0):
            df['dea4upd'].iat[i] = -1
            if (DEA4[i-1]>0):
                df['dea4pos'].iat[i] = i
            elif (DE4[i-1]<0):
                df['dea4pos'].iat[i] = df['dea4pos'].iat[i-1]
            
        if (MACD4[i]>0):
            df['macd4upd'].iat[i] = 1
            if(MACD4[i-1]>0):
                df['macd4pos'].iat[i] = df['macd4pos'].iat[i-1]
            elif(MACD4[i-1]<0):
                df['macd4pos'].iat[i] = i
        elif(MACD4[i]<0):
            df['macd4upd'].iat[i] = -1
            if (MACD4[i-1]>0):
                df['macd4pos'].iat[i] = i
            elif (DE4[i-1]<0):
                df['macd4pos'].iat[i] = df['macd4pos'].iat[i-1]
                    
        ddxx1 = DIFF4[i]-DIFF4[i-1]
        ddxx0 = DIFF4[i-1]-DIFF4[i-2]
        if ddxx1>0):
            df['dx4upd'].iat[i] = 1
            if (ddxx0>0):
                df['dx4pos'].iat[i] = df['dx4pos'].iat[i-1]
            elif (ddxx0<0):
                df['dx4pos'].iat[i] = i
        elif (ddxx1<0):
            df['dx4upd'].iat[i] = -1
            if (ddxx0>0):
                df['dx4pos'].iat[i] = i
            elif (ddxx0<0):
                df['dx4pos'].iat[i] = df['dx4pos'].iat[i-1]
                    
        """
        # ===================== end ============================================

    return df

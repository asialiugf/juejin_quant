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

'''
计算 斐波那契数列
当x为14时，输出
[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
'''


def _fibs( x ):
    fibs = [1, 2]
    for i in range( x-2 ):
        fibs.append( fibs[-2]+fibs[-1] )
    return fibs


def calc_ma( df ):
    ar = [5, 8, 10, 13, 20, 21, 30, 34, 40, 55, 60, 89, 120, 144, 233, 250, 377, 500, 610]
    for i in ar:
        s = str( i )
        s = "ma"+s
        df[s] = MA( df['close'], i )                    # 计算MA(5) MA(13) ... MA(610)

    fibs = _fibs( 14 )

    sum = abs( df['ma5']-df['ma8'] )                    # 需要优化！！
    for i in range( 3, 13 ):
        for j in range( i+1, 14 ):
            ma_b = "ma"+str( fibs[i] )
            ma_e = "ma"+str( fibs[j] )
            sum = sum+abs( df[ma_b]-df[ma_e] )
    df['diff'] = sum-abs( df['ma5']-df['ma8'] )         # 需要优化！！

    df['x8610'] = df['close']-df['ma20']

    print( 'yyyyyyyyyyyy', df['diff'] )

    return df


def calc_ema( df ):
    ar = [5, 8, 10, 13, 20, 21, 30, 34, 40, 55, 60, 89, 120, 144, 233, 250, 377, 500, 610]
    for i in ar:
        s = str( i )
        s = "ema"+s
        df[s] = EMA( df['close'], i )                    # 计算EMA(5) EMA(13) ... EMA(610)

    df['xema610'] = df['ma8']-df['ma610']
    df['xema377'] = df['ma8']-df['ma377']
    df['xema233'] = df['ma8']-df['ma233']
    df['xema144'] = df['ma8']-df['ma144']
    df['xema89'] = df['ma8']-df['ma89']
    df['xema55'] = df['ma8']-df['ma55']
    df['xema34'] = df['ma8']-df['ma34']
    df['xema21'] = df['ma8']-df['ma21']

    return df


def calc_kdj( df, x ):
    loc1 = 5
    loc2 = 20
    loc3 = 50
    loc4 = 80
    loc5 = 95

    N1 = -1
    N2 = -1
    N3 = -1
    var = 4
    if (x == 1):
        N1 = 9*var
        N2 = 3*var
        N3 = 3*var
        N4 = 5
        k = 'k1'
        d = 'd1'
        j = 'j1'
        e = 'e1'                # E 值： j值 的ema
        jpos = 'jpos1'          # J 值拐点位置
        jupd = 'jupd1'          # J 值 上升或下降
        jepos = 'jepos1'        # J 值 穿越 E值 的位置
        jeupd = 'jeupd1'        # J 值 在 E值 的上还是下
        jloc = 'jloc1'          # J 值 的位置 1 ( <loc1 )  2:( loc1<x<loc2 ) 3:< loc2<x<loc3 ) 4: ......

    if (x == 2):
        N1 = 9*var*var
        N2 = 3*var*var
        N3 = 3*var*var
        N4 = 5*var
        k = 'k2'
        d = 'd2'
        j = 'j2'
        e = 'e2'                # E 值： j值 的ema
        jpos = 'jpos2'          # J 值拐点位置
        jupd = 'jupd2'          # J 值 上升或下降
        jepos = 'jepos2'        # J 值 穿越 E值 的位置
        jeupd = 'jeupd2'        # J 值 在 E值 的上还是下
        jloc = 'jloc2'          # J 值 的位置 1 ( <loc1 )  2:( loc1<x<loc2 ) 3:< loc2<x<loc3 ) 4: ......

    if (x == 3):
        N1 = 9*var*var*var
        N2 = 3*var*var*var
        N3 = 3*var*var*var
        N4 = 5*var*var
        k = 'k3'
        d = 'd3'
        j = 'j3'
        e = 'e3'                # E 值： j值 的ema
        jpos = 'jpos3'          # J 值拐点位置
        jupd = 'jupd3'          # J 值 上升或下降
        jepos = 'jepos3'        # J 值 穿越 E值 的位置
        jeupd = 'jeupd3'        # J 值 在 E值 的上还是下
        jloc = 'jloc3'          # J 值 的位置 1 ( <loc1 )  2:( loc1<x<loc2 ) 3:< loc2<x<loc3 ) 4: ......

    if (x == 4):
        N1 = 9*var*var*var*var
        N2 = 3*var*var*var*var
        N3 = 3*var*var*var*var
        N4 = 5*var*var*var*var
        k = 'k4'
        d = 'd4'
        j = 'j4'
        e = 'e4'                # E 值： j值 的ema
        jpos = 'jpos4'          # J 值拐点位置
        jupd = 'jupd4'          # J 值 上升或下降
        jepos = 'jepos4'        # J 值 穿越 E值 的位置
        jeupd = 'jeupd4'        # J 值 在 E值 的上还是下
        jloc = 'jloc4'          # J 值 的位置 1 ( <loc1 )  2:( loc1<x<loc2 ) 3:< loc2<x<loc3 ) 4: ......

    rt = KDJ( df, N1, N2, N3 )
    df[k] = rt['KDJ_K']
    df[d] = rt['KDJ_D']
    df[j] = rt['KDJ_J']
    df[e] = EMA( rt['KDJ_J'], N4 )

    jj = df[j].values
    ee = df[e].values

    df[jpos] = np.nan
    df[jupd] = np.nan
    df[jepos] = np.nan
    df[jeupd] = np.nan
    df[jloc] = np.nan

    df[jpos].iat[0] = 0
    df[jupd].iat[0] = 1
    df[jepos].iat[0] = 0
    df[jeupd].iat[0] = 1

    ddxx1 = np.nan
    ddxx0 = np.nan

    for i in range( 2, len( df ) ):
        df[jpos].iat[i] = 0
        df[jupd].iat[i] = 1
        df[jepos].iat[i] = 0
        df[jeupd].iat[i] = 1

        # ======================= j value location from 0 to 110? ==========
        if (jj[i]<=loc1):
            df[jloc].iat[i] = 1
        elif (loc1<jj[i] and jj[i]<=loc2):
            df[jloc].iat[i] = 2
        elif (loc2<jj[i] and jj[i]<=loc3):
            df[jloc].iat[i] = 3
        elif (loc3<jj[i] and jj[i]<=loc4):
            df[jloc].iat[i] = 4
        elif (loc4<jj[i] and jj[i]<=loc5):
            df[jloc].iat[i] = 5
        elif (loc5<jj[i]):
            df[jloc].iat[i] = 6

        # ======================= j value > j(-1) value ? =================
        cha1 = jj[i]-jj[i-1]
        cha0 = jj[i-1]-jj[i-2]
        if (cha1>0):
            df[jupd].iat[i] = 1
            if (cha0>0):
                df[jpos].iat[i] = df[jpos].iat[i-1]
            elif (cha0<0):
                df[jpos].iat[i] = i
        elif (cha1<0):
            df[jupd].iat[i] = -1
            if (cha0>0):
                df[jpos].iat[i] = i
            elif (cha0<0):
                df[jpos].iat[i] = df[jpos].iat[i-1]

        # ======================= j value > e value ? =================
        cha1 = jj[i]-ee[i-1]
        cha0 = jj[i-1]-ee[i-2]
        if (cha1>0):
            df[jeupd].iat[i] = 1
            if (cha0>0):
                df[jepos].iat[i] = df[jepos].iat[i-1]
            elif (cha0<0):
                df[jepos].iat[i] = i
        elif (cha1<0):
            df[jeupd].iat[i] = -1
            if (cha0>0):
                df[jepos].iat[i] = i
            elif (cha0<0):
                df[jepos].iat[i] = df[jepos].iat[i-1]
    return df


def calc_kdj1( df ):
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


def calc_macd( df, x ):
    N1 = -1
    N2 = -1
    N3 = -1
    var = 4

    if (x == 1):
        N1 = 12
        N2 = 26
        N3 = 9
        diff = 'diff1'
        dea = 'dea1'
        macd = 'macd1'
        deapos = 'deapos1'
        deaupd = 'deaupd1'
        macdpos = 'macdpos1'
        macdupd = 'macdupd1'
        dxpos = 'dxpos1'
        dxupd = 'dxupd1'
        d0pos = 'd0pos1'
        d0upd = 'd0upd1'

    if (x == 2):
        N1 = 12*var
        N2 = 26*var
        N3 = 3*var
        diff = 'diff2'
        dea = 'dea2'
        macd = 'macd2'
        deapos = 'deapos2'
        deaupd = 'deaupd2'
        macdpos = 'macdpos2'
        macdupd = 'macdupd2'
        dxpos = 'dxpos2'
        dxupd = 'dxupd2'
        d0pos = 'd0pos2'
        d0upd = 'd0upd2'

    if (x == 3):
        N1 = 12*var*var
        N2 = 26*var*var
        N3 = 3*var*var
        diff = 'diff3'
        dea = 'dea3'
        macd = 'macd3'
        deapos = 'deapos3'
        deaupd = 'deaupd3'
        macdpos = 'macdpos3'
        macdupd = 'macdupd3'
        dxpos = 'dxpos3'
        dxupd = 'dxupd3'
        d0pos = 'd0pos3'
        d0upd = 'd0upd3'

    if (x == 4):
        N1 = 12*var*var*var
        N2 = 26*var*var*var
        N3 = 3*var*var*var
        diff = 'diff4'
        dea = 'dea4'
        macd = 'macd4'
        deapos = 'deapos4'
        deaupd = 'deaupd4'
        macdpos = 'macdpos4'
        macdupd = 'macdupd4'
        dxpos = 'dxpos4'
        dxupd = 'dxupd4'
        d0pos = 'd0pos4'
        d0upd = 'd0upd4'

    if (x == 5):
        N1 = 12
        N2 = 26
        N3 = 9
        diff = 'diff5'
        dea = 'dea5'
        macd = 'macd5'

        deapos = 'deapos5'
        deaupd = 'deaupd5'
        macdpos = 'macdpos5'
        macdupd = 'macdupd5'
        dxpos = 'dxpos5'
        dxupd = 'dxupd5'
        d0pos = 'd0pos5'
        d0upd = 'd0upd5'

    rt = MACD( df, N1, N2, N3 )
    df[diff] = rt['DIFF']
    df[dea] = rt['DEA']
    df[macd] = rt['MACD']
    # print(df[dea])

    diffs = df[diff].values
    deas = df[dea].values
    macds = df[macd].values

    df[deapos] = np.nan          # dea 穿越0轴
    df[deaupd] = np.nan          # dea 穿越0轴
    df[macdpos] = np.nan       # macd 穿越0轴
    df[macdupd] = np.nan       # macd 穿越0轴   MACD 0轴上或下，表示diff穿越dea，可用来表示DEA拐点
    df[dxpos] = np.nan        # diff值 拐点位置
    df[dxupd] = np.nan        # diff值 上升或下降
    df[d0pos] = np.nan        # diff值 穿越 0轴
    df[d0upd] = np.nan        # diff值 穿越 0轴

    df[deapos].iat[0] = 0
    df[deaupd].iat[0] = 1
    df[macdpos].iat[0] = 0
    df[macdupd].iat[0] = 1
    df[dxpos].iat[0] = 0
    df[dxupd].iat[0] = 1
    df[d0pos].iat[0] = 0
    df[d0upd].iat[0] = 1

    ddxx1 = np.nan
    ddxx0 = np.nan
    for i in range( 2, len( df ) ):
        df[deapos].iat[i] = i
        df[deaupd].iat[i] = 0
        df[macdpos].iat[i] = i
        df[macdupd].iat[i] = 0
        df[dxpos].iat[i] = i
        df[dxupd].iat[i] = 0
        df[d0pos].iat[i] = 0
        df[d0upd].iat[i] = 1

        # ======================= diff值 穿越 0轴 ========================================
        if (diffs[i]>0):
            df[d0upd].iat[i] = 1
            if (diffs[i-1]>0):
                df[d0pos].iat[i] = df[d0pos].iat[i-1]
            elif (diffs[i-1]<0):
                df[d0pos].iat[i] = i
        elif (diffs[i]<0):
            df[d0upd].iat[i] = -1
            if (diffs[i-1]>0):
                df[d0pos].iat[i] = i
            elif (diffs[i-1]<0):
                df[d0pos].iat[i] = df[d0pos].iat[i-1]

        # ======================= diff值 穿越 0轴========================================
        if (deas[i]>0):
            df[deaupd].iat[i] = 1
            if (deas[i-1]>0):
                df[deapos].iat[i] = df[deapos].iat[i-1]
            elif (deas[i-1]<0):
                df[deapos].iat[i] = i
        elif (deas[i]<0):
            df[deaupd].iat[i] = -1
            if (deas[i-1]>0):
                df[deapos].iat[i] = i
            elif (deas[i-1]<0):
                df[deapos].iat[i] = df[deapos].iat[i-1]

        # ======================= diff值 穿越 0轴 ========================================
        if (macds[i]>0):
            df[macdupd].iat[i] = 1
            if (macds[i-1]>0):
                df[macdpos].iat[i] = df[macdpos].iat[i-1]
            elif (macds[i-1]<0):
                df[macdpos].iat[i] = i
        elif (macds[i]<0):
            df[macdupd].iat[i] = -1
            if (macds[i-1]>0):
                df[macdpos].iat[i] = i
            elif (macds[i-1]<0):
                df[macdpos].iat[i] = df[macdpos].iat[i-1]

        # ======================= dx || diff - REF(diff,1) || position and up down =================
        ddxx1 = diffs[i]-diffs[i-1]
        ddxx0 = diffs[i-1]-diffs[i-2]
        if (ddxx1>0):
            df[dxupd].iat[i] = 1
            if (ddxx0>0):
                df[dxpos].iat[i] = df[dxpos].iat[i-1]
            elif (ddxx0<0):
                df[dxpos].iat[i] = i
        elif (ddxx1<0):
            df[dxupd].iat[i] = -1
            if (ddxx0>0):
                df[dxpos].iat[i] = i
            elif (ddxx0<0):
                df[dxpos].iat[i] = df[dxpos].iat[i-1]

                # ===================== end ===============================================================

    return df


def calc_macd1( df ):
    var1 = 4
    macd1 = MACD( df, 12, 26, 9 )
    macd2 = MACD( df, 12*var1, 26*var1, 3*var1 )
    macd3 = MACD( df, 12*var1*var1, 26*var1*var1, 3*var1*var1 )
    macd4 = MACD( df, 12*var1*var1*var1, 26*var1*var1*var1, 3*var1*var1*var1 )
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

    DIFF1 = df['DIFF1'].values
    DEA1 = df['DEA1'].values
    MACD1 = df['MACD1'].values

    DIFF2 = df['DIFF2'].values
    DEA2 = df['DEA2'].values
    MACD2 = df['MACD2'].values

    DIFF3 = df['DIFF3'].values
    DEA3 = df['DEA3'].values
    MACD3 = df['MACD3'].values

    DIFF4 = df['DIFF4'].values
    DEA4 = df['DEA4'].values
    MACD4 = df['MACD4'].values

    df['dea1pos'] = np.nan          # dea穿越0轴
    df['dea1upd'] = np.nan          # dea在0轴之上还是下
    df['dea2pos'] = np.nan
    df['dea2upd'] = np.nan
    df['dea3pos'] = np.nan
    df['dea3upd'] = np.nan
    df['dea4pos'] = np.nan
    df['dea4upd'] = np.nan

    df['macd1pos'] = np.nan       # macd 0轴上下改变的位置
    df['macd1upd'] = np.nan       # macd 0轴上或下   MACD 0轴上或下，表示diff穿越dea，可用来表示DEA拐点
    df['macd2pos'] = np.nan
    df['macd2upd'] = np.nan
    df['macd3pos'] = np.nan
    df['macd3upd'] = np.nan
    # df['macd4pos'] = np.nan
    # df['macd4upd'] = np.nan

    df['dx1pos'] = np.nan        # diff值 拐点位置
    df['dx1upd'] = np.nan        # diff值 上升或下降
    df['dx2pos'] = np.nan
    df['dx2upd'] = np.nan
    df['dx3pos'] = np.nan
    df['dx3upd'] = np.nan
    # df['dx4pos'] = np.nan
    # df['dx4upd'] = np.nan


    df['macd1pos'].iat[0] = 0
    df['macd1upd'].iat[0] = 1

    df['macd2pos'].iat[0] = 0
    df['macd2upd'].iat[0] = 1

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

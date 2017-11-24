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


def strategy01( df ):
    df['stt01_bs'] = np.nan

    jupd1 = df['jupd1'].values
    jupd2 = df['jupd2'].values
    jupd3 = df['jupd3'].values
    jupd4 = df['jupd4'].values

    jeupd1 = df['jeupd1'].values
    jeupd2 = df['jeupd2'].values
    jeupd3 = df['jeupd3'].values
    jeupd4 = df['jeupd4'].values

    for i in range( 610, len( df ) ):
        if (jupd4[i]>0 and jupd3[i]>0):             # 4上 3上
            if (jupd2[i]>0):
                df['stt01_bs'].iat[i] = 1
            elif (jupd2[i]<0):
                df['stt01_bs'].iat[i] = 0

        elif (jupd4[i]<0 and jupd3[i]<0):           # 4下 3下
            if (jupd2[i]>0):
                df['stt01_bs'].iat[i] = 0
            elif (jupd2[i]<0):
                df['stt01_bs'].iat[i] = -1

        # elif (jupd4[i]>0 and jupd3[i]<0):           # 4上 3下
        else:
            """
            if(jupd1[i]>0 and jupd2[i]>0):
                df['stt01_bs'].iat[i] = 1
            elif(jupd1[i]<0 and jupd2[i]<0):
                df['stt01_bs'].iat[i] = -1
            else:
                df['stt01_bs'].iat[i] = 0
            """
            df['stt01_bs'].iat[i] = 0

        """
        elif(jupd4[i]<0 and jupd3[i]>0):            # 4下 3上
            pass
        """
    return df


def strategy02( df ):
    df['stt02_bs'] = np.nan

    diff1 = df['diff1'].values
    diff2 = df['diff2'].values
    diff3 = df['diff3'].values
    diff4 = df['diff4'].values

    d0upd1 = df['d0upd1'].values
    d0upd2 = df['d0upd2'].values
    d0upd3 = df['d0upd3'].values
    d0upd4 = df['d0upd4'].values

    dxupd1 = df['dxupd1'].values
    dxupd2 = df['dxupd2'].values
    dxupd3 = df['dxupd3'].values
    dxupd4 = df['dxupd4'].values

    for i in range( 610, len( df ) ):
        if (dxupd4[i]>0 and dxupd3[i]>0 and d0upd2[i]>0):             # 4上 3上
            df['stt02_bs'].iat[i] = 1


        elif (dxupd4[i]<0 and dxupd3[i]<0 and d0upd2[i]<0):           # 4下 3下
            df['stt02_bs'].iat[i] = -1

        else:
            df['stt02_bs'].iat[i] = 0

    return df

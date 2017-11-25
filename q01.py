# -*- coding: utf-8 -*-

from gmsdk.api import StrategyBase
import numpy as np
import pandas as pd

df = pd.DataFrame( columns = ['strtime','strendtime', 'open', 'close'] )


class Mystrategy( StrategyBase ):
    def __init__( self, *args, **kwargs ):
        super( Mystrategy, self ).__init__( *args, **kwargs )

    def on_login( self ):
        print( 'on_login' )
        pass

    def on_error( self, code, msg ):
        pass

    def on_tick( self, tick ):
        # print('on_tick')
        # print(tick.open)

        pass

    def on_bar( self, bar ):
        global df
        # print('on_bar')
        print( bar.open, bar.close, bar.strtime, bar.strendtime )
        dd = bar.strtime[0:10]
        tt = bar.strtime[11:19]
        mm = bar.strendtime[0:10]
        nn = bar.strendtime[11:19]

        df.loc[len(df)] = { 'strtime': dd+' '+tt,
                            'strendtime': mm+' '+nn ,
                            'open': bar.open,
                            'close': bar.close }

        print(df.tail(3))
        print(' ')

        # print(type(bar[0]))
        pass

    def on_execrpt( self, res ):
        pass

    def on_order_status( self, order ):
        pass

    def on_order_new( self, res ):
        pass

    def on_order_filled( self, res ):
        pass

    def on_order_partiall_filled( self, res ):
        pass

    def on_order_stop_executed( self, res ):
        pass

    def on_order_canceled( self, res ):
        pass

    def on_order_cancel_rejected( self, res ):
        pass


if __name__ == '__main__':
    myStrategy = Mystrategy(
        username = 'asialiugf@sina.com',
        password = 'it@iZ23psatkqsZ',
        strategy_id = 'a3a17499-d18c-11e7-aefa-68f7283cd5ae',
        subscribe_symbols = 'SHFE.ru1801.tick,SHFE.ru1801.bar.15,SHFE.ru1801.bar.5',
        mode = 4,
        td_addr = '127.0.0.1:8001'
    )
    myStrategy.backtest_config(
        start_time = '2017-11-15 9:00:00',
        end_time = '2017-11-16 15:00:00',
        initial_cash = 1000000,
        transaction_ratio = 1,
        commission_ratio = 0,
        slippage_ratio = 0,
        price_type = 1,
        bench_symbol = 'SHSE.000300' )
    print( 'test!' )
    ret = myStrategy.run()
    print( 'exit code: ', ret )

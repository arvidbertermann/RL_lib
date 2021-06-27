from numba.typed import List
#import dqr.lobtester.numbed_data_provider as numbed_data_provider

import pandas as pd
import qpython
import qpython.qconnection
import pickle
import dill as pickle
import numpy as np
import cloudpickle
import pyspark.serializers
pyspark.serializers.cloudpickle = cloudpickle
import openfigi
import datetime as dt

class KDBDataProvider():
    def __init__(self, path, port, sym, date, endDate = None, use_bbo=True):
        self.sym = sym
        self.date = date
        q = qpython.qconnection.QConnection('localhost', port, pandas=True)
        q.open()
        q("\\l " + path)
        
        if use_bbo:
            if endDate == None:
                endDate = date
            kdb_df = pd.DataFrame(q("select from mkt_orders where date within (" + date + ";" + endDate + "), sym=`" + sym + ", lob_actual=1")) # TODO concat
#             kdb_df = pd.DataFrame(q("select from mkt_orders where date=" + date + ", sym=`" + sym + ", lob_actual=1"))
            kdb_df = kdb_df.astype({'sym': 'string'})
            kdb_df['px'] = np.where(kdb_df['qtype'] == 1, kdb_df['px'], 0)
            kdb_df['qty'] = np.where(kdb_df['qtype'] == 1, kdb_df['qty'], 0)
            kdb_df['trade_dir'] = np.where(kdb_df['qtype'] == 1, kdb_df['trade_dir'], 0)
        else:
            kdb_df = pd.DataFrame(q("select from mkt_orders where date=" + date + ", sym=`" + sym + ", lob_actual=1"))
            kdb_df = kdb_df[kdb_df.qtype == 1]
        self.data = kdb_df
    
    def get_time(self):
        return self.data.exch_time.astype(np.int64).values
    
    def get_trade_price(self):
        return self.data.px.values

    def get_trade_qty(self):
        return self.data.qty.values * self.data.trade_dir.values
    
    def get_ask_price(self, lvl):
        return self.data['a_px_' + str(lvl)].values
    
    def get_bid_price(self, lvl):
        return self.data['b_px_' + str(lvl)].values

    def get_ask_qty(self, lvl):
        return self.data['a_qty_' + str(lvl)].values
    
    def get_bid_qty(self, lvl):
        return self.data['b_qty_' + str(lvl)].values
    
    def get_qtype(self):
        return self.data['qtype'].values

    def get_data(self):
        return self.data

    def get_sym(self):
        return self.sym

    def get_date(self):
        return self.date
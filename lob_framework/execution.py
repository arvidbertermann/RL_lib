import datetime as dt

class MPExecution():

    def __init__(self, lob, delay=0):
        lob = lob[lob['qtype'] == 1]
        self.lob = lob
        self.execution = "Mid Price"
        self.delay = dt.timedelta(microseconds=delay)
        self.output_dict = dict(mid_price=[], ex_price=[], time=[])
        self.cash = 0
        self.pnl = 0
        self.pnls = []

    def execute_order(self, time, exp):
        obs = self.lob[self.lob["exch_time"] == time].iloc[-1,:]
        mid = obs["a_px_0"] + obs["b_px_0"]
        temp = self.lob[self.lob["exch_time"] > time + self.delay]
        next_obs = temp.iloc[0,:]
        next_mid = next_obs["a_px_0"] + next_obs["b_px_0"]
        reward = (next_mid - mid)/2
        if exp == "b":
            return reward, mid, next_mid
        return -1*reward, mid, next_mid

    def update_pnl(self, state, action, x_1, x_2):
        if action == "s":
            self.cash += 2 * x_1
            self.pnl = self.cash - x_2
        elif action == "b":
            self.cash -= 2 * x_1
            self.pnl = self.cash + x_2
        else:
            if int(state) < 0:
                self.pnl = self.pnl - x_2 + x_1
            else:
                self.pnl = self.pnl + x_2 - x_1

    def first_pnl(self, exp, x_1, x_2):
        if exp == "s":
            self.cash += x_1
            #self.pnl = self.cash - x_2
        else:
            self.cash -= x_1
            #self.pnl = self.cash + x_2

    def append_pnl(self, exp, x_2):
        #if exp == "b":
        #    self.pnl += x_2
        #else:
        #    self.pnl -= x_2
        self.pnls.append(self.pnl)
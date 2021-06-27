
class MPExecution():

    def __init__(self, lob, delay=None):
        self.lob = lob
        self.execution = "Mid Price"
        self.delay = delay
        self.output_dict = dict(mid_price=[], ex_price=[], time=[])

    def execute_order(self, time, action):
        return 0


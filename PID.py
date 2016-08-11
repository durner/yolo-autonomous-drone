class PID(object):
    def __init__(self, K_p=0.4, K_d=0.00, K_i=0.00, dt=0.5):
        self.K_p = K_p
        self.K_d = K_d
        self.K_i = K_i
        self.dt = dt
        self.w = 0
        self.velocity = 0
        self.errorsum = 0
        self.actual_previous = 0

    def step(self, desired, actual):
        self.errorsum = (desired - actual) * self.dt
        self.velocity = (actual - self.actual_previous) / self.dt
        u = self.K_p * (desired - actual) + self.K_d * (self.w - self.velocity) + self.K_i * self.errorsum
        self.actual_previous = actual
        return u

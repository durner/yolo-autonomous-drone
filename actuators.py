import time
from libardrone import libardrone

from PID import PID

class Actuator(object):
    def __init__(self, drone, picture_width, desired_move):
        self.turn = PID(K_p=0.6, K_d=0.1)
        self.move = PID(K_p=0.15, K_d=0.01)
        self.height = PID(K_p=0.2, K_d=0.00)
        self.picture_width = picture_width
        self.desired_move = desired_move
        self.drone = drone
        time.sleep(0.05)
        self.drone.takeoff()
        time.sleep(0.05)

    def step(self, wdithmid, width):
        desired_turn = self.picture_width / 2
        actual_turn = wdithmid
        actual_move = width

        ut = self.turn.step(desired_turn, actual_turn)

        um = self.move.step(self.desired_move, actual_move)

        height = 550
        nav_data = self.drone.get_navdata()
        nav_data = nav_data[0]
        uh = self.height.step(height, nav_data['altitude'])

        self.drone.at(libardrone.at_pcmd, True, 0, self.moveDrone(um), self.heightDrone(uh), self.turnDrone(ut))

    def turnDrone(self, u):
        speed = - u / (self.picture_width / 2.)
        print "move horizontal to" + str(speed)
        return speed

    def moveDrone(self, u):
        speed = - u / (self.picture_width / 2.)
        print "move near to" + str(speed)
        return speed

    def heightDrone(self, u):
        speed = u / 500
        print "height near to" + str(speed)
        return speed
from libardrone import libardrone

from PID import PID

class Actuator(object):
    def __init__(self, drone, picture_width, desired_move):
        self.turn = PID(K_p=0.4)
        self.move = PID(K_p=0.3)
        self.picture_width = picture_width
        self.desired_move = desired_move
        self.drone = drone

    def step(self, wdithmid, width):
        desired_turn = self.picture_width / 2
        actual_turn = wdithmid
        actual_move = width
        ut = self.turn.step(desired_turn, actual_turn)
        um = self.move.step(self.desired_move, actual_move)
        self.drone.at(libardrone.at_pcmd, True, 0, self.moveDrone(um), 0, self.turnDrone(ut))

    def turnDrone(self, u):
        speed = - u / (self.picture_width / 2.)
        print "move horizontal to" + str(speed)
        return speed

    def moveDrone(self, u):
        speed = - u / self.picture_width
        print "move near to" + str(speed)
        return speed


class VerticalActuator(object):
    def __init__(self, drone):
        self.drone = drone
        self.drone.takeoff()
        print "drone takeoff"
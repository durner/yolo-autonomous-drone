import libardrone.libardrone as libardrone
import getch
import thread
import cv2

def main():
	drone = libardrone.ARDrone(True)
	drone.reset()
	
	try:
	   thread.start_new_thread( getKeyInput, (drone, ) )
	   thread.start_new_thread( getVideoImage, (drone, ) )
	except:
	   print "Error: unable to start thread"

	while not stop:
		pass

	print("Shutting down...")
	cv2.destroyAllWindows()
	drone.halt()
	print("Ok.")

def getKeyInput(drone):
	_Getch = getch.Getch()
	global stop
	while True:
		key = _Getch.impl() 
		if key == "t":		drone.takeoff()
		elif key == " ":	drone.land()
		elif key == "0":	drone.hover()
		elif key == "w":	drone.move_forward()
		elif key == "s":	drone.move_backward()
		elif key == "a":	drone.move_left()
		elif key == "d":	drone.move_right()
		elif key == "q":	drone.turn_left()
		elif key == "e":	drone.turn_right()
		elif key == "8":	drone.move_up()
		elif key == "2":	drone.move_down()
		elif key == "c":	stop = True
	

def getVideoImage(drone):
	while True:
		try:
			# print pygame.image
			pixelarray = drone.get_image()
			if pixelarray != None:
				cv2.imshow('frame', pixelarray)
				cv2.waitKey(1)
		except:
			pass

if __name__ == '__main__':
	stop = False
	main()

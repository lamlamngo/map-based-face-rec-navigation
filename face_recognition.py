#!/usr/bin/env python
from __future__ import print_function
import roslib
#roslib.load_manifest('my_package')
import rospy,math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion
# from sound_play.msg import SoundRequest
# from sound_play.libsoundplay import SoundClient
import cv2, sys, numpy, os
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist

#Written by Lam Ngo
#Adapted from https://github.com/aquibjaved/Real-time-face-recognition-in-python-using-opencv-

class face_regconition:

    def __init__(self):
        self.image_pub = rospy.Publisher("face_regconition",Image)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw",Image,self.callback)
        #self.path = os.path.join('/home/generic/catkin_ws/src/image_test/src/datasets', '/home/generic/catkin_ws/src/image_test/src/lam')
        self.size = 4
        self.datasets = "/home/generic/picture/"
        self.width = 130
        self.height = 100
        (self.images, self.lables, self.names, self.id) = ([], [], {}, 0)
        for (subdirs, dirs, files) in os.walk(self.datasets):
            for subdir in dirs:
                self.names[self.id] = subdir
                subjectpath = os.path.join(self.datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    lable = self.id
                    self.images.append(cv2.imread(path, 0))
                    self.lables.append(int(lable))
                self.id += 1
        (self.images, self.lables) = [numpy.array(lis) for lis in [self.images, self.lables]]
        self.model = cv2.createFisherFaceRecognizer()
        self.model.train(self.images, self.lables)
        self.faces = None
        self.gray = None
        self.goal_sent = False
        self.x = 0
        self.y = 0

    def callback(self,data):
        rospy.loginfo("we do go in here")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('/home/generic/catkin_ws/src/image_test/src/haarcascade_frontalface_default.xml')
        self.faces = face_cascade.detectMultiScale(self.gray,1.3,4)
        return self.faces

    def facerec(self):
        if self.faces != None:
            for (x,y,w,h) in self.faces:
                face = self.gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face,(self.width,self.height))
                prediction = self.model.predict(face_resize)
                if prediction[1]<500:
                    rospy.loginfo('I see you')
                    rospy.loginfo('%s - %.0f' % (self.names[prediction[0]],prediction[1]))
                else:
                    rospy.loginfo('i do not recognize you')
        else:
            rospy.loginfo('I do not see anyone')
        self.faces = None
    def goto(self, pos, quat):

        # Send a goal
        self.goal_sent = True
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = 'map'
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose = Pose(Point(pos['x'], pos['y'], 0.000),
                                     Quaternion(quat['r1'], quat['r2'], quat['r3'], quat['r4']))

	# Start moving
        self.move_base.send_goal(goal)

	# Allow TurtleBot up to 60 seconds to complete task
        success = self.move_base.wait_for_result(rospy.Duration(60))

        state = self.move_base.get_state()
        result = False

        if success and state == GoalStatus.SUCCEEDED:
            # We made it!
            result = True
        else:
            self.move_base.cancel_goal()

        self.goal_sent = False
        return result
    # rotate function
    def rotate(self, relative_angle, isClockwise):
#        say("i am fucking rotating")
    	# publish to the right topic
    	pub = rospy.Publisher('/mobile_base/commands/velocity',Twist,queue_size=10)

    	#creates a blank twist message
    	outData = Twist()

    	#get initial time
    	t0 = rospy.get_rostime().secs
    	while t0 == 0:
    		t0 = rospy.get_rostime().secs
    	#initialize the current angle to be 0
    	current_angle = 0

    	#Set rospy rate to be 10hz
    	rate = rospy.Rate(10)

    	# speed is 10 degrees per second.
    	speed = self.degrees2radians(30)

    	if isClockwise:
    		outData.angular.z = -abs(speed)
    	else:
    		outData.angular.z = abs(speed)

    	while current_angle < relative_angle:
    		pub.publish(outData)
    		current_angle = speed * (rospy.get_rostime().secs-t0)
    		rate.sleep()

    	print ("relative angle: ", relative_angle)
    	print ("current angle: ", current_angle)

    def shutdown(self):
        if self.goal_sent:
            self.move_base.cancel_goal()
            rospy.loginfo("Mission failed")
        else:
            rospy.loginfo("Mission complete")
            rospy.sleep(1)

    def degrees2radians(self, angle):
    	return angle * (math.pi/180.0)

    # def say(whattosay):
    #     voice = 'voice_kal_diphone'
    #     volume = 1.0
    #     s = whattosay
    #     rospy.loginfo(whattosay)
    #     soundhandle = SoundClient()
    #     rospy.sleep(1)
    #     soundhandle.say(s, voice)
    #     rospy.sleep(1)

def main(args):
    ic = face_regconition()
    rospy.init_node('face_regconitionfuckyeah', anonymous=True)
    while (1):
        ic.facerec()
        ic.rotate(ic.degrees2radians(95),False)
        rospy.sleep(3.)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

import argparse
import struct
import sys
import copy
import rospy

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

import baxter_interface


class baxter(object):
	def __init__(self,hover_distance = 0.1):
		self.ns = "ExternalTools/right/PositionKinematicsNode/IKService"
		self._iksvc = rospy.ServiceProxy(self.ns,SolvePositionIK)
		self._ikreq = SolvePositionIKRequest()
		self._limb = baxter_interface.Limb('right')
		self._gripper = baxter_interface.Gripper('right')
		self.hover_distance = hover_distance
		self.safety_distance = 0.02
		self._rs = baxter_interface.RobotEnable(baxter_interface.CHECK_VERSION)
		self._init_state = self._rs.state().enabled
		self._verbose = True
		print("Enabling robot... ")
		self._rs.enable()
		self.wait = 3.0
		rospy.wait_for_service(self.ns,5.0)


	def gripper_open(self):
		print("open the gripper")
		self._gripper.open()
		rospy.sleep(1.0)

	def gripper_close(self):
		print("close the gripper")
		self._gripper.close()
		rospy.sleep(1.0)

	def ik_request(self,pose):
	    print(pose)
	    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
	    self._ikreq.pose_stamp.append(PoseStamped(header=hdr, pose=pose))
	    try:
	        resp = self._iksvc(self._ikreq)
	    except (rospy.ServiceException, rospy.ROSException), e:
	        print("s")
	        rospy.logerr("Service call failed: %s" % (e,))
	        return False
	    # Check if result valid, and type of seed ultimately used to get solution
	    # convert rospy's string representation of uint8[]'s to int's
	    resp_seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
	    limb_joints = {}
	    if (resp_seeds[0] != resp.RESULT_INVALID):
	        # seed_str = {
	        #             self._ikreq.SEED_USER: 'User Provided Seed',
	        #             self._ikreq.SEED_CURRENT: 'Current Joint Angles',
	        #             self._ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
	        #            }.get(resp_seeds[0], 'None')
	        if self._verbose:
	            print("IK Solution SUCCESS - Valid Joint Solution Found from Seed Type: {0}")

	        # Format solution into Limb API-compatible dictionary
	        limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
	        if self._verbose:
	            print("IK Joint Solution:\n{0}".format(limb_joints))
	            print("------------------")
	    else:
	    	print("no valid joint")
	        rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
	        return False
	    return limb_joints

	def move_to(self,pose):
		self._iksvc = rospy.ServiceProxy(self.ns,SolvePositionIK)
		self._ikreq = SolvePositionIKRequest()
		limb_joints = self.ik_request(pose)

		if limb_joints == False : 
			return False
		else : 
			print('Start Moving to the Pose')
			self._limb.move_to_joint_positions(limb_joints,timeout=10.0)
			return True
		



	# def _retract(self):
 #        # retrieve current pose from endpoint
 #        current_pose = self._limb.endpoint_pose()
 #        ik_pose = Pose()
 #        ik_pose.position.x = current_pose['position'].x
 #        ik_pose.position.y = current_pose['position'].y
 #        ik_pose.position.z = current_pose['position'].z + self._hover_distance
 #        ik_pose.orientation.x = current_pose['orientation'].x
 #        ik_pose.orientation.y = current_pose['orientation'].y
 #        ik_pose.orientation.z = current_pose['orientation'].z
 #        ik_pose.orientation.w = current_pose['orientation'].w
 #        joint_angles = self.ik_request(ik_pose)
 #        # servo up from current pose
 #        self._guarded_move_to_joint_position(joint_angles)

	def pick(self,pose):
		print('='*50)
		print('Start picking up')
		print("1. approach ")
		self.gripper_open()
		rospy.sleep(1.0)
		up = copy.deepcopy(pose)
		up.position.z = up.position.z + self.hover_distance
		print(up)
		move_return = self.move_to(up)
		if move_return == False : return False
		rospy.sleep(self.wait)
		print("2. hands down")
		print(pose)
		move_return = self.move_to(pose)
		f = self._limb.endpoint_effort()['force'].z
		if f <= -5 :
			print('move up slightly')
			pose.position.z += self.safety_distance
			move_return = self.move_to(pose)
			rospy.sleep(self.wait)
		rospy.sleep(self.wait)
		if move_return == False : return False
		
		print("3. grab")
		self.gripper_close()
		rospy.sleep(1.0)
		print("4. hands up")
		self.move_to(up)
		rospy.sleep(1.0)
		print('='*50)
		return True

	def place(self,pose):
		print('='*50)
		print('Start Place Down')
		print("1. approach ")
		#self.gripper_open()
		up = copy.deepcopy(pose)
		up.position.z = up.position.z + self.hover_distance
		move_return = self.move_to(up)
		rospy.sleep(self.wait)
		if move_return == False : return False
		print("2. hands down")
		self.move_to(pose)
		for i in range(6):
			f = self._limb.endpoint_effort()['force'].z
			if f <= -5 :
				print('move up slightly')
				pose.position.z += self.safety_distance
				self.move_to(pose)
				rospy.sleep(self.wait)
		rospy.sleep(self.wait)
		print("3. let it go")
		self.gripper_open()
		print("4. hands up")
		self.move_to(up)
		print('='*50)
		return True

from baxter import baxter
import rospy
import tensorflow as tf
import numpy as np
import cv2
import os

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

# EXAMPLE
# from PickandPlace import PnP
# pnp = PnP(rs,pipeline) # realsense , pipeline
# pnp.pick_and_place([start_pix_x, start_pix_y, 0, goal_pix_x, goal_pix_y, 0],depth_image)

class PnP(object):
	def __init__(self,rs, pipeline):

		self.retry = 6
		self.init_baxter()
		self.rs = rs
		self.pipeline = pipeline


	def pick_and_place(self,input_arr,depth_image):

		kernel = np.ones((10,10),np.float32)/100
		depth_image = cv2.filter2D(depth_image,-1,kernel)

		# get color intrinsics
		frames = self.pipeline.wait_for_frames()
		color_frame = frames.get_color_frame()
		intrin = color_frame.profile.as_video_stream_profile().intrinsics

		start_pix_x, start_pix_y, pick_ori, goal_pix_x, goal_pix_y, place_ori = input_arr

		# get start pose 
		depth_pixel = [555 - int(start_pix_x * 380/256.), 420 - int(start_pix_y * 380/256.)] # for resized/cropped image
		depth = depth_image[start_pix_y,start_pix_x] * 0.001
		#depth_pixel = [start_pix_x, start_pix_y] # if you use raw frame from realsense
		#depth = np.array(aligned_depth_frame.data)[depth_pixel[1], depth_pixel[0]] * 0.001 #self._depth_scale
		point = self.rs.rs2_deproject_pixel_to_point(intrin, depth_pixel, depth)

		pose_x, pose_y, pose_z = point
		print(point)
		position = Point(x = 0.675 - pose_y, y= - 0.3 - pose_x + 0.03, z= 0.49 - pose_z - 0.025)
		orientation = Quaternion(x=0.0,y=1.0,z=0.0,w=0.0)
		start_pose = Pose(position=position, orientation=orientation)
		pick_return = robot.pick(start_pose)

		depth_pixel = [555 - int(goal_pix_x * 380/256.), 420 - int(goal_pix_y * 380/256.)]
		#depth_pixel = [goal_pix_x, goal_pix_y]
		#depth = np.array(aligned_depth_frame.data)[depth_pixel[1], depth_pixel[0]] * 0.001 #self._depth_scale
		depth = depth_image[start_pix_y,start_pix_x] * 0.001
		point = self.rs.rs2_deproject_pixel_to_point(intrin, depth_pixel, depth)

		pose_x, pose_y, pose_z = point
		print(point)
		position = Point(x = 0.675 - pose_y, y= - 0.3 - pose_x + 0.03, z= 0.49 - pose_z + 0.0)	            
		orientation = Quaternion(x=0,y=1,z=0,w=0)
		final_pose = Pose(position=position, orientation=orientation)
		place_return = robot.place(final_pose)


	def init_baxter(self):
		robot = baxter()
		position = Point(x = 0.675, y=-0.3, z= 0.4)
		orientation = Quaternion(x=0.0,y=1.0,z=0.0,w=0.0)
		pose = Pose(position=position, orientation=orientation)
		robot.gripper_open()
		robot.move_to(pose)










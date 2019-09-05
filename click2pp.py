from baxter import baxter
import argparse
import struct
import sys
import copy
import rospy
import tensorflow as tf
from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)

import pyrealsense2 as rs
import numpy as np
import cv2

import os
import joblib
import scipy.misc
from skimage.transform import resize
from baxter_core_msgs.msg import EndpointState

from PickandPlace import PnP


def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #cv2.circle(images,(x,y),10000,(255,0,0),-1)
        mouseX,mouseY = x,y
        print('point x:{0}, y:{1}'.format(mouseX,mouseY))

def move_baxter(poses=None, state = 'start'):
	robot = baxter()

	#if poses:
	poses = poses.squeeze()
	s_px, s_py, s_pz, s_ox, s_oy, s_oz, s_ow, g_px, g_py, g_pz, g_ox, g_oy, g_oz, g_ow = poses
	# start_x, start_y = transition(start[0],start[1])
	# goal_x, goal_y = transition(goal[0],goal[1])

	position = Point(x = s_px, y=s_py, z= s_pz)
	orientation = Quaternion(x=s_ox,y=s_oy,z=s_oz,w=s_ow)
	start_pose = Pose(position=position, orientation=orientation)
	position = Point(x = g_px, y=g_py, z= g_pz)
	orientation = Quaternion(x=g_ox,y=g_oy,z=g_oz,w=g_ow)
	final_pose = Pose(position=position, orientation=orientation)
	print(start_pose)
	print(final_pose)
	# robot.move_to(pose)
	if state == 'picked' : 
		place_return = robot.place(final_pose)
		if place_return == False : return 2
	else :
		pick_return = robot.pick(start_pose)
		if pick_return == False : return 1
		place_return = robot.place(final_pose)
		if place_return == False : return 2
	return 0

def pos_callback(data):
	global robot_pose, robot_orientation
	robot_pose = data.pose.position
	robot_orientation = data.pose.orientation


def init_baxter():
	robot = baxter()

	position = Point(x = 0.675, y=-0.3, z= 0.4)
	orientation = Quaternion(x=0.0,y=1.0,z=0.0,w=0.0)
	pose = Pose(position=position, orientation=orientation)
	robot.gripper_open()
	robot.move_to(pose)
	
def move_baxter(pose_x, pose_y, pose_z, mode='pick'):
	robot = baxter()
 	if mode == 'pick' : 
	    position = Point(x = 0.700 - pose_y, y= - 0.3 - pose_x + 0.03, z= 0.49 - pose_z - 0.025)
	else :
	    position = Point(x = 0.725 - pose_y, y=0.0 - pose_x, z= 0.5 - pose_z + 0.05)
	orientation = Quaternion(x=0.0,y=1.0,z=0.0,w=0.0)
	pose = Pose(position=position, orientation=orientation)
	#robot.gripper_open()
	robot.move_to(pose)



def main():

	rospy.init_node("obin_click2pickplace")
	rospy.Subscriber("/robot/limb/right/endpoint_state",EndpointState,pos_callback)
	
	# streaming
	pipeline = rs.pipeline()

	# Create a config and configure the pipeline to stream
	#  different resolutions of color and depth streams
	config = rs.config()
	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

	# Start streaming
	profile = pipeline.start(config)

	# Getting the depth sensor's depth scale (see rs-align example for explanation)
	depth_sensor = profile.get_device().first_depth_sensor()
	depth_scale = depth_sensor.get_depth_scale()
	print("Depth Scale is: ", depth_scale)

	# We will be removing the background of objects more than
	#  clipping_distance_in_meters meters away
	clipping_distance_in_meters = 1  # 1 meter
	clipping_distance = clipping_distance_in_meters / depth_scale

	# Create an align object
	# rs.align allows us to perform alignment of depth frames to others frames
	# The "align_to" is the stream type to which we plan to align depth frames.
	align_to = rs.stream.color
	align = rs.align(align_to)

	path_image = '/home/obin/t2b_dataset/image'
	path_depth = '/home/obin/t2b_dataset/depth'

	path_pp = '/home/obin/t2b_dataset/pickandplace'

	EPOCH = 500
	BATCH_SIZE = 1
	LEARNING_RATE = 1e-5
	HIDDEN_UNITS = 8
	SAVE = 20
	VALID = 5

	pnp = PnP(rs,pipeline)
	init_baxter()
	robot = baxter()
	# Streaming loop
	try:
		while True:
			# Get frameset of color and depth
			frames = pipeline.wait_for_frames()
			# frames.get_depth_frame() is a 640x360 depth image
			color_frame = frames.get_color_frame()
			intrin = color_frame.profile.as_video_stream_profile().intrinsics
			aligned_frames = align.process(frames)

			# Get aligned frames
			aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
			color_frame = aligned_frames.get_color_frame()

			# Validate that both frames are valid
			if not aligned_depth_frame or not color_frame:
			    continue

			depth_image = np.asanyarray(aligned_depth_frame.get_data())
			color_image = np.asanyarray(color_frame.get_data())


			img_center = [250, 365]
			img_width = 190
			color_image = color_image[img_center[0]-img_width:img_center[0]+img_width,
			                         img_center[1]-img_width:img_center[1]+img_width, :]
			depth_image = depth_image[img_center[0]-img_width:img_center[0]+img_width,
			                         img_center[1]-img_width:img_center[1]+img_width]
			color_image = resize(np.fliplr(np.flipud(color_image)), [256, 256], preserve_range=False)
			depth_image = resize(np.fliplr(np.flipud(depth_image)), [256, 256], preserve_range=True)


			depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03)*25, cv2.COLORMAP_JET)
			global images 
			images = np.hstack((color_image, depth_colormap)) #bg_removed
			try :
				images = cv2.circle(images, (mouseX,mouseY), 3, (255,255,255), -1)
			except : 
				pass
			try : 
				images = cv2.circle(images, (start_pix_x,start_pix_y), 3, (0,0,255), -1)
				images = cv2.circle(images, (goal_pix_x,goal_pix_y), 3, (0,255,0), -1)
			except : pass
			cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
			cv2.imshow('Align Example', images[:,:,[2,1,0]])
			cv2.setMouseCallback('Align Example',draw_circle)

			key = cv2.waitKey(1)
			# Press esc or 'q' to close the image window, 's' to save files
			if key & 0xFF == ord('q') or key == 27:
				cv2.destroyAllWindows()
				break

			elif key & 0xFF == ord('c'):
				start_depth = depth_image[mouseY,mouseX]
				print('set start pose x: {0}, y: {1}, d: {2}'.format(mouseX,mouseY, start_depth))
				print('in case of : {0}'.format(np.asanyarray(aligned_depth_frame.get_data())[mouseY,mouseX]))
				start_pix_x = mouseX
				start_pix_y = mouseY

			elif key & 0xFF == ord('v'):
				goal_depth = depth_image[mouseY,mouseX]
				print('set goal pose x: {0}, y: {1}, d: {2}'.format(mouseX,mouseY, goal_depth))
				goal_pix_x = mouseX
				goal_pix_y = mouseY


			elif key & 0xFF == ord('a'):
				print(robot_pose)
				print(robot_orientation)
				print('\n')


			elif key & 0xFF == ord('i'):
				print("initialize baxter")
				init_baxter()
			elif key & 0xFF == ord('n'):
				robot.gripper_open()

			elif key & 0xFF == ord('m'):
				robot.gripper_close()
				
			elif key & 0xFF == ord('7'):
			    depth_pixel = [555 - int(start_pix_x * 380/256.), 420 - int(start_pix_y * 380/256.)]
			    #depth_pixel = [start_pix_x, start_pix_y]
			    depth = np.array(aligned_depth_frame.data)[depth_pixel[1], depth_pixel[0]] * 0.001 #self._depth_scale
			    print(depth)
			    point = rs.rs2_deproject_pixel_to_point(intrin, depth_pixel, depth)
			    print(point)
			    move_baxter(point[0], point[1], point[2],'pick')
			    
			elif key & 0xFF == ord('5'):
			    pnp.pick_and_place([start_pix_x, start_pix_y, 0, goal_pix_x, goal_pix_y, 0],depth_image)


	finally:
	    pipeline.stop()



	return 0

if __name__ == '__main__':
	sys.exit(main())	

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

	position = Point(x = 0.675, y=0.0, z= 0.4)
	orientation = Quaternion(x=0.0,y=1.0,z=0.0,w=0.0)
	pose = Pose(position=position, orientation=orientation)
	robot.gripper_open()
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

	num_data = len(os.listdir(path_pp+'/data'))
	print(num_data)
	EPOCH = 500
	BATCH_SIZE = 1
	LEARNING_RATE = 1e-5
	HIDDEN_UNITS = 8
	SAVE = 20
	VALID = 5
	tf.reset_default_graph()
	#with tf.Session() as sess:
	sess = tf.Session()
	#model = Model(sess, 8, 14, BATCH_SIZE, LEARNING_RATE, HIDDEN_UNITS)

	pnp = PnP(sess)
	init_baxter()
	# Streaming loop
	try:
		while True:
			# Get frameset of color and depth
			frames = pipeline.wait_for_frames()
			# frames.get_depth_frame() is a 640x360 depth image

			# Align the depth frame to color frame
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

			#color_image = color_image / 255.0

			# Remove background - Set pixels further than clipping_distance to grey
			# grey_color = 153
			# depth_image_3d = np.dstack(
			#    (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
			# bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

			# Render images
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
			cv2.imshow('Align Example', images)
			cv2.setMouseCallback('Align Example',draw_circle)

			key = cv2.waitKey(1)
	        # Press esc or 'q' to close the image window, 's' to save files
			if key & 0xFF == ord('q') or key == 27:
				cv2.destroyAllWindows()
				break
			elif key & 0xFF == ord('s'):
				print('Start: Saving %04d data to the path...' % (num_data))
				scipy.misc.imsave(os.path.join(path_image, '%04d.png'%(num_data)), color_image)
				joblib.dump(depth_image, os.path.join(path_depth, '%04d.dat.gz'%(num_data)))
				#joblib.dump([robot_pose,robot_orientation],os.path.join())
				print('Done : Saved %04d data to the path...' % (num_data))
				print('=======================================')
				num_data += 1
			elif key & 0xFF == ord('c'):
				start_depth = depth_image[mouseY,mouseX]
				print('set start pose x: {0}, y: {1}, d: {2}'.format(mouseX,mouseY, start_depth))
				print('in case of : {0}'.format(depth_image[mouseX,mouseY]))
				start_pix_x = mouseX
				start_pix_y = mouseY

			elif key & 0xFF == ord('v'):
				goal_depth = depth_image[mouseY,mouseX]
				print('set goal pose x: {0}, y: {1}, d: {2}'.format(mouseX,mouseY, goal_depth))
				goal_pix_x = mouseX
				goal_pix_y = mouseY

			elif key & 0xFF == ord('m'):
				#print('move..!')
				print('=======================================')
				print('move..! start ({0},{1}) -> goal ({2},{3})'.format(start_pix_x,start_pix_y,goal_pix_x,goal_pix_y))
				#move_baxter([start_pix_x,start_pix_y],[goal_pix_x,goal_pix_y])
				scipy.misc.imsave(os.path.join(path_pp, 'image/%04d.png'%(num_data)), color_image)
				joblib.dump(depth_image, os.path.join(path_pp, 'depth/%04d.dat.gz'%(num_data)))
				curr_data = dict()
				curr_data['image'] = color_image
				curr_data['depth'] = depth_image
				curr_data['start_pix'] = [start_pix_x, start_pix_y]
				curr_data['goal_pix'] = [goal_pix_x,goal_pix_y]

			elif key & 0xFF == ord('1'):
				print('mode : horizontal orientation')
				curr_data['orientation'] = 0
			elif key & 0xFF == ord('2'):
				print('mode : vertical orientation')
				curr_data['orientation'] = 1

			elif key & 0xFF == ord('a'):
				print(robot_pose)
				print(robot_orientation)
				print('\n')
			elif key & 0xFF == ord('p'):
				print("Pick pose", robot_pose, robot_orientation)
				curr_data['pick'] = np.array([robot_pose, robot_orientation])
				robot = baxter()
				robot.gripper_close()

			elif key & 0xFF == ord('l'):
				print("Place pose", robot_pose, robot_orientation)
				curr_data['place'] = np.array([robot_pose, robot_orientation])
				joblib.dump(curr_data, os.path.join(path_pp,'data/%04d.dat.gz'%(num_data)))
				print('Done : Saved %04d data to the path...' % (num_data))
				print('=======================================')
				num_data += 1
				robot = baxter()
				robot.gripper_open()

			elif key & 0xFF == ord('i'):
				print("initialize baxter")
				init_baxter()

			elif key & 0xFF == ord('5'):
				print("move...>!!!!")
				start_pix_x -= 5
				start_pix_y -= 10
				goal_pix_x -= 5
				goal_pix_y -= 10
				
				pnp.pick_and_place([start_pix_x, start_pix_y, 0, goal_pix_x, goal_pix_y, 0],depth_image)
				
				# mean_depth = 713.3002964231947
				# state = 'start'
				# for i in range(6):
				# 	sz = (depth_image[start_pix_y,start_pix_x]-mean_depth)/mean_depth
				# 	gz = (depth_image[goal_pix_y,goal_pix_x]-mean_depth)/mean_depth

				# 	start_x = (start_pix_x-128)/128.0
				# 	start_y = (start_pix_y-128)/128.0
				# 	goal_x = (goal_pix_x-128)/128.0
				# 	goal_y = (goal_pix_y-128)/128.0
				# 	x = np.array([start_x, start_y, sz, 0, goal_x, goal_y,gz, 0])

				# 	answer = []

				# 	answer.extend(np.squeeze(pick_pose_model.predict(x[0:2].reshape(-1,2))))
				# 	answer.extend(pick_z_model.predict(x[2].reshape(-1,1))[0])
				# 	pick_ori_model = pick_ori_models[int(x[3])]
				# 	answer.extend(pick_ori_model.predict(x[0:3].reshape(-1,3))[0])

				# 	answer.extend(place_pose_model.predict(x[4:6].reshape(-1,2))[0])
				# 	answer.extend(place_z_model.predict(x[6].reshape(-1,1))[0])
				# 	place_ori_model = place_ori_models[int(x[7])]
				# 	answer.extend(place_ori_model.predict(x[4:7].reshape(-1,3))[0])

				# 	#y = model.predict(np.array(x).reshape(-1,8))
				# 	answer[9] = -0.10
				# 	move_return = move_baxter(np.array(answer),state)
				# 	if move_return == 0 : 
				# 		print('break!')
				# 		break
				# 	elif move_return == 1 :
				# 		print('retry pick {0}'.format(i))
				# 		start_pix_x += np.random.randint(-1,2)*5
				# 		start_pix_y += np.random.randint(-1,2)*5
				# 		state = 'start'
				# 		continue
				# 	elif move_return == 2 :
				# 		print('retry place {0}'.format(i))
				# 		goal_pix_x += np.random.randint(-1,2)*5
				# 		goal_pix_y += np.random.randint(-1,2)*5
				# 		state = 'picked'
				# 		continue


	finally:
	    pipeline.stop()



	return 0

if __name__ == '__main__':
	sys.exit(main())
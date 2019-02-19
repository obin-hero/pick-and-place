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

class Model(object):
    def __init__(self, sess, name, input_size, output_size, BATCH_SIZE=1, LEARNING_RATE=1e-4, HIDDEN_UNITS=4):
        self.sess = sess
        self.batch_size = BATCH_SIZE
        self.learning_rate = LEARNING_RATE
        self.input = tf.placeholder(tf.float32, [None, input_size])
        self.target = tf.placeholder(tf.float32, [None, output_size])
        self.training = tf.placeholder(tf.bool)
        self.name = name
        with tf.variable_scope(name):
            init = tf.contrib.layers.xavier_initializer()
            if 'ori' in name : HIDDEN_UNITS = 8
            x = tf.layers.dense(self.input, HIDDEN_UNITS, kernel_initializer=init)
            self.output = tf.layers.dense(x, output_size, kernel_initializer=init)
            self.mse_loss = tf.reduce_mean(tf.square(self.target - self.output))
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.mse_loss)
        
    def predict(self,inputs,pick_info = None):

        feed_dict = {self.input : inputs,self.training : False}
        pred_y = self.sess.run(self.output, feed_dict=feed_dict)
        return pred_y
    

        return pred_y

def pretty_print_input(arr, ori = False):
    arr = np.squeeze(arr)
    if ori == True : 
        s_x, s_y, s_d, s_ori = arr[0:4]
        g_x, g_y, g_d, g_ori = arr[4:8]
        
        print('start pixel        goal pixek')
        print('x : %04f    x : %04f'%(s_x,g_x))
        print('y : %04f    y : %04f'%(s_y,g_y))
        print('z : %04f    z : %04f'%(s_d,g_d))
        print('o : %04f    o : %04f'%(s_ori,g_ori))
        print('\n')
    else : 
        s_x, s_y, s_d = arr[0:3]
        g_x, g_y, g_d = arr[3:6]
        
        print('start pixel        goal pixek')
        print('x : %04f    x : %04f'%(s_x,g_x))
        print('y : %04f    y : %04f'%(s_y,g_y))
        print('z : %04f    z : %04f'%(s_d,g_d))
        print('\n')
    
    
def pretty_print_out(arr):
    arr = np.squeeze(arr)
    s_px, s_py, s_pz, s_ox, s_oy, s_oz, s_ow = arr[0:7]
    g_px, g_py, g_pz, g_ox, g_oy, g_oz, g_ow = arr[7:14]
    
    print('start pose        goal pose')
    print('x : %04f    x : %04f'%(s_px,g_px))
    print('y : %04f    y : %04f'%(s_py,g_py))
    print('z : %04f    z : %04f'%(s_pz,g_pz))
    print('x : %04f    x : %04f'%(s_ox,g_ox))
    print('y : %04f    y : %04f'%(s_oy,g_oy))
    print('z : %04f    z : %04f'%(s_oz,g_oz))
    print('w : %04f    w : %04f'%(s_ow,g_ow))
    print('\n')
    


class PnP(object):
	def __init__(self,sess):
		
		#rospy.init_node("pick_and_place")
		print('Load Pick and Place Network')

		self.pick_pose_model = Model(sess,'pick_pose', 3, 3)
		self.pick_ori_models = []
		for i in range(3): 
		    pick_ori_model = Model(sess,'pick_ori_'+str(i), 3, 4)
		    self.pick_ori_models.append(pick_ori_model)
		# pick 7 
		self.place_pose_model = Model(sess,'place_pose', 6, 3)
		#self.place_z_model = Model(sess,'place_z', 1, 1)
		self.place_ori_models = []
		for i in range(3): 
		    place_ori_model = Model(sess,'place_ori_'+str(i),7,4)
		    self.place_ori_models.append(place_ori_model)

		saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_networks/to")
		if checkpoint and checkpoint.model_checkpoint_path:
			saver.restore(sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find trained network weights")

		self.retry = 6
		self.init_baxter()


	def pick_and_place(self,input_arr,depth_image):

		'''

		* input_arr shape(6,1) or (4,1)

		[start_pixel_x, start_pixel_y, pick_orientation,goal_pixel_x, goal_pixel_y, goal_orientation]
		or
		[start_pixel_x, start_pixel_y, goal_pixel_x, goal_pixel_y]

		- if orientation is not specified, we'll set it to 0
		
		* pick_orientation / goal_orientation 
		which direction should the robot pick/place a block, 
		horizontally - 0, vertical & in the right - 1, vertical & in the left -2

		'''
		print('Start Pick and Place')

		state = 'start'
		for i in range(self.retry):
			print('retry?')
			answer = []
			print(input_arr)
			x = np.array(self.preprocess_data(input_arr,depth_image))
			print('x : ',x)
			pre_pos = np.squeeze(self.pick_pose_model.predict(x[0:3].reshape(-1,3)))
			pick_ori_model = self.pick_ori_models[int(x[3])]
			pre_ori = pick_ori_model.predict(x[0:3].reshape(-1,3))[0]
			answer.extend(pre_pos)
			answer.extend(pre_ori)

			inp = np.concatenate([x[4:7],pre_pos]).reshape(-1,6)
			answer.extend(self.place_pose_model.predict(inp)[0])
			place_ori_model = self.place_ori_models[int(x[7])]
			inp = np.concatenate([x[4:7],pre_ori]).reshape(-1,7)
			answer.extend(place_ori_model.predict(inp)[0])

			move_return = self.move_baxter(np.array(answer),state,'pnp')
			if move_return == 0 : 
				print('nice!') 
				break
			elif move_return == 1 :
				# pick falied
				print('retry pick {0}'.format(i))
				input_arr[0] += np.random.randint(-1,2)*5
				input_arr[1] += np.random.randint(-1,2)*5
				state = 'start'
				continue
			elif move_return == 2 :
				# place falied
				print('retry place {0}'.format(i))
				input_arr[3] += np.random.randint(-1,2)*5
				input_arr[4] += np.random.randint(-1,2)*5
				state = 'picked'
				continue

	def preprocess_data(self, input_arr, depth_image):
		
		arr_length = len(input_arr)
		if arr_length == 2 : 
			start_pix_x, start_pix_y, = input_arr
			start_ori = 0
		
		elif len(input_arr) == 3 :
			start_pix_x, start_pix_y, pick_ori = input_arr

		elif len(input_arr) == 4 : 
			start_pix_x, start_pix_y, goal_pix_x, goal_pix_y = input_arr
			pick_ori = place_ori = 0
		
		elif len(input_arr) == 6 :
			start_pix_x, start_pix_y, pick_ori, goal_pix_x, goal_pix_y, place_ori = input_arr

		else : 
			print('input shape is wrong! dont know what did you mean')
			return None

		# preprocess depth image
		depth_image  = (750 - depth_image)/750
		depth_image[np.where(depth_image>= 0.5)] = 0
		depth_image[np.where(depth_image < 0)] = 0
		depth_image = depth_image*5

		kernel = np.ones((10,10),np.float32)/100
		depth_image = cv2.filter2D(depth_image,-1,kernel)

		if arr_length == 2 or arr_length == 3 :
			# pick or place
			start_depth = depth_image[start_pix_y,start_pix_x]
			start_x, start_y = (np.array([start_pix_x,start_pix_y])-128)/128.0
			return [start_pix_x, start_pix_y, start_depth, start_ori]

		elif arr_length == 4 or arr_length == 6 :
			# pick and place
			start_depth = depth_image[start_pix_y,start_pix_x]
			goal_depth = depth_image[goal_pix_y,goal_pix_x]

			arr = [start_pix_x, start_pix_y, start_depth, pick_ori, goal_pix_x, goal_pix_y, goal_depth, place_ori]
			pretty_print_input(arr,True)
			
			start_x, start_y = (np.array([start_pix_x,start_pix_y])-128)/128.0
			print("start x {0}, start y {1}".format(start_x,start_y))

			ad = np.power([start_x, start_y],3) * np.power(start_depth, 3) * 0.1
			print("adjust {0}".format(ad))
			start_x, start_y = np.array([start_x,start_y]) - ad
			print("start x {0}, start y {1}".format(start_x,start_y))
			goal_x, goal_y = (np.array([goal_pix_x,goal_pix_y])-128)/128.0
			print("goal x {0}, goal y {1}".format(goal_x,goal_y))
			ad = np.power([goal_x, goal_y],3) * np.power(goal_depth, 3) * 0.1
			print("adjust {0}".format(ad))
			goal_x, goal_y = np.array([goal_x,goal_y]) - ad 
			print("goal x {0}, goal y {1}".format(goal_x,goal_y))
			return [start_x, start_y, start_depth, pick_ori, goal_x, goal_y, goal_depth, place_ori]


	def move_baxter(self,poses, state = 'start', mode = 'pnp'):
		robot = baxter()

		#if poses:
		poses = np.squeeze(poses)
		print(mode)
		if mode == 'pnp':
			print('pick and place mode')
			s_px, s_py, s_pz, s_ox, s_oy, s_oz, s_ow, g_px, g_py, g_pz, g_ox, g_oy, g_oz, g_ow = poses

			position = Point(x = s_px, y=s_py, z= s_pz)
			orientation = Quaternion(x=s_ox,y=s_oy,z=s_oz,w=s_ow)
			start_pose = Pose(position=position, orientation=orientation)
			position = Point(x = g_px, y=g_py, z= g_pz)
			orientation = Quaternion(x=g_ox,y=g_oy,z=g_oz,w=g_ow)
			final_pose = Pose(position=position, orientation=orientation)
			print('----------start pose----------')
			print(start_pose)
			print('----------final pose----------')
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

		elif mode == 'pick' or mode == 'place':
			print('pick or place mode')
			s_px, s_py, s_pz, s_ox, s_oy, s_oz, s_ow = poses

			position = Point(x = s_px, y=s_py, z= s_pz)
			orientation = Quaternion(x=s_ox,y=s_oy,z=s_oz,w=s_ow)
			start_pose = Pose(position=position, orientation=orientation)

			print('----------pose----------')
			print(start_pose)

			if mode == 'pick':
				pick_return = robot.pick(start_pose)
				if pick_return == False : return 1
			elif mode == 'place' : 
				place_return = robot.place(start_pose)
				if place_return == False : return 1
			return 0


	def init_baxter(self):
		robot = baxter()
		position = Point(x = 0.675, y=0.0, z= 0.4)
		orientation = Quaternion(x=0.0,y=1.0,z=0.0,w=0.0)
		pose = Pose(position=position, orientation=orientation)
		robot.gripper_open()
		robot.move_to(pose)
		









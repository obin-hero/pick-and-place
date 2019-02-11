import sys
import numpy as np
import cv2
import os
import joblib
import scipy.misc
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        #cv2.circle(images,(x,y),10000,(255,0,0),-1)
        mouseX,mouseY = x,y
        print('point x:{0}, y:{1}'.format(mouseX,mouseY))
def click(event):

    print(str(event.xdata)+str(event.ydata))



def main():
	pp_path = '/home/obin/t2b_dataset/pickandplace'
	data_list = np.sort([pp_path+'/data/'+x for x in os.listdir(os.path.join(pp_path,'data'))])
	pp_path = '/home/obin/t2b_dataset/pickandplace/0202'
	data_list2 = np.sort([pp_path+'/data/'+x for x in os.listdir(os.path.join(pp_path,'data'))])
	data_list = np.hstack((data_list,data_list2))
	num_data = len(data_list)
	print('num of data : ', num_data)

	global curr_idx
	curr_idx = 0
	curr_data = joblib.load(os.path.join(pp_path,data_list[curr_idx]))
	global end_flag
	end_flag = False

	while(True):

		# plt.subplot(121)
		# plt.imshow(curr_data['image'])
		# plt.scatter(curr_data['start_pix'][0], curr_data['start_pix'][1], s=20, c='w', marker='o')
		# plt.scatter(curr_data['goal_pix'][0], curr_data['goal_pix'][1], s=20, c='w', marker='x')
		# plt.subplot(122)
		# kernel = np.ones((10,10),np.float32)/100
		# dst = cv2.blur(curr_data['depth'],(10,10))
		
		curr_data = joblib.load(os.path.join(pp_path,data_list[curr_idx]))

		fig = plt.figure(curr_idx)

		curr_depth = -curr_data['depth'] + 715.37
		curr_depth = curr_depth/715.37
		curr_depth[np.where(curr_depth>=0.5)]=0
		curr_depth[np.where(curr_depth<=0)]=0
		curr_depth = curr_depth *2

		curr_depth = denoise_tv_chambolle(curr_depth, weight=0.25, multichannel=False)
		curr_depth = cv2.blur(curr_depth,(20,20))

		# plt.imshow(curr_depth)
		# plt.show()

		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(curr_data['depth'], alpha=0.03), cv2.COLORMAP_JET)
		#global images 
		#images = cv2.hconcat((curr_data['image'], curr_data['depth'])) #bg_removed
		#cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)

		a1 = fig.add_subplot(121)
		a1.imshow(curr_data['image'])
		a1.scatter(curr_data['start_pix'][0], curr_data['start_pix'][1], s=20, c='w', marker='o')
		a1.scatter(curr_data['goal_pix'][0], curr_data['goal_pix'][1], s=20, c='w', marker='x')
		a2 = fig.add_subplot(122)
		a2.scatter(curr_data['start_pix'][0], curr_data['start_pix'][1], s=20, c='w', marker='o')
		a2.scatter(curr_data['goal_pix'][0], curr_data['goal_pix'][1], s=20, c='w', marker='x')
		a2.imshow(curr_data['depth'])
		#cv2.setMouseCallback('Align Example',draw_circle)

		def click(event):
			x = int(event.xdata)
			y = int(event.ydata)
			print('x :{0}, y:{1}, depth:{2}, depth?:{3}'.format(event.xdata,event.ydata, curr_data['depth'][x,y],curr_data['depth'][y,x]))

		def key_press(event):
			global curr_idx
			global end_flag
			if event.key == 'q':
				end_flag = True

			elif event.key == 'n':
				curr_idx += 1
				plt.close()
				print(curr_idx)

		if end_flag : break

			



		fig.canvas.mpl_connect('button_press_event',click)
		fig.canvas.mpl_connect('key_press_event',key_press)
		plt.show()


if __name__ == '__main__':
	sys.exit(main())

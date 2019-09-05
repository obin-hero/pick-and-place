# pick-and-place

```
source baxter_real.sh
python click2pp.py
```
### make sure
1. you are connected with the correct wi-fi network
2. ROS is installed - if it is not kinetic, you should edit baxter_real.sh
3. realsense library is installed - `pip install pyrealsense2` may be..?
- you might need to install librealsense 


#### if you are able to load the saved network

1. press 'i' to initialize baxter's pose 
2. click any point on the RGB image of the camera window 
3. press 'c' to set the point as a start point
4. click another point on the RGB image and press 'v' to set the point as a goal point.
5. Finally, press '5'.
6. Baxter's right arm will move from the start point to the goal point.





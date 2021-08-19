# Honda Dataset

After initial parsing of the raw data, we have a one to one correspondance with driving image frames and [1, 8] Tensor
where, the 8 different values represent the CANbus data as under :
|  No   | Description |
| ----- | ----------- |
|   1   | accel_pedal : pedalangle (percent) |
|   2   | steer : steer_angle                |
|   3   | steer : steer_speed                |
|   4   | speed : longitudinal speed (m/s)   |
|   5   | brake_pedal : pressure(kPa)        |
|   6   | turn_signal : left turn (binary)   |
|   7   | turn_signal : right turn (binary)  |
|   8   | yaw (degree/s)                     |


## Camera 

The image size is | W x H :: 1280 x 720 |.

In the intial stage for the pre-processing, we need to crop out the hood of the ego vehicle. It is also seen in previous
works that if the sky/background present in the top portion of the frame is removed (cropped out), it helps in
generalization of the model. (Removing uncessary data that doesn't have any effect on the control of the ego-vehicle)

After this, we can try to add noise in the form of Gaussian blur of some intensity. We can also look into resizing the
images for faster training. (at loss of some level of information)



import random

import tensorflow as tf
import tensorflow_io as tfio

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import cv2
import numpy as np

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
LONG_STEPS = 100000000000

fig = plt.figure()
image = plt.imread( "F:\\datasets\\downloads\\cats_name\\train\\Symbols\\01.jpg" )

center_image = plt.imread( "F:\\temp\\20221217\\center_image.png" )
center_image = tf.image.resize(center_image, [224, 224])
center_image = tf.keras.utils.img_to_array( tfio.experimental.color.rgba_to_rgb( center_image ) )

center_image_left = plt.imread( "F:\\temp\\20221217\\center_image_left.png" )
center_image_left = tf.image.resize(center_image_left, [224, 224])
center_image_left = tf.keras.utils.img_to_array( tfio.experimental.color.rgba_to_rgb( center_image_left ) )

center_image_right = plt.imread( "F:\\temp\\20221217\\center_image_right.png" )
center_image_right = tf.image.resize(center_image_right, [224, 224])
center_image_right = tf.keras.utils.img_to_array( tfio.experimental.color.rgba_to_rgb( center_image_right ) )
im = plt.imshow(image)

##############################################################################################
temp_image_center_layer_1 = tf.keras.layers.Conv2D( 3, ( 3, 3 ), strides=(32, 32), padding='valid', activation='relu' )( tf.expand_dims(center_image, axis=0) )
temp_image_center_layer_1 = tf.keras.layers.Softmax()( temp_image_center_layer_1 )
temp_image_center_layer_1 = tf.math.argmax( tf.squeeze( temp_image_center_layer_1 ), axis=0 )
temp_image_center_layer_1 = tf.math.argmax( tf.squeeze( temp_image_center_layer_1 ), axis=1 ).numpy()
temp_image_center_layer_1 = [ 1, 255, 255, 255, 255, 255, 1 ]

temp_image_center_left_layer_1 = tf.keras.layers.Conv2D( 3, ( 3, 3 ), strides=(32, 32), padding='valid', activation='relu' )( tf.expand_dims(center_image_left, axis=0) )
temp_image_center_left_layer_1 = tf.keras.layers.Softmax()( temp_image_center_left_layer_1 )
temp_image_center_left_layer_1 = tf.math.argmax( tf.squeeze( temp_image_center_left_layer_1 ), axis=0 )
temp_image_center_left_layer_1 = tf.math.argmax( tf.squeeze( temp_image_center_left_layer_1 ), axis=1 ).numpy()
temp_image_center_left_layer_1 = [ 1, 1, 1, -100, -150, -200, -255 ]

temp_image_center_right_layer_1 = tf.keras.layers.Conv2D( 3, ( 3, 3 ), strides=(32, 32), padding='valid', activation='relu' )( tf.expand_dims(center_image_right, axis=0) )
temp_image_center_right_layer_1 = tf.keras.layers.Softmax()( temp_image_center_right_layer_1 )
temp_image_center_right_layer_1 = tf.math.argmax( tf.squeeze( temp_image_center_right_layer_1 ), axis=0 )
temp_image_center_right_layer_1 = tf.math.argmax( tf.squeeze( temp_image_center_right_layer_1 ), axis=1 ).numpy()
temp_image_center_right_layer_1 = [ 255, 200, 150, 100, 1, 1, 1 ]
##############################################################################################

global previous_center_move_scores
previous_center_move_scores = 0
global previous_leftmove_scores
previous_leftmove_scores = 0
global previous_rightmove_scores
previous_rightmove_scores = 0

global frames_count
frames_count = 0

global stacks_frames
stacks_frames = [ tf.zeros([ 224, 224, 3 ]).numpy(), tf.zeros([ 224, 224, 3 ]).numpy(), tf.zeros([ 224, 224, 3 ]).numpy(), tf.zeros([ 224, 224, 3 ]).numpy() ]
stacks_frames = tf.constant( stacks_frames, shape=(4, 224, 224, 3) )

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Definition
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def animate( image ):
	global frames_count
	global stacks_frames
	
	
	
	
	

	######
	new_image, x_label = locate_movement( sample_video[frames_count] )
	
	######
	
	image = tf.squeeze( sample_video[frames_count] )
	# image = tf.squeeze( image )
	image = tf.keras.utils.array_to_img( image )
	
	# print( image.shape )
	
	######
	im.set_array( new_image )
	# im.set_array( image )
	######
	
	
	plt.xlabel( x_label, fontsize=22  )
	plt.ylabel( "", fontsize=22  )
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.show()
	
	if frames_count + 1 < 60:
		frames_count = frames_count + 1
	else:
		frames_count = 0
	
	return im,

def format_frames(frame, output_size):
	"""
	Pad and resize an image from a video.

	Args:
	frame: Image that needs to resized and padded. 
	output_size: Pixel size of the output frame image.

	Return:
	Formatted frame with padding of specified output size.
	"""
	frame = tf.image.convert_image_dtype(frame, tf.float32)
	frame = tf.image.resize_with_pad(frame, *output_size)
	return frame

def frames_from_video_file(video_path, n_frames, output_size = (224,224), frame_step = 15):
	"""
	Creates frames from each video file present for each category.

	Args:
	  video_path: File path to the video.
	  n_frames: Number of frames to be created per video file.
	  output_size: Pixel size of the output frame image.

	Return:
	  An NumPy array of frames in the shape of (n_frames, height, width, channels).
	"""
	# Read each video frame by frame
	result = []
	src = cv2.VideoCapture(str(video_path))  

	video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

	need_length = 1 + (n_frames - 1) * frame_step

	if need_length > video_length:
		start = 0
	else:
		max_start = video_length - need_length
		start = random.randint(0, max_start + 1)

	src.set(cv2.CAP_PROP_POS_FRAMES, start)
	# ret is a boolean indicating whether read was successful, frame is the image itself
	ret, frame = src.read()
	result.append(format_frames(frame, output_size))

	for _ in range(n_frames - 1):
		for _ in range(frame_step):
			ret, frame = src.read()
		if ret:
			frame = format_frames(frame, output_size)
			result.append(frame)
		else:
			result.append(np.zeros_like(result[0]))
	src.release()
	result = np.array(result)[..., [2, 1, 0]]

	return result

def locate_movement( image ):
	global stacks_frames
	global previous_center_move_scores
	global previous_leftmove_scores
	global previous_rightmove_scores
	
	stacks_frames = tf.concat([stacks_frames, tf.expand_dims(sample_video[frames_count], axis=0)], axis=0)
	stacks_frames = stacks_frames[1:5,:,:,:]
	
	image_1 = tf.constant( stacks_frames[0,:,:,:], shape=( 224, 224, 3 ) )
	image_2 = tf.constant( stacks_frames[1,:,:,:], shape=( 224, 224, 3 ) )
	image_3 = tf.constant( stacks_frames[2,:,:,:], shape=( 224, 224, 3 ) )
	image_4 = tf.constant( stacks_frames[3,:,:,:], shape=( 224, 224, 3 ) )
	
	new_image = ( image_1 - image_2 ) + ( image_3 - image_4 )
	gray_scale = tf.image.rgb_to_grayscale( new_image )
	
	print( image_1.shape )
	print( center_image.shape )
	
	contrast_image = ( image_1 - image_2 )  - center_image
	
	width = contrast_image.shape[0]
	height = contrast_image.shape[1]
	width_half = int( contrast_image.shape[0] / 2 )
	height_half = int( contrast_image.shape[1] / 2 )
	
	temp_image_layer_1 = tf.keras.layers.Conv2D( 3, ( 3, 3 ), strides=(32, 32), padding='valid', activation='relu' )( tf.expand_dims(contrast_image, axis=0) )
	temp_image_layer_2 = tf.keras.layers.Conv2D( 1, ( 3, 3 ), strides=(4, 4), padding='valid', activation='relu' )( tf.expand_dims(temp_image_layer_1, axis=0) )
	temp_label_layer_1 = tf.keras.layers.Softmax()( temp_image_layer_1 )
	temp_label_layer_1 = tf.math.argmax( tf.squeeze( temp_label_layer_1 ), axis=0 )
	temp_label_layer_1 = tf.math.argmax( tf.squeeze( temp_label_layer_1 ), axis=1 ).numpy()
	
	# X_scales_direction = { "right": [0, 1], "left": [1, 0], "full": [1, 1], "none": [0, 0] }
	# X_label = list([ key for ( key, value ) in X_scales_direction.items() if np.array_equal(value, tf.math.argmax( tf.squeeze( temp_image_layer_2 ), axis=0 ).numpy() ) ])

	move_center_score = tf.reduce_sum( temp_image_center_layer_1 - temp_label_layer_1 ).numpy()
	move_left_score = tf.reduce_sum( temp_image_center_left_layer_1 - temp_label_layer_1 ).numpy()
	move_right_score = tf.reduce_sum( temp_image_center_right_layer_1 - temp_label_layer_1 ).numpy()
	
	print( move_left_score )
	print( move_right_score )
	
	x_label = "none"
	
	if move_center_score - previous_center_move_scores == 0 and move_left_score - previous_leftmove_scores == 0 and move_right_score - previous_rightmove_scores == 0 :
		x_label = "initail vel." + " " + str( move_center_score - previous_center_move_scores ) + " " + str( move_left_score - previous_leftmove_scores ) +  " " +  str( move_right_score - previous_rightmove_scores )
		
		#######################################################################################################
		# draw a box around the image
		box = np.array([0.22, 0.268, 0.52, 0.56])
		boxes = box.reshape([1, 1, 4])
		# alternate between red and blue
		colors = np.array([[255, 0, 255], [255, 0, 255]])
		image_4 = tf.image.draw_bounding_boxes( tf.expand_dims( image_4, axis=0 ), boxes, colors)
		#######################################################################################################
	
	elif move_center_score - previous_center_move_scores > 1 and move_left_score - previous_leftmove_scores > 1 and move_right_score - previous_rightmove_scores > 1 :
		x_label = "moving left" + " " + str( move_center_score - previous_center_move_scores ) + " " + str( move_left_score - previous_leftmove_scores ) +  " " +  str( move_right_score - previous_rightmove_scores )
	else:
		x_label = "moving right" + " " + str( move_center_score - previous_center_move_scores ) + " " + str( move_left_score - previous_leftmove_scores ) +  " " +  str( move_right_score - previous_rightmove_scores )
		
	previous_center_move_scores = move_center_score
	previous_leftmove_scores = move_left_score
	previous_rightmove_scores = move_right_score
	
	print( contrast_image.shape )
	
	return tf.keras.utils.array_to_img( tf.squeeze( image_4 ) ), x_label

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
video_path = "F:\\Pictures\\actor_note.pin\\90605779_126412262270684_895736804324836774_n.avi"
sample_video = frames_from_video_file(video_path, n_frames = 60)

while frames_count < 60 :
	ani = animation.FuncAnimation(fig, animate, interval=0, blit=True)
	plt.show()

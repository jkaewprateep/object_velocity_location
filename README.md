# object_velocity_location
Simply find object locate on screen using Python

## Object location locate by SoftMax() ###

Important tasks if your games or project require acion responses to player or object like dimentsions, one way is by weight of how much of current from the previous change you know that the background is not moving but if it is we will discuss about it in next project i.e. Street Fighter II Ultimates and other games. The problem is the networks senses as we do then we need to use grid mapping functoins as the ruler as we also difficult to senses accurate of shape sacels in details when we moving with velocity object suc as car or plane. Grid + Patterns is efficient and easy to implement someone called it image segmentation.

### Pattern mapping ###

It is possible to use only Python and Tensorflow to work as famous networks by perform simple tasks such as image locates inside the input picture, the concepts is easy as you moving right the value is change or moving left.
```
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
```

### Minimum-Maximum value ###

Simply weight of the image which sides is most weight, you can create it scales as 2 to 4 to 8 to 16 this is called grid segmentation for save the computation power otherwise you need to rulers all the image input if you need accurate inches but if not velocity image and SoftMax() is enough.
```
temp_image_layer_1 = tf.keras.layers.Conv2D( 3, ( 3, 3 ), strides=(32, 32), padding='valid', activation='relu' )( tf.expand_dims(contrast_image, axis=0) )
temp_image_layer_2 = tf.keras.layers.Conv2D( 1, ( 3, 3 ), strides=(4, 4), padding='valid', activation='relu' )( tf.expand_dims(temp_image_layer_1, axis=0) )
temp_label_layer_1 = tf.keras.layers.Softmax()( temp_image_layer_1 )
temp_label_layer_1 = tf.math.argmax( tf.squeeze( temp_label_layer_1 ), axis=0 )
temp_label_layer_1 = tf.math.argmax( tf.squeeze( temp_label_layer_1 ), axis=1 ).numpy()
```

## Files and Directory ##
1. sample.py : sample code for prove the logic
2. bandicam 2022-12-17 11-31-54-783.gif : result
3. README.md : readme file

## Result image ##
Label key target with clustering

![Alt text](https://github.com/jkaewprateep/object_velocity_location/blob/main/bandicam%202022-12-17%2011-31-54-783.gif?raw=true "Title")

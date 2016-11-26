to recreate: 

## this grabs the needed tensorflow image 
docker run -it gcr.io/tensorflow/tensorflow:latest-devel
# ctrl-D if you're still in Docker and then:
 cd $HOME
 mkdir tf_files
 cd tf_files
 ## this grabs the needed images to run the classification on flowers
 curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
 tar xzf flower_photos.tgz

# On OS X, see what's in the folder:
% open flower_photos


-------- if you would like to run classification on other categories remove the unneeded ones like so
% cd $HOME/tf_files/flower_photos
% rm -rf dandelion sunflowers tulips

# On OS X, make sure the flowers are gone!
% open .

## this makes your local files accesssible to the docker container
docker run -it -v $HOME/tf_files:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel


## once docker starts you can type this to see the contents of the container

# ls /tf_files/
flower_photos  flower_photos.tgz

## run this to pull in the needed code to classify images. Current image does not come with it 
# cd /tensorflow
# git pull

## start your image training with this long command 

# python tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=/tf_files/bottlenecks \
--how_many_training_steps 500 \
--model_dir=/tf_files/inception \
--output_graph=/tf_files/retrained_graph.pb \
--output_labels=/tf_files/retrained_labels.txt \
--image_dir /tf_files/flower_photos

## after that ^ command is completed running exit docker (command + d)
## create a file called label_image.py in the same directory as tf_files

__________________________________________________________________________________________________________________

Include this code below saved in label_image.py

__________________________________________________________________________________________________________________

import tensorflow as tf

# change this as you see fit
image_path = sys.argv[1]

# Read in the image_data
image_data = tf.gfile.FastGFile(image_path, 'rb').read()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("/tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    
    predictions = sess.run(softmax_tensor, \
             {'DecodeJpeg/contents:0': image_data})
    
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))


__________________________________________________________________________________________________________________


# ctrl-D to exit Docker and then:

## this links your local files to the docker container (could be redundant from above)
% curl -L https://goo.gl/tx3dqg > $HOME/tf_files/label_image.py 

## restart the docker image 
docker run -it -v $HOME/tf_files:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel 

## now run the classifier!

You can change the daisy/21652746_cc379e0eea_m.jpg to any other image.

python /tf_files/label_image.py /tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg



# dog_image_classifier
# dog_image_classifier
# image_classifier
# image_classifier

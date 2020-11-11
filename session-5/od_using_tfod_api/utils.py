import tensorflow as tf

def fix_cudnn_bug(): 
    # during training, tf will throw cudnn initialization error: failed to get convolution algos
    # the following codes somehow fix it
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
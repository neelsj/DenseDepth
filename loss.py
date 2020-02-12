import keras.backend as K
import tensorflow as tf

def masked_mean(x, mask):    
    return tf.reduce_mean(tf.boolean_mask(x, mask))

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
    
    mask = tf.greater(y_true, 0)
    y_pred = tf.where(mask, y_pred, tf.zeros_like(y_pred))    

    # Point-wise depth
    l_depth = masked_mean(K.abs(y_pred - y_true), mask)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = masked_mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), mask)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * l_edges) + (w3 * l_depth)
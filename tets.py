import tensorflow as tf
import keras

sess = tf.InteractiveSession()

# Sensitivity, true positive rate, recall: proportion of actual positives that are correctly identified as such
def sensitivity(y_true, y_pred): 
    true    = tf.cast(tf.constant([[1,0]]), tf.float32)
    y_true  = keras.backend.round(tf.cast(y_true, tf.float32))
    y_pred  = keras.backend.round(tf.cast(y_pred, tf.float32))
    #Predicted positive that are actually positive
    positive_found = tf.divide(tf.count_nonzero(tf.multiply( tf.cast(tf.equal(y_true, y_pred),tf.int32), tf.cast(tf.equal(y_true,true),tf.int32))), true.get_shape()[1])
    #All positives
    all_positives  = tf.divide(tf.count_nonzero(tf.equal(y_true, true)), true.get_shape()[1])
    return positive_found / (all_positives + keras.backend.epsilon())

# Specificity, true negative rate: proportion of actual negatives that are correctly identified as such
def specificity(y_true, y_pred): # Background_identified/Real_Background
    false   = tf.cast(tf.constant([[0,1]]), tf.float32)
    y_true  = keras.backend.round(tf.cast(y_true, tf.float32))
    y_pred  = keras.backend.round(tf.cast(y_pred, tf.float32))
    #Predicted negative that are actually negative
    negatives_found = tf.divide(tf.count_nonzero(tf.multiply( tf.cast(tf.equal(y_true, y_pred),tf.int32), tf.cast(tf.equal(y_true,false),tf.int32))), false.get_shape()[1])
    #All negative
    all_negatives   = tf.divide(tf.count_nonzero(tf.equal(y_true, false)), false.get_shape()[1])
    return negatives_found / (all_negatives + keras.backend.epsilon())

y_true = tf.constant([[1, 0], [1, 0], # 4 Pos, 2 Neg
                      [1, 0], [0, 1],
                      [1, 0], [0, 1]])

y_pred = tf.constant([[0.9, 0.1], [0.75, 0.25], # 3 Pos corretto, 1 Pos incorretto, 1 Neg corretto, 1 Neg incorretto
                      [0.3, 0.7], [0.8, 0.2],
                      [0.8, 0.2], [0.4, 0.6]])

sens = sensitivity(y_true, y_pred)
print 'sensitivity is', sens.eval()

spec = specificity(y_true, y_pred)
print 'specificity is', spec.eval()

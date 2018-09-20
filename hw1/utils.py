import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import pickle
game_shapes = {'Ant-v2':(111,8),'HalfCheetah-v2':(17,6),'Hopper-v2':(11,3),'Humanoid-v2':(376,17),'Reacher-v2':(11,2),'Walker2d-v2':(17,6)}
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def Read_Data(filename):
    with open('./expert_data1/{}.pkl'.format(filename), 'rb') as f:
        data = pickle.loads(f.read())
    x = data['observations']
    y = data['actions']
    return x,y

def tf_reset():
    try:
        sess.close()
    except:
        pass
    tf.reset_default_graph()
    return tf.Session()

def create_model_dynamic(obs_s,act_s,no_layers=2,no_nodes=[20,20]):
    # create inputs
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_s])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None,act_s])
    nn = tf.layers.dense(input_ph,no_nodes[0],activation = tf.nn.relu)
    for i in range (no_layers-1):
        nn = tf.layers.dense(nn,no_nodes[i+1],activation = tf.nn.relu)
    out_pred = tf.layers.dense(nn,act_s)
    return input_ph,output_ph,out_pred

def create_model(obs_s,act_s,no_nodes=[20,20]):
    # create inputs
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_s])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None,act_s])
    # create variables
    with tf.variable_scope('layer_0'):
        W0 = tf.get_variable(name='W0', shape=[obs_s,no_nodes[0]], initializer=tf.contrib.layers.xavier_initializer())
        b0 = tf.get_variable(name='b0', shape=[no_nodes[0]], initializer=tf.constant_initializer(0.))

    with tf.variable_scope('layer_1'):
        W1 = tf.get_variable(name='W1', shape=[no_nodes[0], no_nodes[1]], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable(name='b1', shape=[no_nodes[1]], initializer=tf.constant_initializer(0.))

    with tf.variable_scope('layer_2'):
        W2 = tf.get_variable(name='W2', shape=[no_nodes[1], act_s], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable(name='b2', shape=[act_s], initializer=tf.constant_initializer(0.))
    weights = [W0, W1, W2]
    biases = [b0, b1, b2]
    activations = [tf.nn.relu, tf.nn.relu, None]

    # create computation graph
    layer = input_ph
    for W, b, activation in zip(weights, biases, activations):
        layer = tf.matmul(layer, W) + b
        if activation is not None:
            layer = activation(layer)
    output_pred = layer
    
    return input_ph, output_ph, output_pred

def load_bc_policy(name,no_nodes):
    global game_shapes
    sess = tf_reset()
    input_ph,output_ph,output_pred = create_model(game_shapes[name][0],game_shapes[name][1],no_nodes=no_nodes)
    saver = tf.train.Saver()
    saver.restore(sess, "./bc_experts/{}.ckpt".format(name))
    return input_ph,output_ph,output_pred,sess

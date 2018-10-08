
import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tf_util
import gym
import load_policy
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

def create_model(obs_s,act_s,no_layers=2,no_nodes=[20,20]):
    # create inputs
    input_ph = tf.placeholder(dtype=tf.float32, shape=[None, obs_s])
    output_ph = tf.placeholder(dtype=tf.float32, shape=[None,act_s])
    nn = tf.layers.dense(input_ph,no_nodes[0],activation = tf.nn.relu)
    for i in range (no_layers-1):
        nn = tf.layers.dense(nn,no_nodes[i+1],activation = tf.nn.relu)
    out_pred = tf.layers.dense(nn,act_s)
    return input_ph,output_ph,out_pred

def load_bc_policy(name,in_shape,out_shape,no_layers=2,no_nodes=[20,20]):
    global game_shapes
    sess = tf_reset()
    input_ph,output_ph,output_pred = create_model(in_shape,out_shape,no_nodes=no_nodes)
    saver = tf.train.Saver()
    saver.restore(sess, "./bc_experts/{}.ckpt".format(name))
    return input_ph,output_ph,output_pred,sess

def load_dag_policy(name,in_shape,out_shape,no_layers=2,no_nodes=[20,20]):
    global game_shapes
    sess = tf_reset()
    input_ph,output_ph,output_pred = create_model(in_shape,out_shape,no_nodes=no_nodes)
    saver = tf.train.Saver()
    saver.restore(sess, "./dag_experts/{}.ckpt".format(name))
    return input_ph,output_ph,output_pred,sess

def train_dag_model(x,y,name ='Ant-v2',model_name=None,no_layers=2,no_nodes=[64,64],iteration=100000,create=False):
    if model_name is None:
        model_name = name
    sess = tf_reset()
    input_data,output_data,output_predictions = create_model(x.shape[1],y.shape[1],no_layers=no_layers,no_nodes=no_nodes)
    # create loss
    mse = tf.reduce_mean(0.5 * tf.square(output_predictions - output_data))
    # create optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)
    saver = tf.train.Saver()
    if create:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # create saver to save model variables
    else:
        saver.restore(sess,"./dag_experts/{}.ckpt".format(name))
    # run training
    batch_size = 128
    for training_step in range(iteration):
        # get a random subset of the training data
        indices = np.random.randint(low=0, high=x.shape[0], size=batch_size)
        input_batch = x[indices]
        output_batch = y[indices]
        # run the optimizer and get the mse
        _, mse_run = sess.run([opt, mse], feed_dict={input_data: input_batch, output_data: output_batch})
        # print the mse every so often
        if training_step % 1000 == 0:
            print('{0:04d} mse: {1:.10f}'.format(training_step, mse_run),end='\r')
            saver.save(sess, './dag_experts/{}.ckpt'.format(model_name))

def evaluate_dag(envname,model_name=None,no_nodes=[64,64],render=False,max_timesteps=1000,num_rollouts=20):
    if model_name is None:
        model_name = envname
    print('loading and building behavior cloning policy')
    in_ph,out_ph,out_pred,sess = load_dag_policy(model_name,game_shapes[envname][0],game_shapes[envname][1],no_nodes=no_nodes)
    print('loaded and built')
    observations =[]
    with sess:
        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit 
        returns = []
        for i in range(num_rollouts):
            print('iter', i,end='\r')
            obs = env.reset()
            observations.append(obs)
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action =sess.run(out_pred,feed_dict={in_ph:[obs]})
                obs, r, done, _ = env.step(action)
                observations.append(obs)
                totalr += r
                steps += 1
                if render:
                    env.render()
                #if steps % 100 == 0: print("%i/%i"%(steps, max_steps),end='\r')
                if steps >= max_steps:
                    break
            returns.append(totalr)
        return np.array(observations),np.mean(returns),np.std(returns)

def evaluate_model(envname,model_name=None,no_nodes=[64,64],render=False,max_timesteps=1000,num_rollouts=20):
    if model_name is None:
        model_name = envname
    print('loading and building behavior cloning policy')
    in_ph,out_ph,out_pred,sess = load_bc_policy(model_name,game_shapes[envname][0],game_shapes[envname][1],no_nodes=no_nodes)
    print('loaded and built')
    with sess:
        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action =sess.run(out_pred,feed_dict={in_ph:[obs]})
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps),end='\r')
                if steps >= max_steps:
                    break
            returns.append(totalr)
        return returns,np.mean(returns),np.std(returns)

def take_expert_opinion(envname,observations):
    policy_fn = load_policy.load_policy('experts/{}.pkl'.format(envname))
    with tf.Session():
        tf_util.initialize()
        actions = policy_fn(observations)
        return actions

def get_expert_data(envname,amount=1000):
    x,y = Read_Data(envname)
    x = x[0:1000]
    y = y[0:1000,0,:]
    return x,y
    
def train_model(name ='Ant-v2',model_name=None,no_layers=2,no_nodes=[64,64],iteration=100000):
    if model_name is None:
        model_name = name
    x,y = Read_Data(name)
    sess = tf_reset()
    input_data,output_data,output_predictions = create_model(x.shape[1],y.shape[2],no_layers=no_layers,no_nodes=no_nodes)
    # create loss
    mse = tf.reduce_mean(0.5 * tf.square(output_predictions - output_data))
    # create optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)
    # initialize variables
    sess.run(tf.global_variables_initializer())
    # create saver to save model variables
    saver = tf.train.Saver()
    # run training
    batch_size = 128
    for training_step in range(iteration):
        # get a random subset of the training data
        indices = np.random.randint(low=0, high=x.shape[0], size=batch_size)
        input_batch = x[indices]
        output_batch = y[indices,0]
        # run the optimizer and get the mse
        _, mse_run = sess.run([opt, mse], feed_dict={input_data: input_batch, output_data: output_batch})
        # print the mse every so often
        if training_step % 1000 == 0:
            print('{0:04d} mse: {1:.5f}'.format(training_step, mse_run),end='\r')
            saver.save(sess, './bc_experts/{}.ckpt'.format(model_name))

def evaluate_expert(envname,render=False,max_timesteps=1000,num_rollouts=3):
    policy_fn = load_policy.load_policy('experts/{}.pkl'.format(envname))
    print('loaded {} expert'.format(envname))

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit
        returns = []

        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps),end='\r')
                if steps >= max_steps:
                    break
            returns.append(totalr)
        return returns,np.mean(returns),np.std(returns)

import argparse

import gym
from replay_memory import ReplayMemory

import random
import numpy as np

from keras.layers import Dense,Input
from keras.models import Model
from keras.optimizers import RMSprop
import tensorflow as tf
def create_model(observation_space, action_space, args):
    assert isinstance(observation_space, gym.spaces.Box)
    assert isinstance(action_space, gym.spaces.Discrete)
    assert len(observation_space.shape) == 1

    # TODO: Create a model:
    #   - input to the model is the same as observation_space.shape,
    #   - output of the model is action_space.n Q-values
    # Use two fully-connected layers with 64 units and tanh activation.
    # Use mean squared error loss function and RMSprop optimizer.
    # YOUR CODE HERE

    x = Input(observation_space.shape)
    h = Dense(64,activation='tanh')(x)
    h = Dense(64,activation='tanh')(h)
    out= Dense(action_space.n)(h)
    model = Model(inputs=[x],outputs=[out])
    model.compile(optimizer=RMSprop(lr=args.learning_rate) ,loss='mse')

    return model


def train_model(model, target_model, batch, args):
    obs, actions, rewards, dones, obs_next = batch
    # TODO: Train a model:
    #   1. Do feedforward pass for this observation with main model and for next observation with target model.
    #   2. Calculate Q-value regression targets, cutting the returns at episode end.
    #      if episode end:
    #        Q(s,a) = r(s,a)
    #      else:
    #        Q(s,a) = r(s,a) + gamma * max_a' Q'(s', a')
    #      NB! You are working with vectors, so try to do away without for loops!
    #   3. Set the targets for Q-values of chosen actions.
    #      NB! For other Q-values you might need to set their targets the same as their original values, so the error is zero.
    #   4. Train the main model with one gradient update. Return loss and mean Q-value for statistics.
    #   5. Implement Double Q-learning:
    #      Q(s,a) = r(s,a) + gamma * Q'(s', argmax_a' Q(s', a'))
    #      NB! You need to do additional forward pass for this!
    # YOUR CODE HERE

    qobs = model.predict(obs)
    q_obs_next = target_model.predict(obs_next)
    test = qobs.copy()
    qobs[np.arange(qobs.shape[0]),actions]= \
            rewards+args.gamma*np.amax(q_obs_next,axis=1)*np.logical_not(dones)
    qmean = np.mean(qobs)
    loss = model.train_on_batch(obs,qobs)

    return loss, qmean


def update_target(model, target_model):
    # TODO: Copy main model weights to target model.

    # YOUR CODE HERE
    w = model.get_weights()
    target_model.set_weights(w)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument('--exp_name', type=str, default='cartpole')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_timesteps', type=int, default=100000)
    parser.add_argument('--replay_size', type=int, default=100000)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.1)
    parser.add_argument('--epsilon_steps', type=int, default=10000)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--learning_start', type=int, default=1000)
    parser.add_argument('--update_freq', type=int, default=10)
    parser.add_argument('--train_freq', type=int, default=1)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--log_freq', type=int, default=1000)
    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # create environment and add standard wrappers
    env = gym.make(args.env_name)

    # create experiment directory
    #logdir = os.path.join('data', args.exp_name + '_' + args.env_name, str(e))
    #if not os.path.exists(logdir):
    #    os.makedirs(logdir)

    # open results file
    #csvfile = open(os.path.join(logdir, 'log.txt'), 'w')
    #csvwriter = csv.writer(csvfile, delimiter='\t')
    #csvwriter.writerow(["Time", "Iteration", "AverageReturn", "StdReturn", "MaxReturn", "MinReturn",
                      # "EpLenMean", "EpLenStd", "TimestepsThisBatch", "TimestepsSoFar"])

    # write params file
    #with open(os.path.join(logdir, 'params.json'), 'w') as f:
    #    json.dump(vars(args), f)

    # create main model and target model.
    model = create_model(env.observation_space, env.action_space, args)
    model.summary()
    target_model = create_model(env.observation_space, env.action_space, args)
    # copy main model weights to target
    update_target(model, target_model)

    # create replay memory
    replay_memory = ReplayMemory(args.replay_size, env.observation_space.shape)

    # statistics
    loss = 0
    qmean = 0
    rewards = []
    lengths = []
    episode_num = 0
    episode_reward = 0
    episode_length = 0
    num_iterations = 0

    # reset the environment
    obs = env.reset()
    # loop for args.num_timesteps steps
    for t in range(args.num_timesteps):
        # calculate exploration rate
        if t < args.epsilon_steps:
            # anneal epsilon linearly from args.epsilon_start to args.epsilon_end
            epsilon = args.epsilon_start - (t / args.epsilon_steps) * (args.epsilon_start - args.epsilon_end)
        else:
            # after ars.epsilon_steps steps set exploration to fixed rate
            epsilon = args.epsilon_end

        # TODO: Choose action:
        #  - with probability epsilon choose random action,
        #  - otherwise choose action with maximum Q-value.

        # YOUR CODE HERE
        if np.random.random()<=epsilon:
            action = np.random.randint(0,env.action_space.n)
        else:
            action = np.argmax(model.predict(obs[np.newaxis,:]))

        # step environment
        next_obs, reward, done, info = env.step(action)
        if args.render:
            env.render()

        # TODO: Add current experience to replay memory

        # YOUR CODE HERE
        replay_memory.add(obs,action,reward,done,next_obs)

        # statistics
        episode_reward += reward
        episode_length += 1
        # if episode ended
        if done:
            # reset environment
            obs = env.reset()

            # statistics
            episode_num += 1
            rewards.append(episode_reward)
            lengths.append(episode_length)
            episode_reward = 0
            episode_length = 0
        else:
            # otherwise add new observation to history
            obs = next_obs

        # training
        if t >= args.learning_start and t % args.train_freq == 0:
            # TODO: Train model:
            #  - sample minibatch from replay memory
            #  - train model with that minibatch

            # YOUR CODE HERE
            batch = replay_memory.sample(args.batch_size)
            loss,qmean = train_model(model,target_model,batch,args)
            num_iterations += 1
            # update target model after fixed number of iterations
            if num_iterations % args.update_freq == 0:
                update_target(model, target_model)

            # save model after fixed number of iterations
            if num_iterations % args.save_freq == 0:
                # TODO: save model
                model.save('data/model_{:.2f}_{:.2f}_{}.h5'.format(loss,qmean,num_iterations))

        # log statistics
        if (t + 1) % args.log_freq == 0:
            print("Timestep %d: episodes %d, reward %f, length %f, iterations: %d, epsilon: %f, loss: %f, Q-mean: %f, replay: %d" %
                  (t + 1, len(rewards), np.mean(rewards), np.mean(lengths), num_iterations, epsilon, loss, qmean, replay_memory.count))
            rewards = []
            lengths = []

    print("Done")

"""
This is an TensorFlow Keras (TF2) implementaion of a vanilla Deep Q Leaning algorithm to play the
OpenAI Gym Atari Games. 
I made this for excercise only. kredits for all Core parts go to https://github.com/danielegrattarola/deep-q-atari
However, it is built to run in VS-Code or Jupyter Notebook and rewritten for Python 3.7 and 
TF.Keras nomenclature.
Hyper parameters are optimized for learning pong in accorance with move37 course: 
https://github.com/colinskow/move37/tree/master/dqn
Requires:
Python 3.7
Tensorflow >= 1.12
gym
gym['atari']
pillow
"""

# %%
# --------------------------------IMPORTS------------------------------------------
from PIL import Image
from random import randrange, randint
import numpy as np
import tensorflow as tf
import gym
import numpy as np
import random
import logging


# %%
# -----------------------------DQN_MODEL CLASS DEF----------------------------------
class DQN_Model:
    def __init__(self, actions, input_shape,
                 minibatch_size=32,
                 learning_rate=0.00025,
                 discount_factor=0.99,
                 dropout_prob=0.1,
                 load_path=None,
                 logger=None):

        # Parameters
        self.actions = actions  # Size of the network output
        self.discount_factor = discount_factor  # Discount factor of the MDP
        self.minibatch_size = minibatch_size  # Size of the training batches
        self.learning_rate = learning_rate  # Learning rate
        self.dropout_prob = dropout_prob  # Probability of dropout
        self.logger = logger
        self.training_history_csv = 'training_history.csv'

        # if self.logger is not None:
        #    self.logger.to_csv(self.training_history_csv, 'Loss,Accuracy')

        # Deep Q Network as defined in the DeepMind article on Nature
        # Ordering channels first: (samples, channels, rows, cols)
        self.model = tf.keras.Sequential()

        # First convolutional layer
        self.model.add(tf.keras.layers.Conv2D(32, 8, strides=(4, 4),
                                              padding='valid',
                                              activation='relu',
                                              input_shape=input_shape,
                                              data_format='channels_first'))

        # Second convolutional layer
        self.model.add(tf.keras.layers.Conv2D(64, 4, strides=(2, 2),
                                              padding='valid',
                                              activation='relu',
                                              input_shape=input_shape,
                                              data_format='channels_first'))

        # Third convolutional layer
        self.model.add(tf.keras.layers.Conv2D(64, 3, strides=(1, 1),
                                              padding='valid',
                                              activation='relu',
                                              input_shape=input_shape,
                                              data_format='channels_first'))

        # Flatten the convolution output
        self.model.add(tf.keras.layers.Flatten())

        # First dense layer
        self.model.add(tf.keras.layers.Dense(512, activation='relu'))

        # Output layer
        self.model.add(tf.keras.layers.Dense(self.actions))

        # Load the network weights from saved model
        if load_path is not None:
            self.load(load_path)

        self.model.compile(loss='mean_squared_error',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def train(self, batch, DQN_target):
        # Generate training inputs (X) and targets (Y)
        X = []
        Y = []
        for datapoint in batch:
            # Inputs are the states
            X.append(datapoint['source'].astype(np.float64))

            # Y is either reward (if done) or the famous Q learning formula
            target = datapoint['reward']
            if not datapoint['final']:
                next_Q_value = np.max(DQN_target.predict(
                    datapoint['dest'].astype(np.float64)))
                target = datapoint['reward'] + \
                    self.discount_factor * next_Q_value
            Y_temp = self.model.predict(datapoint['source'].astype(np.float64))
            Y_temp[0][datapoint['action']] = target
            Y.append(Y_temp)

        # prepare for NN fitting
        X = np.asarray(X).squeeze()
        Y = np.asarray(Y).squeeze()

        # Holy shit, finally the NN gets to do it's job! Fit new data (backpropagation, sgd n shit)
        history = self.model.fit(
            X,
            Y,
            batch_size=self.minibatch_size,
            epochs=1,
            verbose=0
        )

        # Log loss and accuracy
        # if self.logger is not None:
        #    self.logger.to_csv(self.training_history_csv,
        #                       [history.history['loss'][0], history.history['acc'][0]])

    def predict(self, state):
        # a forward pass though the network
        return self.model.predict(state.astype(np.float64), batch_size=1)

    def save(self, filename=None, append=''):
        """
        Saves the model weights to disk.
        :param filename: file to which save the weights (must end with ".h5")
        :param append: suffix to append after "model" in the default filename
            if no filename is given
        """
        f = ('model%s.h5' % append) if filename is None else filename
        if self.logger is not None:
            self.logger.info(f'Saving model as {f}')
        self.model.save_weights(self.logger.path + f)

    def load(self, path):
        """
        Loads the model's weights from path.
        :param path: h5 file from which to load teh weights
        """
        if self.logger is not None:
            self.logger.info('Loading weights from file...')
        self.model.load_weights(path)


# %%
# -----------------------------DQN_AGENT CLASS DEF---------------------------------
class DQN_Agent:
    def __init__(self,
                 actions,
                 network_input_shape,
                 replay_memory_size=1024,
                 minibatch_size=32,
                 learning_rate=0.00025,
                 discount_factor=0.9,
                 dropout_prob=0.1,
                 epsilon=1,
                 epsilon_decrease_rate=0.99,
                 min_epsilon=0.1,
                 load_path=None,
                 logger=None):

        # Parameters
        self.network_input_shape = network_input_shape  # Shape of the DQN input
        self.actions = actions  # Size of the discrete action space
        self.learning_rate = learning_rate  # Learning rate for the DQN
        self.dropout_prob = dropout_prob  # Dropout probability of the DQN
        self.load_path = load_path  # Path from which to load the DQN's weights
        self.replay_memory_size = replay_memory_size  # Size of replay memory
        self.minibatch_size = minibatch_size  # Size of a DQN minibatch
        self.discount_factor = discount_factor  # Discount factor of the MDP
        self.epsilon = epsilon  # Probability of taking a random action
        self.epsilon_decrease_rate = epsilon_decrease_rate  # See update_epsilon
        self.min_epsilon = min_epsilon  # Minimum value for epsilon
        self.logger = logger

        # Replay memory
        self.experiences = []
        self.training_count = 0

        # Instantiate the deep Q-networks
        # Main DQN
        self.DQN = DQN_Model(
            self.actions,
            self.network_input_shape,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            minibatch_size=self.minibatch_size,
            dropout_prob=self.dropout_prob,
            load_path=self.load_path,
            logger=self.logger
        )

        # Target DQN used to generate targets
        self.DQN_target = DQN_Model(
            self.actions,
            self.network_input_shape,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor,
            minibatch_size=self.minibatch_size,
            dropout_prob=self.dropout_prob,
            load_path=self.load_path,
            logger=self.logger
        )

        # DQN and Target_DQN should have the same weights. They will also sync from time to time.
        self.DQN_target.model.set_weights(self.DQN.model.get_weights())

    def get_actions(self, state):
        # returns an action for the state using epsion greedy for the balace of exploitation and exploration
        if random.random() < self.epsilon:
            return(randint(0, self.actions - 1))
        else:
            q_values = self.DQN.predict(state)
            return np.argmax(q_values)

    def get_max_q(self, state):
        # return the maximum Q value predicted on the given state
        q_values = self.DQN.predict(state)
        idxs = np.argwhere(q_values == np.max(q_values)).ravel()
        return np.random.choice(idxs)

    def get_random_state(self):
        # Returns a random state from the replay buffer
        return self.experiences[randrange(0, len(self.experiences))]['source']

    def add_experience(self, source, action, reward, dest, final):
        """ 
        Add SARS' (Actually SARS'D) touple to experience buffer.
        """
        # Treat like a buffer. collection.deque could be used here as well. Faster?
        if len(self.experiences) >= self.replay_memory_size:
            self.experiences.pop(0)
        # Add a tuple (source, action, reward, dest, final) to replay memory
        self.experiences.append({'source': source,
                                 'action': action,
                                 'reward': reward,
                                 'dest': dest,
                                 'final': final})
        # Periodically log how many samples we've gathered so far
        if (len(self.experiences) % 1000 == 0) and (len(self.experiences) < self.replay_memory_size) and (self.logger is not None):
            self.logger.info(
                f"Gathered {len(self.experiences)} samples of {self.replay_memory_size}")

    def sample_batch(self):
        # Samples self.minibatch_size random transitions from replay memory
        batch = []
        for _ in range(self.minibatch_size):
            batch.append(self.experiences[randrange(0, len(self.experiences))])
        return batch

    def train(self):
        # trains DQN on a minibatch of transitions.
        # This passes a batch and the helper-DQN to the DQN.train method in the DQN Class.
        self.training_count += 1
        if self.training_count%1000 == 0:
            logger.info(f"Traing session: {self.training_count}  -  epsion: {self.epsilon}")
        batch = self.sample_batch()
        self.DQN.train(batch, self.DQN_target)

    def update_epsilon(self):
        """
        Decreases the probability of picking a random action, to improve
        exploitation.
        """
        if self.epsilon - self.epsilon_decrease_rate > self.min_epsilon:
            self.epsilon -= self.epsilon_decrease_rate
        else:
            self.epsilon = self.min_epsilon

    def reset_target_network(self):
        """
        Updates the target DQN with the current weights of the main DQN.
        """
        if self.logger is not None:
            self.logger.info('Updating target network...')
        self.DQN_target.model.set_weights(self.DQN.model.get_weights())

    def quit(self):
        """
        Saves the DQN and the target DQN to file.
        """
        if self.load_path is None:
            if self.logger is not None:
                self.logger.info('Quitting...')
            self.DQN.save(append='_DQN')
            self.DQN_target.save(append='_DQN_target')


# %%
# ----------------------------------FRAME PREP-------------------------------------
""" 
Mutations for the frames. This model / agent works with compose frame method. 
4 Frames are made black and white, shrunk and each of the 4 frames is used as a chanel. This is
enough to let the NN understand "what is happening right now". 
LSTM would be another way but much more complex.
"""

IMG_SIZE = None  # Set in main. Needs to be in line with the the input for the neural net!!!


def preprocess_observation(obs):
    # Convert to gray-scale and resize it
    image = Image.fromarray(obs, 'RGB').convert('L').resize(IMG_SIZE)
    # Convert image to array and return it
    return np.asarray(image.getdata(), dtype=np.uint8).reshape(image.size[1],
                                                               image.size[0])


def get_next_state(current, obs):
    # Next state is composed by the last 3 images of the previous state and the
    # new observation
    return np.append(current[1:], [obs], axis=0)


# %%
# -------------------------------------PARAMETERS---------------------------------
# Here: Learn to play pong. Best hyper parameters are differnt for other games.
ENVIRONMENT = "PongNoFrameskip-v4"
IMG_SIZE = (84, 110)
TRAIN = True
LOAD = None
VIDEO = 10  # number of episodes when to render. Set to 0 for never.
DEBUG = False
MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 10000
TARGET_NETWORK_UPDATE_FREQUENCY = 1000
#AVG_VAL_COMPUTATION_FREQENCY = 5e4
DISCOUNT_FACTOR = 0.99
UPDATE_FREQUENCY = 4
LEARNING_RATE = 1e-4
EPSILON = 1
MIN_EPSIOLON = 0.02
EPSILON_DECREASE = 5e-5
REPLAY_START_SIZE = 5e4
INITIAL_RANDOM_ACTIONS = 30
DROPOUT = 0.0
MAX_EPISODES = np.inf
MAX_EPISODE_LENGTH = np.inf
MAX_FRAMES_NUMBER = 5e7
TEST_FREQUENCY = 250000
VALISATION_FRAMES = 135000
TEST_STATES = 30

MEAN_REWARD_GOAL = 19.5

# Initial logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# %%
# ---------------------------------------------MAIN-------------------------------


# Setup
env = gym.make(ENVIRONMENT)
network_input_shape = (4, 110, 84)  # Dimension ordering: 'th' (channels first)
DQA = DQN_Agent(env.action_space.n,
                network_input_shape,
                replay_memory_size=REPLAY_MEMORY_SIZE,
                minibatch_size=MINIBATCH_SIZE,
                learning_rate=LEARNING_RATE,
                discount_factor=DISCOUNT_FACTOR,
                dropout_prob=DROPOUT,
                epsilon=EPSILON,
                epsilon_decrease_rate=EPSILON_DECREASE,
                min_epsilon=MIN_EPSIOLON,
                load_path=LOAD,
                logger=logger)

# Set Counter
episode = 0
frame_counter = 0
total_rewards = []

while episode < MAX_EPISODES:
    # Log
    logger.info(f"Episode {episode}")
    score = 0

    # Observe reward and initialize first state
    obs = preprocess_observation(env.reset())
    current_state = np.array([obs, obs, obs, obs])

    # Main Loop
    t = 0
    frame_counter += 1
    while t < MAX_EPISODE_LENGTH:
        # Stop if it takes too long.
        if frame_counter > MAX_FRAMES_NUMBER:
            DQA.quit()

        # Render
        if VIDEO != 0 and episode % VIDEO == 0:
            env.render()

        # Ask the DQN Neural Net for the next action
        action = DQA.get_actions(np.asarray([current_state]))

        # SARS'
        obs, reward, done, info = env.step(action)
        obs = preprocess_observation(obs)

        next_state = get_next_state(current_state, obs)
        frame_counter += 1

        # add SARS' to buffer
        clipped_reward = np.clip(reward, -1, 1)  # Clip the reward
        DQA.add_experience(np.asarray([current_state]),
                           action,
                           clipped_reward,
                           np.asarray([next_state]),
                           done)

        # Train the agent
        if t % UPDATE_FREQUENCY == 0 and len(DQA.experiences) >= REPLAY_START_SIZE:
            DQA.train()
            # from time to time update DQA target
            if DQA.training_count % TARGET_NETWORK_UPDATE_FREQUENCY == 0 and DQA.training_count >= TARGET_NETWORK_UPDATE_FREQUENCY:
                DQA.reset_target_network()
            # TODO: add logging of the mean score and the mean Q values here

        # Update epsion
        if len(DQA.experiences) > REPLAY_START_SIZE:
            DQA.update_epsilon()

        # Update current state and score
        current_state = next_state
        score += reward
        t += 1

        #if episode done then log and break loop
        if done or t == MAX_EPISODE_LENGTH - 1:
            logger.info(f"Done. Lenght:{t+1} Score:{score}")
            break

        # Calc and print mean reward (over last 1000 frames)
        total_rewards.append(reward)
        mean_reward = np.mean(total_rewards[-1000:]) #
        if frame_counter % 5000 == 0:
            print(
                f"Main Loop: episode: {episode}  t: {t}  frame: {frame_counter}  mean_reward: {mean_reward}")

    episode += 1

import numpy as np
from numpy import random as rnd
import random
from matplotlib import pyplot as plt
import torch
import torch.nn

# ## First, tabular Q-learning
# game from Shaked Zychlinski
#
# https://github.com/shakedzy/notebooks/blob/master/q_learning_and_dqn/Q%20Learning%20and%20Deep%20Q%20Network.ipynb
#
# and structure inspired by Gaston Sivori, Okinawa Institute of Science and Technology
class Environment:
    """
    A generalizable class for board environments for Qlearning agents
    """

    board = None
    board_size = 0

    def __init__(self, board_size=4):
        self.board_size = board_size
        self.game_over = False
        self.terminal_state = np.ones(4).astype(int)
        self.reset()

    def reset(self):  # new game
        self.state = np.zeros(self.board_size).astype(
            int
        )  # [0,0,0,0,...0]: an empty board
        self.game_over = False

    def execute_action(self, action):  # execute the agent's action
        if self.state[action] == 0:
            self.state[action] = 1
            self.game_over = len(np.where(self.state == 0)[0]) == 0
            return 1
        else:
            return -1


class Agent:
    """
    A generalizable calss for Q learning agents in our game environment
    """

    def __init__(self, board_size):
        # this qdim definition looks a bit hairy, but is just saying we want a qtable w/ dimensions like [2,2,2,2,4]
        # meaning that there are boardsize binary states and boardsize actions we can take in the last index
        qdim = tuple([2] * board_size) + (board_size,)
        self.Q = np.zeros(
            qdim
        )  # just make 4x1 games for now: 4 actions per 2^4 possible game states
        self.epsilon = 0.9  # exploration rate
        self.gamma = 0.9  # discount
        self.lr = 0.1  # learning rate

    def select_action(self, state):
        if rnd.rand() < self.epsilon:  # take a greedy action
            return np.argmax(self.Q[tuple(state)])
        else:
            return random.choice(list(range(board_size)))  # take a random action

    def greedy_action(self, state):
        return np.argmax(self.Q[tuple(state)])

    def update(self, old_state, new_state, action, reward):
        q_old = self.Q[tuple(old_state) + (action,)]  # Old Q value
        future_action = self.greedy_action(new_state)  # Select next best action

        EV_new = self.Q[
            tuple(new_state) + (future_action,)
        ]  # What is reward for the best next action?

        if sum(new_state) == board_size:
            EV_new = 0  # we are in a terminal state, so EV not a meaningful value

        rpe = self.lr * (reward + self.gamma * EV_new - q_old)

        self.Q[tuple(old_state) + (action,)] += rpe  # update


class RLInterface:
    """
    A class that brings the Qlearning agent together with its environment
    """

    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment
        self.rewlist = []

    def step(self):  # advance one timestep
        old_state = env.state.copy()  # take a copy of the agent's current state

        action = self.agent.select_action(old_state)  # agent selects action
        rew = self.env.execute_action(
            action
        )  # execute agent action into the environment
        new_state = (
            env.state.copy()
        )  # find out what state this brought us to for new expected value approximation

        rpe = self.agent.update(old_state, new_state, action, rew)  # update Q

        return rew

    def runTrials(self, nTrials):  # run the agent and environment through nTrials games
        for i in range(nTrials):
            env.reset()
            total_rew = 0
            while not self.env.game_over:
                rew = self.step()
                total_rew += rew
            self.rewlist.append(total_rew)


board_size = 4  # default board size is 4
agent = Agent(board_size)
env = Environment(board_size)
rl = RLInterface(
    agent, env
)  # give the interface an agent and environment to bring together

rl.runTrials(1000)

plt.title("Reward Over time")
plt.xlabel("Game Number")
plt.ylabel("Rewards Received in Game")
plt.plot(rl.rewlist)

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F


# just a feed forward neural network to estimate Q(s,a) values
class DQN(nn.Module):
    def __init__(self, envstate_dim, action_dim):
        super(DQN, self).__init__()
        self.input_dim = envstate_dim
        self.output_dim = action_dim

        self.ff = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 124),
            nn.ReLU(),
            nn.Linear(124, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim),
        )

    def forward(self, state):
        qvals = self.ff(state)
        return qvals


# replay buffers implemented as lists. this is actually recommended by Python over Deques for random retrieval
class Buffer:
    def __init__(self):
        self.buffer = []

    def size(self):
        return len(self.buffer)

    # add a memory
    def push(self, state, action, new_state, reward):
        experience = (state, action, new_state, reward)
        self.buffer.append(experience)

    # take a random sample to perform learning on decorrelated transitions
    def sample(self, batch_size):
        batchSample = random.sample(self.buffer, batch_size)
        # now need to put everyone in the correct columns
        state_batch = []
        action_batch = []
        new_state_batch = []
        reward_batch = []

        # prepare the batch sample for training
        for experience in batchSample:
            state, action, new_state, reward = experience
            state_batch.append(state)
            action_batch.append(action)
            new_state_batch.append(new_state)
            reward_batch.append(reward)
        return (state_batch, action_batch, reward_batch, new_state_batch)


# a class for agents that use feedforward neural networks to calculate Q(s,a)
class DeepAgent:
    def __init__(self, board_size):
        self.policy_net = DQN(
            board_size, board_size
        )  # network used to calculate policy
        self.target_net = DQN(
            board_size, board_size
        )  # network used to calculate target
        self.target_net.eval()  # throw that baby in eval mode because we don't care about its gradients
        self.target_update = 50  # update our target network ever 50 timesteps
        self.replay_buffer = Buffer()  # replay buffer implemented as a list
        self.eps_start = 0.1  # initial exploration rate
        self.eps_end = 0.95  # ultimate exploration value
        self.eps_decay = 300  # decay parameter for exploration rate
        self.epsilon = self.eps_start  # initialize epsilon
        self.gamma = 0.99  # discount

        self.optimizer = torch.optim.SGD(
            self.policy_net.parameters(), lr=0.01, momentum=0.9
        )
        #         self.optimizer = torch.optim.RMSprop(self.policy_net.parameters()) # experiment w/ different optimizers
        #         self.optimizer = torch.optim.Adam(self.policy_net.parameters())
        self.huber_loss = F.smooth_l1_loss

    def select_action(self, state):
        state = torch.FloatTensor(state).float()
        if rnd.rand() < self.epsilon:  # greedy action
            with torch.no_grad():
                qvals = self.policy_net.forward(
                    state
                )  # forward run through the policy network
                action = np.argmax(
                    qvals.detach().numpy()
                )  # need to detach from auto_grad before sending to numpy
        else:
            action = random.choice(list(range(board_size)))
        return action

    def update(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return
        batch = self.replay_buffer.sample(batch_size)

        self.optimizer.zero_grad()  # zero_grad before computing loss

        loss = self.compute_loss(batch)

        loss.backward()  # get the gradients

        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()  # backpropagate

        return loss

    def compute_loss(self, batch):
        states, actions, rewards, next_states = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        curr_Q = self.policy_net.forward(states).gather(
            1, actions.unsqueeze(1)
        )  # calculate the current Q(s,a) estimates
        next_Q = self.target_net.forward(next_states)  # calculate Q'(s,a) (EV)
        max_next_Q = torch.max(next_Q, 1)[0]  # equivalent of taking a greedy action
        expected_Q = rewards + self.gamma * max_next_Q  # Calculate total Q(s,a)

        loss = self.huber_loss(
            curr_Q, expected_Q.unsqueeze(1)
        )  # unsqueeze is really important here to match dims!
        return loss


class DeepRLInterface:
    def __init__(self, agent, environment):
        self.agent = agent
        self.env = environment
        self.rewlist = []
        self.batch_size = 50  # sample 50 experiences when we update

    def step(self):  # same process as above
        state = env.state.copy()
        action = self.agent.select_action(state)  # agent selects action
        rew = self.env.execute_action(
            action
        )  # execute agent action into the environment
        new_state = env.state.copy()
        if not np.all(
            rl.env.state == rl.env.terminal_state
        ):  # don't add terminal states to replay buffer
            self.agent.replay_buffer.push(state, action, new_state, rew)

        loss = self.agent.update(self.batch_size)
        self.losslist.append(loss)  # append loss to assess performance over time
        return state, action, rew, new_state

    def runTrials(self, nTrials):
        counter = 0  # for batch training
        self.clist = []
        self.rewlist = []
        self.losslist = []
        self.eps = []
        for i in range(nTrials):
            env.reset()
            total_rew = 0
            tstates, tactions, trews, tnewstates = (
                [],
                [],
                [],
                [],
            )  # accumulate states to debug
            while (
                not self.env.game_over
            ):  # while the game is not over, keep taking actions
                state, action, rew, new_state = self.step()
                total_rew += rew

                tstates.append(state)
                tactions.append(action)
                trews.append(rew)
                tnewstates.append(tnewstates)
                counter += 1

            self.rewlist.append(total_rew)

            if counter % self.agent.target_update == 0:  # update the target network
                self.agent.target_net.load_state_dict(
                    self.agent.policy_net.state_dict()
                )
            # update agent epsilon
            self.agent.epsilon = self.agent.eps_end + (
                self.agent.eps_start - self.agent.eps_end
            ) * np.exp(-1.0 * counter / self.agent.eps_decay)
            self.eps.append(self.agent.epsilon)


# %%
board_size = 4
agent = DeepAgent(board_size)
env = Environment(board_size)
rl = DeepRLInterface(agent, env)

rl.runTrials(500)


# %%


loss_list = [el for el in rl.losslist]
loss_list_ = []
for el in loss_list:
    if type(el) == torch.Tensor:
        el = float(el.detach().numpy())
    loss_list_.append(el)

rew_list = [el for el in rl.rewlist]
rew_list_ = []
for el in rew_list:
    if type(el) == torch.Tensor:
        el = float(el.detach().numpy())
    rew_list_.append(el)

plt.title("Batch Loss")
plt.plot(loss_list_)
plt.xlabel("Timestep")
plt.ylabel("Loss")
plt.figure()
plt.title("Game Rewards")
plt.xlabel("Game")
plt.ylabel("Accumulated Reward")
plt.plot(rew_list_)

# Q values check out (this is for a boardsize = 4 case)
for state in [
    [0, 1, 1, 1],
    [1, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [1, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
]:
    q = list(rl.agent.policy_net(torch.FloatTensor(state)).detach().numpy())
    argmax = np.argmax(
        list(rl.agent.policy_net(torch.FloatTensor(state)).detach().numpy())
    )
    print("Q values for state ", state, ": ", q, "-> action", argmax)

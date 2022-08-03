"""
    This file contains the helper methods used for the calculations and plotting in tic_tac_toe.ipynb
"""

### -------------- Imports -------------- ###

import os
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from tic_env import TictactoeEnv, OptimalPlayer

# Deep Q-learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import DQN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Set seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


### -------------- Constants -------------- ###

ALPHA = 0.05
GAMMA = 0.99
EPSILON = 0.3 # Arbitrary
DELTA = 1 # For Huber loss
E_MIN = 0.1
E_MAX = 0.8

# Deep Q-learning
BATCH_SIZE = 64 # Take batches of 64
TARGET_UPDATE = 500 # Update every 500 game

# Type
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


### -------------- Plotting -------------- ###

def plot_subplots(rows, cols, x, ys, hyperparam, suptitle, 
                  xlabel, ylabel, labels=[], legend_title='', subtitle=''):
    """ Function for plotting subplots
        Args: 
            (rows, cols): (Int, Int). Subplot format.
            x: Iterable[Int]. Values of x-axis.
            ys: ndarray[Int] of shape (num_subplots, num_labels, len(x)). Values to be plotted.
                Example: If plotting both M_opt and M_rand in each subplot, num_labels=2
            hyperparam: Iterable[Float]. Hyperparameter that differs over subplots.
            suptitle: String. Super title.
            xlabel: String. Label of x-axis.
            ylabel: String. Label of y-axis.
            labels: Iterable[String] of length ys.shape[1]. Optional labels.
            legend_title: String. Optional legend title.
            subtitle: String. Optional (generic) subtitle of each subplot, is combined with hyperparam. 
    """

    fig, axs = plt.subplots(rows, cols, figsize=(18, 14), sharex=True, sharey=True)
    axs = axs.flatten()

    # Reveal x-ticks and y-ticks on all subplots -> Doesnt work
    for ax in axs:
        for (x_tick, y_tick) in zip(ax.get_xticklabels(), ax.get_yticklabels()):
            x_tick.set_visible(True)
            y_tick.set_visible(True)

    # Plot
    plt.suptitle(suptitle, fontsize=30)
    for i, (hp, y) in enumerate(zip(hyperparam, ys)):
        for reward in y:   
            sns.lineplot(x=x, y=reward, ax=axs[i])
        axs[i].set_title(f'{subtitle} = {hp}', fontsize=24)
        if labels != []:
            axs[i].legend(title=legend_title, labels=labels, loc='lower right')

    fig.text(0.5, 0.04, xlabel, ha='center', fontsize=26)
    fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical', fontsize=26)


### -------------- Q-Learning -------------- ###

# Thanks to Jeremy Zhang for the idea of how to encode the state
# https://towardsdatascience.com/reinforcement-learning-implement-tictactoe-189582bea542
def encode(S, A):
    ''' Unique encoder for a given state-action pair \n
        Args: 
            S: TictactoeEnv. Represents the current state.
            A: Tuple[Int] or Int. Represents the action to be taken.
        Returns:
            String. Unique representation of (S, A)
    '''
    if isinstance(A, tuple):
        A = A[0] * 3 + A[1]
    return f'{S.grid.reshape(9)}:{A}'

def get_legal_moves(S):
    ''' Helper for getting legal moves given state S \n
        Args:
            S: TictactoeEnv. Current state.
        Returns: 
            List[Int]. Legal moves.
    '''
    legal_moves = []
    for i in range(9):
        if S.check_valid(i):
            legal_moves.append(i)
    return legal_moves


def choose_action(S, e, Q, player='X'):
    """ Choose next action given current state, epsilon and saved states \n
        Args: 
            S: TictactoeEnv. Current state.
            e: Float. Epsilon value in range [0, 1].
            Q: Dict[String, Float]. Learned Q function.
        Returns: 
            Int. A legal action.

    """
    actions = get_legal_moves(S)
    if np.random.uniform(0, 1) <= e:
        i = np.random.choice(len(actions))
        return actions[i]
    else:
        value_max = -9999
        for a in actions:
            # Get Q(S, a) from saved, or 0 if not encountered
            value = Q.get(encode(S, a), 0)
            if value > value_max:
                value_max = value
                A = a
        return A


### -------------- Deep Q-Learning -------------- ###

def grid2tensor(grid, player = 'X'):
    """ Helper for converting numpy grid to tensor grid
        Args:
            grid: ndarray[Int]. Grid representation of current state.
            player: Char. Player to compute the tensor grid for. 
        Returns: 
            torch.tensor[Int]. Tensor representation of grid
    """
    # Returning None if Grid is None
    if grid is None:
        return None

    result_p = np.zeros((3, 3))
    result_o = np.zeros((3, 3))
    # Convert player to integer
    player_value = 1 if player == 'X' else -1
    # Retrieve indices of both players positions
    player_mask = np.where(grid == player_value)
    opposition_mask = np.where(grid == - player_value)
    # Insert values into final result
    result_p[player_mask] = 1
    result_o[opposition_mask] = 1
    result = np.array([result_p, result_o])
    # Reshape to shape expected by neural network and convert to tensor
    return torch.from_numpy(result).type(torch.float32)[None]


# The following function: optimize_model is a direct copy from the Pytorch DQN tutorial.
# Information can be found here: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html. 
# We also use inspiration from this guide: https://mahowald.github.io/pytorch-dqn/ in our implmentation of the DQN
def optimize_model(policy_net, target_net, memory, optimizer, batch_size):
    if len(memory) < batch_size:
        return # Do nothing

    # Get batch-array of transitions, and convert to Transition of batch-arrays
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Mask of non-final states
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), 
        device=device
    )

    # Non-final states
    if batch_size > 1:
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    # Unpack batch
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Action taken from policy for each state
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute values for next state based on target_net, zero if final
    next_state_values = torch.zeros(batch_size, device=device)
    if batch_size > 1:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()


    # Expected Q--values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss (delta defaults to 1)
    criterion = nn.HuberLoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))


    # Optimize
    optimizer.zero_grad()
    loss.backward()
    for i, param in enumerate(policy_net.parameters()):
        if param.grad is not None:
            param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.detach()


# function is from: https://mahowald.github.io/pytorch-dqn/
# with minor adjustments
def select_model_action(
    device: torch.device, model: DQN, state: torch.tensor, eps: float, S: TictactoeEnv
):
    """Selects an action for the model: either using the policy, or
    by choosing a random valid action (as controlled by `eps`)
    
    Arguments:
        device {torch.device} -- Device
        model {Policy} -- Policy module
        state {torch.tensor} -- Current board state, as a torch tensor
        eps {float} -- Probability of choosing a random state.
    
    Returns:
        Tuple[torch.tensor, bool] -- The action, and a bool indicating whether
                                     the action is random or not.
    """

    sample = random.random()
    if sample >= eps:
        return model.act(state)
    else:
        # per the announcement "A CLARIFICATION FOR THE MINI-PROJECTS"
        # we sample the action uniformly in legal actions for the exploratory actions
        return (
            torch.tensor(
                [[random.choice(get_legal_moves(S))]],
                device=device,
                dtype=torch.long,
            ))


def push_to_memory(memory, state, A, next_state, reward, flip):
    """ Helper for pushing to memory "from both sides of the table" \n
        Args:
            memory: ReplayMemory. Memory to push to.
            state: torch.tensor[Int]. State before taking action.
            A: torch.tensor[Int]. Action that was taken. 
            next_state: torch.tensor[Int]. Next state resulting from action A, or None if end of game
            reward: Int. Reward for taking action A.
            flip: Boolean. True if we flip the perspective of the player. 
    """
    if flip:
        next = None if next_state is None else torch.flip(next_state, (1, )) 
        memory.push(torch.flip(state, (1, )), A, next, 
                    torch.tensor([reward], device=device))
    else:
        memory.push(state, A, next_state, torch.tensor([reward], device=device))
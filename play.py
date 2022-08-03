"""This file contains the method that are used for simulating 
   an actor playing Tic Tac Toe
"""

### -------------- Imports -------------- ###

import random
import numpy as np

import torch

from tic_env import TictactoeEnv, OptimalPlayer
from memory import ReplayMemory
from helpers import choose_action, encode, select_model_action, \
                    grid2tensor, optimize_model, push_to_memory
from helpers import E_MIN, E_MAX, ALPHA, GAMMA, SEED, device, \
                    BATCH_SIZE, TARGET_UPDATE

### -------------- Q-Learning -------------- ###

def test(Q, rand, e=0):
    ''' Function for computing M_opt and M_rand \n
        Args:
            Q: Dict[String, Float]. Learned Q function.
            rand: Boolean. True if calculating M_rand and False if calculating M_opt.
            e: Float. Exploration level, should be zero for testing. 
        Returns: 
            Float. (num_wins - num_losses) / num_games.
    '''
    rewards = []
    S = TictactoeEnv()
    for i in range(500): # Fixed number of test games
        np.random.seed(i) # Different seed for each iteration
        p = OptimalPlayer(int(rand), 'O') if i % 2 == 0 else OptimalPlayer(int(rand), 'X')
        while not S.end:
            if S.current_player != p.player:
                A = choose_action(S, e, Q)
            else:
                A = p.act(S.grid)

            grid, end, winner = S.step(A)
            if end:
                # 1 if policy wins, -1 if policy loses, else 0
                R = - S.reward(player=p.player)
                rewards.append(R)
                S.reset()
                break
    np.random.seed(42)
    return sum(rewards)  / len(rewards)


def train(N, e=0.1, e_opt=0.5, compute_every=250, decrease_epsilon=False, dec_factor=1, 
          compute_tests=False, itself=False):
    """ Function for training a policy Sarsa \n
        Args:
            N: Int. Number of games.
            e: Float. Epsilon value in [0, 1].
            e_opt: Float. Epsilon value for OptimalPlayer in [0, 1]
            compute_every: Int. Number of games before compute training and test average.
            decrease_epsilon: Boolean. True if epsilon should be decreased each game.
            dec_factor: Int. Decreasing factor for epsilon.
            compute_tests: Boolean. True if M_opt and M_rand should be computed.
            itself: Boolean. True if the algorithm should learn by playing itself.
        Returns: 
            Tuple. Training rewards, M_opt and M_rand, losses averaged over buckets of size compute_every, Q function
    """
    Q = {}
    S = TictactoeEnv() # Init env
    avg_rewards = [0]
    curr_rewards = []
    m_opt, m_rand = [0], [0]
    avg_losses = [0]
    for i in range(N): 
        if i % 5000 == 0:
            print(f'Starting game {i}')
        # Compute reward (and test)
        if i % compute_every == 0 and i > 0:
            avg_rewards.append(np.mean(curr_rewards))
            curr_rewards = []
            if compute_tests:
                m_opt.append(test(Q, False))
                m_rand.append(test(Q, True))
        # Decrease epsilon
        if decrease_epsilon:
            e = max(E_MIN, E_MAX*(1 - i / dec_factor))

        # Switch starting player
        opt = OptimalPlayer(e_opt, 'O') if i % 2 == 0 else OptimalPlayer(e_opt, 'X')

        # Keep track of explored states, 
        # along with the action and the following next state (for Deep Q-learning)
        states = []

        # Play game
        while not S.end:
            
            # Our turn
            if S.current_player != opt.player:
                A = choose_action(S, e, Q)
                state_action = encode(S, A)
                states.append(state_action)

            # Optimal algorithm's turn
            else:
                # if we play against ourselves, we choose the action in the moral sense
                if itself:
                    A = choose_action(S, e, Q)
                    state_action = encode(S, A)
                    states.append(state_action)
                else: 
                    A = opt.act(S.grid)
            # Store whether the action is valid 
            valid_action = S.check_valid(A)

            # Perform action if valid 
            if valid_action:
                grid, end, winner = S.step(A)
            if end:
                # Calculate and store our reward
                if valid_action:
                    # calculating the correct sign of reward when we play against ourselves
                    if itself and S.current_player != opt.player:
                        R = S.reward(player=opt.player)
                    else:
                        R = - S.reward(player=opt.player) 
                # if the move is illegal; reward is -1
                else:
                    R = -1
                curr_rewards.append(R) 

                #Storing the next state for use in playing against expert
                next_Q = None 
                # storing the second next state for playing against itself
                # this is due to us saving states from "both sides of the table"
                next2_Q = None

                # iterating over the states LIFO
                for S_A in reversed(states):
                    value = Q.get(S_A, 0)

                    # logic to use correct next Q vlaue when playing against itself
                    if itself:
                        next = next2_Q
                    else:
                        next = next_Q
                    
                    # updating Q values
                    if next is None:
                        Q[S_A] = value + ALPHA * (R - value)
                    else:
                        Q[S_A] = value + ALPHA * (R + GAMMA * next - value)
                    
                    # making ready for next step of the iteration
                    next2_Q = next_Q
                    next_Q = Q[S_A]
                    
                    # if itself, reward must change sign between each iteration
                    if itself:
                        R = R*(-1)
                        

                # Reset and start new game
                S.reset()
                break
    
    # Quick fix
    np.random.seed(SEED)
    random.seed(SEED)
    return avg_rewards, m_opt, m_rand, avg_losses, Q



### -------------- Deep Q-Learning -------------- ###


# We test on policy net, as using the target net would be the same as 
# using the policy net from X iterations backwards
def test_deep(policy_net, rand, e=0):
    ''' Function for computing M_opt and M_rand \n
        Params:
            policy_net: DQN. Learned Q function. 
            rand: Boolean. True if calculating M_rand and False if calculating M_opt.
            e: Float. Exploration level, should be zero for testing. 
        Returns: 
            Float. (num_wins - num_losses) / num_games
    '''
    rewards = []
    S = TictactoeEnv()
    for i in range(500): # Fixed number of test games
        np.random.seed(i) # Different seed for each iteration
        # int(False) = 0 => optimal player, else random player
        p = OptimalPlayer(int(rand), 'O') if i % 2 == 0 else OptimalPlayer(int(rand), 'X')
        while not S.end:
            if S.current_player != p.player:
                A = select_model_action(device, policy_net, grid2tensor(S.grid), e, S)
                A = A.item()
                if not S.check_valid(A):
                    rewards.append(-1)
                    S.reset()
                    break
            else:
                A = p.act(S.grid)

            _, end, _ = S.step(A)
            if end:
                # 1 if policy wins, -1 if policy loses, else 0
                R = - S.reward(player=p.player)
                rewards.append(R)
                S.reset()
                break

    return sum(rewards)  / len(rewards)


def train_deep(N, optimizer, policy_net, target_net, e=0.1, e_opt = 0.5,
            decrease_epsilon=False, dec_factor = 1, memory=ReplayMemory(10000), 
            compute_tests=False, compute_every=250, against_itself=False, batch_size=BATCH_SIZE
            ):
    """ Function for training a deep Q-learning network \n
    Args:
        N: Int. Number of games.
        optimizer: torch.optim.Optimizer. Optimizer to be used.
        policy_net: DQN. Policy network.
        target_net: DQN. Target network.
        e: Float. Epsilon value in [0, 1].
        e_opt: Float. Epsilon value for OptimalPlayer in [0, 1]
        decrease_epsilon: Boolean. True if epsilon should be decreased each game.
        dec_factor: Int. Decreasing factor for epsilon.
        memory: ReplayMemory. Memory to sample from when optimizing model.
        compute_tests: Boolean. True if M_opt and M_rand should be computed.
        compute_every: Int. Number of games before compute training and test average.
        against_itself: Boolean. True if the algorithm should learn by playing itself.
        batch_size: Int. Size of batch to sample from memory. 
    Returns: 
        Tuple[List[Float]]. Training rewards, losses, M_opt and M_rand,
        averaged over buckets of size compute_every 
    """
    # initializing variables to keep track of rewards etc.
    rewards = []
    losses = []

    curr_rew = []
    curr_loss = []

    m_opts, m_rands = [], []


    Q = {}
    #playing N times
    for i in range(N):
        # creating game
        S = TictactoeEnv()
        state = grid2tensor(S.grid)
        opt = OptimalPlayer(e_opt, 'O') if i % 2 == 0 else OptimalPlayer(e_opt, 'X')

        if decrease_epsilon:
            e = max(E_MIN, E_MAX*(1 - i / dec_factor))
        
        if compute_tests and i % compute_every == 0:
            m_opt, m_rand = test_deep(policy_net, False), test_deep(policy_net, True)
            m_opts.append(m_opt)
            m_rands.append(m_rand)
        
        # if optimal player is first, let him do his first action before the loop
        # this ensure every round in the loop is: player act, optimal act, states and rewards pushed to memory
        if S.current_player == opt.player and not against_itself:
            action_opt = opt.act(S.grid)
            state, _, _ = S.step(action_opt)
            state = grid2tensor(state)

        # playing until game ends
        while not S.end:

            # runs if the agent learns by playing against itself
            if against_itself:
                reward = play_against_itself(S, e, policy_net, memory)
                S.reset()
                break
            
            # making an action
            if S.current_player != opt.player:
                A = select_model_action(device, policy_net, grid2tensor(S.grid), e, S)
                valid = S.check_valid(int(A))
                if not valid:
                    end = True
                else:
                    next_state, end, _ = S.step(A.item())

                # optimal player makes his action
                if not S.end and not end:
                    action_opt = opt.act(S.grid)
                    next_state, _, _ = S.step(action_opt)
            else:
                action_opt = opt.act(S.grid)
                next_state, _, _ = S.step(action_opt)

            # calculating rewards if the game is over
            if S.end or end:
                next_state=None 
                if S.end:
                    reward = -S.reward(player=opt.player)
                else:
                    reward= -1
            else:
                reward = 0
            
            # pushing state, action, reward to memory
            next_state = grid2tensor(next_state)
            if type(A) is tuple:
                A = A[1]+A[0]*(3)
                A = torch.tensor([[A]], device=device)
            memory.push((state), A, (next_state), torch.tensor([reward], device=device))
            state = next_state

            # new game if the current game is finished
            if end==True:
                break
        
        # optimizing the model
        loss = optimize_model(policy_net, target_net, memory, optimizer, batch_size=batch_size)
        
        curr_rew.append(reward)
        curr_loss.append(loss)
        
        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if (i%250==0) and i!=0:
            mean_rew = np.asarray([element for element in curr_rew if element!=None]).mean()
            mean_loss = np.nanmean(np.asarray([element for element in curr_loss if element!=None]))
            curr_rew = []
            curr_loss = []
            rewards.append(mean_rew)
            losses.append(mean_loss)
        if i%5000 == 0:
            print(i)
    return rewards, losses, m_opts, m_rands

def play_against_itself(S, e, policy_net, memory):
    """ Function for model to play one game of tic tac toe against itself \n
        Args:
            S: TictactoeEnv. State of game.
            e: Float. Epsilon value in [0, 1].
            policy_net: DQN. Policy network.
            memory: ReplayMemory. Memory to sample from when optimizing model. 
        Returns: 
            List[Float]. Training rewards.
    """
    
    # Init state
    end = False
    flip = True
    # starting by making a move to ensure that we have a previous state and action
    prev_state = grid2tensor(S.grid)
    A = select_model_action(device, policy_net, prev_state, e, S)
    state_2, *_ = S.step(A.item())
    state = grid2tensor(state_2)
    prev_A = A
    A = select_model_action(device, policy_net, state, e, S)
    if not S.check_valid(int(A)):
        push_to_memory(memory, grid2tensor(state_2, player='O'), torch.tensor([[A]]), None, -1, flip=flip)
        return
    
    while True:

        if S.check_valid(int(A)):
            next_state, end, winner = S.step(A.item())
            next_state = grid2tensor(next_state)
            if end:
                # If flip is true, we play from "the other side of the table"
                reward = S.reward('O' if flip else 'X')
                # Check format
                if type(A) is tuple:
                    A = A[1]+A[0]*(3)
                    A = torch.tensor([[A]], device=device)
                if type(prev_A) is tuple:
                    prev_A = prev_A[1]+prev_A[0]*(3)
                    prev_A = torch.tensor([[prev_A]], device=device)
                # Push to memory
                push_to_memory(memory, prev_state, prev_A, None, -reward, not flip)
                push_to_memory(memory, state, A, None, reward, flip)
                break
        else:
            reward = 0
            if type(A) is tuple:
                    A = A[1]+A[0]*(3)
                    A = torch.tensor([[A]], device=device)
            if type(prev_A) is tuple:
                prev_A = prev_A[1]+prev_A[0]*(3)
                prev_A = torch.tensor([[prev_A]], device=device)
            push_to_memory(memory, prev_state, prev_A, None, 0, not flip)
            push_to_memory(memory, state, A, None, -1, flip)
            break # Only get here if one player makes illegal move

        if type(prev_A) is tuple:
                prev_A = prev_A[1]+prev_A[0]*(3)
                prev_A = torch.tensor([[prev_A]], device=device)

        # Push last round to memory and increment state
        push_to_memory(memory, prev_state, prev_A, next_state, 0, not flip)
        flip = not flip # Switch player
        prev_A = A
        A = select_model_action(device, policy_net, next_state, e, S)
        prev_state = state
        state = next_state
        
    return reward
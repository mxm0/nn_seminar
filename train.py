import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from scene_loader import THORDiscreteEnvironment as Environment
from network import ActorCriticFFNetwork
from constants import ACTION_SIZE

import my_optim

import random
from constants import MAX_TIME_STEP, CHECKPOINT_DIR

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def choose_action(pi_values):
    values = []
    sum = 0.0
    for rate in pi_values:
      sum = sum + rate
      value = sum
      values.append(value)

    r = random.random() * sum
    for i in range(len(values)):
      if values[i] >= r:
        return i

def train(rank, scene_scope, task_scope, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)

    #env = create_atari_env(args.env_name)
    #env.seed(args.seed + rank)
    
    env = Environment({
        'scene_name': scene_scope,
        'terminal_state_id': int(task_scope)
      })

    model = ActorCriticFFNetwork(ACTION_SIZE)

    if optimizer is None:
        # TODO: Discount learning rate based on episode length
            optimizer = my_optim.SharedRMSprop(shared_model.parameters(), lr=args.lr, alpha = args.alpha, eps = args.eps)
            optimizer.share_memory()
    
    model.train()

    env.reset()
    state = torch.from_numpy(env.s_t)
    done = True

    episode_length = 0
    for i in range(int(args.max_episode_length)):
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        '''
        if done:
            cx = Variable(torch.zeros(1, 256))
            hx = Variable(torch.zeros(1, 256))
        else:
            cx = Variable(cx.data)
            hx = Variable(hx.data)
        '''

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            print('Thread: ', rank, ', step: ', step, 'epochs:', i)
            episode_length += 1
            logit, value = model(env.s_t, env.target)
            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, Variable(action))
            
            env.step(action)
            #state, reward, done, _ = env.step(action.numpy())
            env.update()
            state = env.s_t
            reward = env.reward
            done = env.terminal

            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                if counter.value % 1000 == 0:
                    print('Now saving data. Please wait.')
                    torch.save(shared_model.state_dict(), CHECKPOINT_DIR + '/' + 'checkpoint.pth.tar')
                counter.value += 1
            
            if done:
                episode_length = 0
                if env.terminal: 
                    print('Task completed')
                counter.value += 1

            if done:
                episode_length = 0
                env.reset()
                state = env.s_t

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            _, value = model(env.s_t, env.target)
            R = value.data

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

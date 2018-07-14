import time
from collections import deque

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from network import ActorCriticFFNetwork
from constants import ACTION_SIZE
from scene_loader import THORDiscreteEnvironment as Environment
import cv2

def test(rank, scene_scope, task_scope, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)
    
    env = Environment({
        'scene_name': scene_scope,
        'terminal_state_id': int(task_scope)
        })
    
    model = ActorCriticFFNetwork(ACTION_SIZE)

    model.eval()

    height, width, layers = env.observation.shape
    video = cv2.VideoWriter('video/' + task_scope + '.mp4',-1,1,(width,height))

    env.reset()
    state = torch.from_numpy(env.s_t)
    reward_sum = 0
    done = True

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0

    img = cv2.cvtColor(env.observation, cv2.COLOR_BGR2RGB)
    video.write(img)
    for i in range(100):
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())

        logit, value = model(env.s_t, env.target)
        prob = F.softmax(logit, dim=1)
        action = prob.max(1, keepdim=True)[1].data.numpy()
        env.step(action[0, 0])
        env.update()        
        img = cv2.cvtColor(env.observation, cv2.COLOR_BGR2RGB)
        video.write(img)
        
        reward = env.reward
        state = env.s_t
        done = env.terminal
        print(env.terminal_state_id, env.current_state_id)
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            print("Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            actions.clear()
            env.reset()
            state = env.s_t
            break

        state = torch.from_numpy(state)
    cv2.destroyAllWindows()
    video.release()

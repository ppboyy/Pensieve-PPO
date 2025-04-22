import multiprocessing as mp
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from env import ABREnv
import ppo2 as network
import torch
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

S_DIM = [6, 8]
A_DIM = 6
ACTOR_LR_RATE = 1e-4
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 1000  # take as a train batch
TRAIN_EPOCH = 100
RANDOM_SEED = 42
SUMMARY_DIR = './ppo'
MODEL_DIR = './models'
TRAIN_TRACES = './posttrain_trace/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = SUMMARY_DIR + '/log'
EPOCH_FILE = os.path.join(SUMMARY_DIR, "epoch.txt")

def load_last_epoch() -> int:
    if not os.path.exists(EPOCH_FILE):
        return -1
    with open(EPOCH_FILE) as f:
        try:
            return int(f.read().strip())
        except ValueError:
            return -1

def save_last_epoch(epoch: int):
    with open(EPOCH_FILE, "w") as f:
        f.write(str(epoch))
# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

# If the user passed a model path on the command line, use it
if len(sys.argv) > 1 and sys.argv[1].endswith(".pth"):
    NN_MODEL = sys.argv[1]
else:
    NN_MODEL = "./pretrain_model/nn_model_ep_155400.pth"    

def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    #os.system('mkdir ' + TEST_LOG_FOLDER)

    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system('python validate.py ' + nn_model)

    # append test performance to the log
    rewards, entropies = [], []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))
        entropies.append(np.mean(entropy[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies)
        
def central_agent():
    
    with open(LOG_FILE + '_test.txt', 'a') as test_log_file:
        env = ABREnv()
        actor = network.Network(state_dim=S_DIM, 
                                action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)
        last_epoch_done = load_last_epoch()                  # e.g. -1 on first run
        start_epoch     = last_epoch_done + 1                # = 0 on first run
        writer = SummaryWriter(SUMMARY_DIR,
                             purge_step=start_epoch)       # don’t re‑plot ol d steps

        
        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            actor.load_model(nn_model)
            print('Model restored:', nn_model)
        # while True:  # assemble experiences from agents, compute the gradients
        #for epoch in range(TRAIN_EPOCH):
            # synchronize the network parameters of work agent
        actor_net_params = actor.get_network_params() ## get the actor network params
        actor.set_network_params(actor_net_params) # set the actor network params

        for epoch in range(start_epoch, start_epoch + TRAIN_EPOCH):
            obs = env.reset()
            s_batch, a_batch, p_batch, r_batch = [], [], [], []
            for step in range(TRAIN_SEQ_LEN):
                s_batch.append(obs)

                action_prob = actor.predict(
                    np.reshape(obs, (1, S_DIM[0], S_DIM[1])))

                # gumbel noise
                noise = np.random.gumbel(size=len(action_prob))
                bit_rate = np.argmax(np.log(action_prob) + noise)

                obs, rew, done, info = env.step(bit_rate)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)
                r_batch.append(rew)
                p_batch.append(action_prob)
                if done:
                    break

            v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
            s_batch = np.stack(s_batch, axis=0)
            a_batch = np.vstack(a_batch)
            p_batch = np.vstack(p_batch)
            v_batch = np.vstack(r_batch)
            actor.train(s_batch, a_batch, p_batch, v_batch, epoch)

            actor_net_params = actor.get_network_params()
            actor.set_network_params(actor_net_params)

        save_last_epoch(epoch)   # remember the final epoch
        print(f"[INFO] Finished training up to epoch {epoch}")

        actor.save_model(MODEL_DIR + '/nn_model_ep_' + str(epoch) + '.pth')
            
        avg_reward, avg_entropy = testing(epoch,
                MODEL_DIR + '/nn_model_ep_' + str(epoch) + '.pth', 
                test_log_file)

        writer.add_scalar('Entropy Weight', actor._entropy_weight, epoch)
        writer.add_scalar('Reward', avg_reward, epoch)
        writer.add_scalar('Entropy', avg_entropy, epoch)
        writer.flush()

"""
def agent(agent_id, net_params_queue, exp_queue):
    env = ABREnv(agent_id)
    actor = network.Network(state_dim=S_DIM, action_dim=A_DIM,
                            learning_rate=ACTOR_LR_RATE)

    # initial synchronization of the network parameters from the coordinator
    actor_net_params = net_params_queue.get() #take the actor network params from the coordinator
    actor.set_network_params(actor_net_params) # set the actor network params

    for epoch in range(TRAIN_EPOCH):
        obs = env.reset()
        s_batch, a_batch, p_batch, r_batch = [], [], [], []
        for step in range(TRAIN_SEQ_LEN):
            s_batch.append(obs)

            action_prob = actor.predict(
                np.reshape(obs, (1, S_DIM[0], S_DIM[1])))

            # gumbel noise
            noise = np.random.gumbel(size=len(action_prob))
            bit_rate = np.argmax(np.log(action_prob) + noise)

            obs, rew, done, info = env.step(bit_rate)

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            a_batch.append(action_vec)
            r_batch.append(rew)
            p_batch.append(action_prob)
            if done:
                break
        v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
        exp_queue.put([s_batch, a_batch, p_batch, v_batch])

        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
"""
"""
def main():

    np.random.seed(RANDOM_SEED)
    #torch.set_num_threads(1)
    # inter-process communication queues
    #net_params_queues = []
    #exp_queues = []
    #for i in range(NUM_AGENTS):
    #    net_params_queues.append(mp.Queue(1))
    #    exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()
"""

if __name__ == '__main__':
    central_agent()

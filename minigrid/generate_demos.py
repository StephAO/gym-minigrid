
import os
import random

import argparse
import gymnasium as gym
from minigrid.oracle_agent import OracleAgent
import torch
import numpy as np



def main(args):

    if args.seed:
        raise NotImplementedError
        # does not seed env
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)


    env_name_list = [
                    "ContrastiveTrajectoryDataset-v0", 
                    "MiniGrid-GoToDoor-8x8-v0",
                    "MiniGrid-GoToObject-8x8-N2-v0"
                    ]
    

    save_dir = args.save_dir

    offset = 0 # change this if do not want to overwrite
    traj_per_env = args.traj_per_env
    min_traj_len = args.min_traj_len
    max_traj_len = args.max_traj_len
    print_freq = args.print_freq
    render = args.render
    save_suffix = args.save_suffix


    if not os.path.isdir(save_dir):
        os.makedirs (save_dir)

    print("=" * 60)
    print("SAVING DEMOS")
    print("=" * 60)
    print("save_dir : ", save_dir)
    print("save_suffix: ", save_suffix)
    print("offset: ", offset)
    print("traj_per_env: ", traj_per_env)
    print("min_traj_len: ", min_traj_len)
    print("max_traj_len: ", max_traj_len)
    print("print_freq: ", print_freq)
    print("render: ", render)
    print("env_name_list: ")
    print(env_name_list)
    print("=" * 60)

    global_traj_len_max = 0

    for env_name in env_name_list:
        
        traj_len_sum = 0
        num_traj = 0
        traj_len_max = 0
        traj_len_min = 1000000

        print("saving demos for: ", env_name)

        env_save_dir = os.path.join(save_dir, env_name)
        if not os.path.isdir(env_save_dir):
            os.makedirs (env_save_dir)
        

        save_data = []

        i = offset
        while i < offset + traj_per_env:
            
            # randomize env parameters
            env_size = random.randint(8, 16)
            env_numObjs = random.randint(env_size - 4, env_size)

            if "Door" in env_name:
                env = gym.make(env_name, size=env_size, render_mode='rgb_array')
            else:
                env = gym.make(env_name, size=env_size, numObjs=env_numObjs, render_mode='rgb_array')

            if "Contrastive" in env_name:
                split = "train"
                env.set_split(split)
            
            oracle = OracleAgent(env, visualize=render)
            demos = oracle.generate_demos(1)

            # path not found / failed trajectory
            if len(demos) < 1:
                continue

            mission, obss, actions, rewards, target_cell, label = demos[0]

            # only save trajectory if traj length in desired range
            traj_len = len(actions)
            if traj_len < min_traj_len or traj_len > max_traj_len:
                continue

            # create trajectory dictionary
            traj = dict()
            temp_images = []
            temp_directions = []
            temp_done = []
            for obs in obss:
                temp_images.append(obs["image"])
                temp_directions.append(obs["direction"])
                temp_done.append(0)
            temp_done[-1] = 1 # done at the end

            traj["images"] = torch.tensor(np.array(temp_images))
            traj["directions"] = torch.tensor(np.array(temp_directions))
            traj["actions"] = torch.tensor(actions)
            traj["rewards"] = torch.tensor(rewards)
            traj["done"] = torch.tensor(temp_done)
            traj["target_cell"] = torch.tensor(target_cell)
            traj["mission"] = mission

            save_data.append(traj)

            # print("=" * 60)
            # print("traj[images]", traj["images"].size())
            # print("traj[directions]", traj["directions"].size(), traj["directions"])
            # print("traj[actions]", traj["actions"].size(), traj["actions"])
            # print("traj[rewards]", traj["rewards"].size(), traj["rewards"])
            # print("traj[done]", traj["done"].size(), traj["done"])
            # print("traj[target_cell]", traj["target_cell"].size(), traj["target_cell"])
            # print("traj[mission]", traj["mission"])


            # print("=" * 60)
            # print("obss[0]: ", len(obss[0]), obss[0].keys())
            # print("obss[0]['image']: ", obss[0]['image'].shape)
            # print("obss[0]['mission']: ", obss[0]['mission'])
            # print("obss[-1]['mission']: ", obss[-1]['mission'])
            # print("obss[0]['direction']: ", obss[0]['direction'].item())
            # print("actions: ", len(actions), actions)
            # print("rewards: ", len(rewards), rewards)
            # print("mission: ", mission)

            if i % print_freq == 0:
                print("completed trajectory num: ", i)

            # book keeping
            i += 1
            traj_len_sum += traj_len
            traj_len_max = max(traj_len_max, traj_len)
            traj_len_min = min(traj_len_min, traj_len)
            num_traj += 1


        # save all the trajectory
        save_data_name = env_name + '_demos_' + str(num_traj) + save_suffix + ".pt"
        save_data_path = os.path.join(env_save_dir, save_data_name)
        torch.save(save_data, save_data_path)
        

        # avg trajectory len
        traj_len_avg = round(traj_len_sum / num_traj, 2)
        global_traj_len_max = max(global_traj_len_max, traj_len_max)

        print("completed saving demos at: ", save_data_path)
        print("traj_len_min: ", traj_len_min)
        print("traj_len_max: ", traj_len_max)
        print("traj_len_avg: ", traj_len_avg)
        print("=" * 60)



    print("completed saving all demos at: ", save_dir)
    print("global_traj_len_max: ", global_traj_len_max)
    print("env_name_list: ")
    print(env_name_list)
    print("=" * 60)




"""
Envs that work with the oracle:

env_name_list = [
                "ContrastiveTrajectoryDataset-v0", 
                "MiniGrid-GoToDoor-8x8-v0",
                "MiniGrid-GoToObject-8x8-N2-v0"
                ]

python3 -m minigrid.generate_demos --traj_per_env 10

python3 -m minigrid.generate_demos --traj_per_env 100000 --print_freq 20000


global_traj_len_max = 30
global avg = 9


"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./minigrid_demos",
        help="save demos in this directory",
    )
    parser.add_argument(
        "--save_suffix",
        type=str,
        default="",
        help="suffix to be added to saved file",
    )
    parser.add_argument(
        "--min_traj_len",
        type=int,
        help="minimum len of trajectory in demo",
        default=5,
    )
    parser.add_argument(
        "--max_traj_len",
        type=int,
        help="maximum len of trajectory in demo",
        default=300,
    )
    parser.add_argument(
        "--traj_per_env",
        type=int,
        help="num of trajectories per env to save",
        default=10,
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        help="print status after saving every n trajectories",
        default=10000,
    )
    parser.add_argument(
        "--render",
        default=False,
        help="render the trajectory",
        action="store_true",
    )

    args = parser.parse_args()
    
    ### save demos ###
    main(args)


    # ## test by loading some saved trajectories ### 
    # env_name = "MiniGrid-GoToObject-8x8-N2-v0"
    # load_dir = "./minigrid_demos/" + env_name
    # load_data_name = env_name + "_demos_" + "100" + ".pt"

    # load_data_path = os.path.join(load_dir, load_data_name)
    
    # print("=" * 60)
    # print("TESTING")
    # print("=" * 60)

    # print("load_data_path: ", load_data_path)

    # traj_data = torch.load(load_data_path)
    
    # for i in range(3): 

    #     traj = traj_data[i]

    #     print("traj num: ", i)
    #     print("traj[images]", traj["images"].size())
    #     print("traj[directions]", traj["directions"].size(), traj["directions"])
    #     print("traj[actions]", traj["actions"].size(), traj["actions"])
    #     print("traj[rewards]", traj["rewards"].size(), traj["rewards"])
    #     print("traj[done]", traj["done"].size(), traj["done"])
    #     print("traj[target_cell]", traj["target_cell"])
    #     print("traj[mission]", traj["mission"])

    #     print("=" * 60)









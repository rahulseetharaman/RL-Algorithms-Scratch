import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict


class CartPole:
    def __init__(self, init_state, environment, M, use_cos=True):
        self.init_state = init_state
        self.environment = environment
        self.M = M
        self.use_cos = use_cos

    def run_episode(self, theta):
        x,v,w,w_ang = self.init_state
        
        g=self.environment['g']
        mc=self.environment['mc']
        mp=self.environment['mp']
        mt=mc+mp
        l=self.environment['l']
        tau=self.environment['tau']
        
        num_timesteps=10
        num_actions = int(num_timesteps/tau)

        total_reward = 0
        
        data = {"v": [], "w_ang": []}

        for i in range(num_actions):
            
            if self.use_cos:
                # normalize x,v,w_ang,w
                v_normalized = (v + 1.6)/3.2
                w_ang_normalized = (w + 2.4)/4.8
                x_normalized = (x+2.4)/4.8
                w_normalized = (w + np.pi/15)/(np.pi/7.5)

                # create the policy vector
                x_vec = np.cos(np.arange(1, self.M+1) * np.pi * x_normalized)
                v_vec = np.cos(np.arange(1, self.M+1) * np.pi * v_normalized)
                w_vec = np.cos(np.arange(1, self.M+1) * np.pi * w_normalized)
                w_ang_vec = np.cos(np.arange(1, self.M+1) * np.pi * w_ang_normalized)
                one = np.ones(1,)
            else:
                # normalize x,v,w_ang,w
                v_normalized = 2*(v + 1.6)/3.2 - 1
                w_ang_normalized = 2*(w + 2.4)/4.8 - 1
                x_normalized = 2*(x+2.4)/4.8 - 1
                w_normalized = 2*(w + np.pi/15)/(np.pi/7.5) - 1

                # create the policy vector
                x_vec = np.sin(np.arange(1, self.M+1) * np.pi * x_normalized)
                v_vec = np.sin(np.arange(1, self.M+1) * np.pi * v_normalized)
                w_vec = np.sin(np.arange(1, self.M+1) * np.pi * w_normalized)
                w_ang_vec = np.sin(np.arange(1, self.M+1) * np.pi * w_ang_normalized)
                one = np.ones(1,)
            
            # perform dot product
            policy_vec = np.concatenate([one, x_vec, v_vec, w_vec, w_ang_vec])
            val = np.sum(theta * policy_vec)
            
            # go left or right based on obtained dot product
            if val <= 0:
                F=-10
            else:
                F=10
            
            # adjust v,w,x,w_ang
            b = (F + mp * l * (w_ang**2) * np.sin(w))/mt
            c = (g * np.sin(w) - b * np.cos(w))/(l * (4/3 - mp/mt * np.cos(w)**2))
            d = b - mp/mt * l * c * np.cos(w)
            x = x + tau * v
            v = v + tau * d
            w = w + tau * w_ang
            w_ang = w_ang + tau * c

            data['v'].append(v)
            data['w_ang'].append(w_ang)

            # terminate if the following occur
            if x > 2.4 or x < -2.4:
                break
                
            if w > np.pi/15 or w < -np.pi/15:
                break
    
            # add reward and continue
            total_reward += 1

        return total_reward, np.min(data['v']), np.max(data['v']), np.min(data['w_ang']), np.max(data['w_ang'])

    def run_simulation(self, theta, n_episodes):
        rewards = []
        v_list = []
        w_ang_list = []
        for n in range(n_episodes):
            r, v1, v2, w1, w2 =self.run_episode(theta)
            rewards.append(r)
            v_list.extend([v1, v2])
            w_ang_list.extend([w1, w2])
        return np.mean(rewards),  np.min(v_list), np.max(v_list), np.min(w_ang_list), np.max(w_ang_list)

    def run_evolutionary_search(self, n_episodes=2, num_perturbations=100, n_iters=50, sigma=0.5, alpha=0.5, ):
        theta = np.random.rand(4*self.M+1)
        orig_theta = theta.copy()
        w_ang_list = []
        v_list = []
        J_for_iter = []
        for c in range(n_iters):
            perturbations = np.random.normal(0,1, (num_perturbations, 4*self.M+1))
            for n in range(num_perturbations):
                perturbation = perturbations[n]
                reward, v1, v2, w1, w2 = self.run_simulation(theta = orig_theta+sigma*perturbation, n_episodes=n_episodes)
                v_list.extend([v1, v2])
                w_ang_list.extend([w1, w2])
                theta = theta + alpha * (1/sigma) * (1/num_perturbations) * reward * perturbation
            orig_theta = theta
            J_Val, _, _, _, _ = self.run_simulation(theta=orig_theta, n_episodes=n_episodes)
            J_for_iter.append(J_Val)
        return J_for_iter


def run_experiment(sigma, num_perturbations, M, N, alpha, use_cos=True):
    c=CartPole(init_state=(0,0,0,0), use_cos=use_cos, M=M, environment={"g":9.8, "mc":1.0, "mp": 0.1, "l": 0.5, "tau": 0.02})
    all_rewards = []
    for r in range(5):
        reward, v1, v2, w1, w2 = c.run_evolutionary_search(alpha=alpha, sigma=sigma, num_perturbations=num_perturbations, n_episodes=N)
        all_rewards.append(reward)
    return np.mean(all_rewards)

def grid_search(use_cos=True):

    best_hyperparams = None
    max_reward = -100.0

    with tqdm(total=24) as pbar:
        for s in sigma_vals:
            for p in num_pertubations_vals:
                for M in M_vals:
                    for N in N_vals:
                        for a in alpha_vals:
                            reward = run_experiment(s, p, M, N, a, use_cos=use_cos)
                            pbar.update(1)
                            if reward > max_reward:
                                max_reward = reward
                                best_hyperparams = (s,p,M,N,a)
                                print(best_hyperparams, max_reward)
        print("End of grid search")
        print(best_hyperparams)
    
    return best_hyperparams


if __name__ == '__main__':
    sigma_vals = [0.8, 0.9, 1.0]
    num_pertubations_vals = [85, 95, 100]
    M_vals = [3,4,5,6]
    N_vals = [1]
    alpha_vals = [0.01, 0.02, 0.03, 0.04]

    
    '''
    Example invocation of grid search function:
    
        best_hyperparams = grid_search(use_cos=False)
    
    '''

    # Best hyperparameters obtained after several rounds of grid search
    best_hyperparams = (1.0, 100, 3, 1, 0.01, False)
    
    hyperparams_set = {
        best_hyperparams: "best.png",
        (0.75, 100, 6, 1, 0.01, True): "high_np.png",
        (1.0, 10, 6, 1, 0.01, True): "low_np.png",
        (0.1, 100, 6, 1, 0.01, True): "low_sigma.png",
        (1.0, 100, 6, 1, 0.01, True): "high_sigma.png",
        (1.0, 100, 6, 1, 0.0005, True): "very_low_alpha.png",
        (1.0, 100, 6, 1, 0.01, False): "use_sine.png",
        (1.0, 100, 3, 1, 0.01, False): "use_sine_M3.png",
        (1.0, 100, 9, 1, 0.01, False): "use_sine_M9.png",
    }

    J_tot = []

    for trial in tqdm(range(3)):
        
        c=CartPole(init_state=(0,0,0,0), use_cos=h[5], M=h[2], environment={"g":9.8, "mc":1.0, "mp": 0.1, "l": 0.5, "tau": 0.02})
        J_vals = c.run_evolutionary_search(alpha=h[4], sigma=h[0], num_perturbations=h[1], n_episodes=h[3])
        J_tot.append(J_vals)

    J_mean = np.mean(J_tot, axis=0)
    J_std = np.std(J_tot, axis=0)


    plt.xlabel("Iterations")
    plt.ylabel("Value function")
    plt.errorbar(range(50), J_mean, yerr=J_std)
    plt.savefig("best.png")
    plt.clf()
    
    # for h, fname in tqdm(hyperparams_set.items()):
    #     J_tot = [] 

    #     for trial in range(5):
    #         c=CartPole(init_state=(0,0,0,0), use_cos=h[5], M=h[2], environment={"g":9.8, "mc":1.0, "mp": 0.1, "l": 0.5, "tau": 0.02})
    #         J_vals = c.run_evolutionary_search(alpha=h[4], sigma=h[0], num_perturbations=h[1], n_episodes=h[3])
    #         J_tot.append(J_vals)
        
    #     J_mean = np.mean(J_tot, axis=0)
    #     J_std = np.std(J_tot, axis=0)
    #     plt.xlabel("Iterations")
    #     plt.ylabel("Value function")
    #     plt.errorbar(range(50), J_mean, yerr=J_std)
    #     plt.savefig(fname)
    #     plt.clf()

    



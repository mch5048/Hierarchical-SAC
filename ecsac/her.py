import numpy as np

class Her_sampler(object):
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future': # elsewhere is 'final'
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func # method that compute batch rewards 

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        """
            Episode_batch consists of ep_obs, ep_obs_1, ep_ag, ep_ag_1, ep_g, ep_act, ep_rew, ep_dn        

            Sequence:
                1. uniformly shuffle time indicies and accompanying data tuple in episodic batch
                2. sample her_indices proportional to the 'future_p'
                3. sample offsets for each future_goal samples corresponding to her_indicies
                4.  
               
        """
        T = episode_batch[0].shape[0]
        rollout_batch_size = episode_batch['actions'].shape[0] #300
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used

        t_samples = np.random.randint(T, size=batch_size) # random indices        
        transitions = [elem[t_samples] for elem in episode_batch] # resort the array to t_samples incides
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p) # portion of future_p : future, else : final
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + future_offset)[her_indexes] # adding 1 her could be problematic
        # replace des_goal with achieved goal / her_application
        future_ag = episode_batch[3][future_t]
        transitions[4][her_indexes] = future_ag 
        # to get the params to re-compute reward
        # compute batch reward from achieved goals and desired goals 
        transitions[6] = np.expand_dims(self.reward_func(transitions[3], transitions[4], None), 1)
        # update mean/std of normalizers       

        return transitions

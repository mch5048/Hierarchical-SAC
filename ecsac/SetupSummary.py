#!/usr/bin/env python
import numpy as np
import random
import tensorflow as tf
from running_mean_std import RunningMeanStd

OBS_SHAPE = (100,100,3)
POS = (7,)
VEL = (7,)
EFF = (7,)
OBJ = (3,)
CRIT_GOAL = (14,)


class SummaryManager(object):

    def __init__(self, sess, obs_shape_list, summary_writer):
        self.sess = sess
        obs_shape_list = obs_shape_list
        self.summary_writer = summary_writer

        self.BS = 1

        self.s_t0_rms = RunningMeanStd(shape=obs_shape_list[0])
        self.s_t1_rms = RunningMeanStd(shape=obs_shape_list[1])
        self.s_t2_rms = RunningMeanStd(shape=obs_shape_list[2])
        self.s_t3_rms = RunningMeanStd(shape=obs_shape_list[3])
        self.s_t4_rms = RunningMeanStd(shape=obs_shape_list[4])
        self.goal_state0_rms = RunningMeanStd(shape=obs_shape_list[5])
        self.goal_state1_rms = RunningMeanStd(shape=obs_shape_list[6])
        self.goal_obs_rms = RunningMeanStd(shape=obs_shape_list[7])
        # achieved goals have the same shape with that of desired goals
        self.achvd_obs_rms = RunningMeanStd(shape=obs_shape_list[10])
        self.achvd_state0_rms = RunningMeanStd(shape=obs_shape_list[8])
        self.achvd_state1_rms = RunningMeanStd(shape=obs_shape_list[9])


    def setup_state_summary(self):
        ols = []

        with tf.variable_scope('image_observation'):
            self.step_obs_ph = tf.placeholder(shape=(self.BS,)+OBS_SHAPE, dtype=tf.float32, name='step_obs')
            self.achvd_obs_ph = tf.placeholder(shape=(self.BS,)+OBS_SHAPE, dtype=tf.float32, name='achvd_obs')
            self.goal_obs_ph = tf.placeholder(shape=(self.BS,)+OBS_SHAPE, dtype=tf.float32, name='goal_obs')
        with tf.variable_scope('full_states'):
            self.step_stt_pos_ph = tf.placeholder(shape=(self.BS,)+POS, dtype=tf.float32, name='state_pos')
            self.step_stt_vel_ph = tf.placeholder(shape=(self.BS,)+VEL, dtype=tf.float32, name='state_vel')
            self.step_stt_eff_ph = tf.placeholder(shape=(self.BS,)+EFF, dtype=tf.float32, name='state_eff')
            self.step_stt_obj_ph = tf.placeholder(shape=(self.BS,)+OBJ, dtype=tf.float32, name='state_obj')
            self.step_stt_crit_ph = tf.placeholder(shape=(self.BS,)+CRIT_GOAL, dtype=tf.float32, name='state_critic')
        with tf.variable_scope('state_summaries'):
            ols.append(tf.summary.image('step_obs', self.step_obs_ph))
            ols.append(tf.summary.image('achvd_obs', self.achvd_obs_ph)) 
            ols.append(tf.summary.image('goal_obs', self.goal_obs_ph)) 
            ols.append(tf.summary.histogram('state_pos', self.step_stt_pos_ph)) 
            ols.append(tf.summary.histogram('state_vel', self.step_stt_vel_ph)) 
            ols.append(tf.summary.histogram('state_eff', self.step_stt_eff_ph)) 
            ols.append(tf.summary.histogram('state_obj', self.step_stt_obj_ph)) 
            ols.append(tf.summary.histogram('state_critic', self.step_stt_crit_ph)) 

        self._state_summary_op = tf.summary.merge(ols)

                                #   0       1           2       3           4       5       6       7
    def update_state_summary(self, step, step_obs, achvd_obs, goal_obs, pos_stt, vel_stt, eff_stt, obs_stt): # inputs are numpy ndarray type

        # reshape observations
        step_obs = np.reshape(step_obs,(-1,100,100,3))
        achvd_obs = np.reshape(achvd_obs,(-1,100,100,3))
        goal_obs = np.reshape(goal_obs,(-1,100,100,3))
        
        pos_stt = np.reshape(pos_stt,(-1,7))
        vel_stt = np.reshape(vel_stt,(-1,7))
        eff_stt = np.reshape(eff_stt,(-1,7))
        obs_stt = np.reshape(obs_stt,(-1,3))


        _crit_state_arr = np.hstack([pos_stt,vel_stt])

        state_summary = self.sess.run(self._state_summary_op, feed_dict={
            self.step_obs_ph : step_obs,
            self.achvd_obs_ph : achvd_obs,
            self.goal_obs_ph : goal_obs,
            self.step_stt_pos_ph : pos_stt,
            self.step_stt_vel_ph : vel_stt,
            self.step_stt_eff_ph : eff_stt,
            self.step_stt_obj_ph : obs_stt,
            self.step_stt_crit_ph : _crit_state_arr,
            })

        self.summary_writer.add_summary(state_summary, step)


    def setup_stat_summary(self): # inputs are numpy ndarray type
        ops = []
        names = []
        stat_summary_list = []


        # porting from np to tf is required
        # ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
        with tf.variable_scope('obs_stat_summary'):
            self.st0_mean = tf.placeholder(shape=(), dtype=tf.float32, name='s_t0_rms_mean')
            self.st0_std= tf.placeholder(shape=(), dtype=tf.float32, name='s_t0_rms_std')

            self.goal_obs_mean = tf.placeholder(shape=(), dtype=tf.float32, name='goal_obs_rms_mean')
            self.goal_obs_std= tf.placeholder(shape=(), dtype=tf.float32, name='goal_obs_rms_std')

            ops +=[self.st0_mean, self.st0_std, 
                    self.goal_obs_mean, self.goal_obs_std]

            names +=['s_t0_rms_mean', 's_t0_rms_std',
             'goal_obs_rms_mean', 'goal_obs_rms_std']

            stats_ops = ops
            stats_names = names

            for op, name in zip(stats_ops, stats_names):
                stat_summary_list.append(tf.summary.scalar(name,op))

        ops2 = []
        names2 = []

        with tf.variable_scope('state_stat_summary'):

            self.st1_mean = tf.placeholder(shape=(), dtype=tf.float32, name='s_t1_rms_mean')
            self.st1_std= tf.placeholder(shape=(), dtype=tf.float32, name='s_t1_rms_std')

            self.st2_mean = tf.placeholder(shape=(), dtype=tf.float32, name='s_t2_rms_mean')
            self.st2_std= tf.placeholder(shape=(), dtype=tf.float32, name='s_t2_rms_std')

            self.st3_mean = tf.placeholder(shape=(), dtype=tf.float32, name='s_t3_rms_mean')
            self.st3_std= tf.placeholder(shape=(), dtype=tf.float32, name='s_t3_rms_std')

            self.st4_mean = tf.placeholder(shape=(), dtype=tf.float32, name='s_t4_rms_mean')
            self.st4_std= tf.placeholder(shape=(), dtype=tf.float32, name='s_t4_rms_std')
                
            self.goal0_mean = tf.placeholder(shape=(), dtype=tf.float32, name='goal_s0_rms_mean')
            self.goal0_std= tf.placeholder(shape=(), dtype=tf.float32, name='goal_s1_rms_std')

            self.goal1_mean = tf.placeholder(shape=(), dtype=tf.float32, name='goal_s1_rms_mean')
            self.goal1_std= tf.placeholder(shape=(), dtype=tf.float32, name='goal_s1_rms_std')

            ops2 +=[self.st1_mean, self.st1_std, 
                    self.st2_mean, self.st2_std, 
                    self.st3_mean, self.st3_std,
                    self.st4_mean, self.st4_std, 
                    self.goal0_mean, self.goal0_std, 
                    self.goal1_mean, self.goal1_std]

            names2 +=['s_t1_rms_mean', 's_t1_rms_std', 
                    's_t2_rms_mean', 's_t2_rms_std', 
                    's_t3_rms_mean', 's_t3_rms_std',
                    's_t4_rms_mean', 's_t4_rms_std',
                    'goal_s0_rms_mean', 'goal_s0_rms_std',
                    'goal_s1_rms_mean', 'goal_s1_rms_std']


            for op2, name2 in zip(ops2, names2):
                stat_summary_list.append(tf.summary.scalar(name2,op2))


        self.stat_summary_op = tf.summary.merge(stat_summary_list)

    def update_stat_summary(self, step): # inputs are numpy ndarray type

        stat_summary = self.sess.run(self.stat_summary_op,
                                        feed_dict={
        self.st0_mean: np.mean(self.s_t0_rms.mean),
        self.st0_std: np.mean(self.s_t0_rms.std), 
        self.st1_mean: np.mean(self.s_t1_rms.mean), 
        self.st1_std: np.mean(self.s_t1_rms.std), 
        self.st2_mean: np.mean(self.s_t2_rms.mean), 
        self.st2_std: np.mean(self.s_t2_rms.std),
        self.st3_mean: np.mean(self.s_t3_rms.mean), 
        self.st3_std: np.mean(self.s_t3_rms.std),
        self.st4_mean: np.mean(self.s_t4_rms.mean), 
        self.st4_std: np.mean(self.s_t4_rms.std), 
        self.goal0_mean: np.mean(self.goal_state0_rms.mean), 
        self.goal0_std: np.mean(self.goal_state0_rms.std), 
        self.goal1_mean: np.mean(self.goal_state1_rms.mean), 
        self.goal1_std: np.mean(self.goal_state1_rms.std),
        self.goal_obs_mean: np.mean(self.goal_obs_rms.mean),
        self.goal_obs_std: np.mean(self.goal_obs_rms.std),
        })

        self.summary_writer.add_summary(stat_summary, step)



        # It's in main script
        # def setup_summary(self):

        #     with tf.variable_scope('training_summary'):
        #         episode_total_reward = tf.Variable(0.,name='total_reward')
        #         episode_q = tf.Variable(0.,name='q_val')
        #         episode_step = tf.Variable(0.,name='step')
        #         episode_l2_loss = tf.Variable(0.,name='l2_crit_loss')
        #         episode_1step_loss = tf.Variable(0.,name='1step_loss')
        #         episode_nstep_loss = tf.Variable(0.,name='nstep_loss')

        #         gs = []
        #         gs.append(tf.summary.scalar('Total_Reward/Episode', episode_total_reward))
        #         gs.append(tf.summary.scalar('Avg_Q/Episode', episode_q))
        #         gs.append(tf.summary.scalar('Took_Steps/Episode', episode_step))
        #         gs.append(tf.summary.scalar('L2_Critic_Loss/Episode', episode_l2_loss))
        #         gs.append(tf.summary.scalar('1step_Loss/Episode', episode_1step_loss))
        #         gs.append(tf.summary.scalar('Nstep_Loss/Episode', episode_nstep_loss))

        #         # histogram summary for mean stddev monitoring
        #         # gs.append(tf.histogram_summary('Nstep_Loss/Episode', episode_nstep_loss))


        #         summary_vars = [episode_total_reward, episode_q,
        #                         episode_step, episode_l2_loss, episode_1step_loss, episode_nstep_loss]
        #         summary_placeholders = [tf.placeholder(tf.float32) for _ in
        #                                 range(len(summary_vars))]
        #         update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
        #                       range(len(summary_vars))]
        #         summary_op = tf.summary.merge(gs)
        #         return summary_placeholders, update_ops, summary_op


# summary manager for SAC

class SummaryManager_HIRO(object):
    """ Implementation of EC_SAC in HIRO architecture
    """
    def __init__(self, sess, obs_shape_list, summary_writer):
        self.sess = sess
        _obs_shape_list = obs_shape_list
        self.summary_writer = summary_writer
        action_shape = (1,8)
        self.BS = 1
        # self.full_stt_rms = RunningMeanStd(shape=obs_shape_list[1])
        self.s_t0_rms = RunningMeanStd(shape=_obs_shape_list[0]) # (100,100,3)
        self.s_t1_rms = RunningMeanStd(shape=_obs_shape_list[1]) # (7,) pos
        self.s_t2_rms = RunningMeanStd(shape=_obs_shape_list[2]) # (7,) vel
        self.s_t3_rms = RunningMeanStd(shape=_obs_shape_list[3]) # (7,) eff
        self.s_t4_rms = RunningMeanStd(shape=_obs_shape_list[4]) # (1,) grip
        self.s_t5_rms = RunningMeanStd(shape=_obs_shape_list[5]) # (7,) ee
        self.s_t6_rms = RunningMeanStd(shape=_obs_shape_list[6]) # (3,) aux
        self.a_t_rms = RunningMeanStd(shape=action_shape) # (3,) aux

    def setup_state_summary(self):
        ols = []

        with tf.variable_scope('image_observation'): # for PPO Actor
            self.step_obs_ph = tf.placeholder(shape=(self.BS,)+OBS_SHAPE, dtype=tf.float32, name='step_obs')

        with tf.variable_scope('full_states'): # for PPO Critic
            self.step_stt_pos_ph = tf.placeholder(shape=(self.BS,)+POS, dtype=tf.float32, name='state_pos')
            self.step_stt_vel_ph = tf.placeholder(shape=(self.BS,)+VEL, dtype=tf.float32, name='state_vel')
            self.step_stt_eff_ph = tf.placeholder(shape=(self.BS,)+EFF, dtype=tf.float32, name='state_eff')
            self.step_stt_obj_ph = tf.placeholder(shape=(self.BS,)+OBJ, dtype=tf.float32, name='state_obj')

        with tf.variable_scope('state_summaries'):
            ols.append(tf.summary.image('step_obs', self.step_obs_ph))
            ols.append(tf.summary.histogram('state_pos', self.step_stt_pos_ph)) 
            ols.append(tf.summary.histogram('state_vel', self.step_stt_vel_ph)) 
            ols.append(tf.summary.histogram('state_eff', self.step_stt_eff_ph)) 
            ols.append(tf.summary.histogram('state_obj', self.step_stt_obj_ph)) 


        self._state_summary_op = tf.summary.merge(ols)

                        #     0      1          2       3         4       5       6  
    def update_state_summary(self, step, step_obs, pos_stt, vel_stt, eff_stt, obs_stt): # inputs are numpy ndarray type

        # reshape observations
        step_obs = np.reshape(step_obs,(-1,100,100,3))
        
        pos_stt = np.reshape(pos_stt,(-1,7))
        vel_stt = np.reshape(vel_stt,(-1,7))
        eff_stt = np.reshape(eff_stt,(-1,7))
        obs_stt = np.reshape(obs_stt,(-1,3))


        _crit_state_arr = np.hstack([pos_stt,vel_stt])

        state_summary = self.sess.run(self._state_summary_op, feed_dict={
            self.step_obs_ph : step_obs,
            self.step_stt_pos_ph : pos_stt,
            self.step_stt_vel_ph : vel_stt,
            self.step_stt_eff_ph : eff_stt,
            self.step_stt_obj_ph : obs_stt,
            })

        self.summary_writer.add_summary(state_summary, step)


    def setup_stat_summary(self): # inputs are numpy ndarray type
        ops = []
        names = []
        stat_summary_list = []


        # porting from np to tf is required
        # ops += [tf.reduce_mean(self.obs_rms.mean), tf.reduce_mean(self.obs_rms.std)]
        with tf.variable_scope('obs_stat_summary'):
            self.st0_mean = tf.placeholder(shape=(), dtype=tf.float32, name='s_t0_rms_mean')
            self.st0_std= tf.placeholder(shape=(), dtype=tf.float32, name='s_t0_rms_std')

            # self.goal_obs_mean = tf.placeholder(shape=(), dtype=tf.float32, name='goal_obs_rms_mean')
            # self.goal_obs_std= tf.placeholder(shape=(), dtype=tf.float32, name='goal_obs_rms_std')

            # ops +=[self.st0_mean, self.st0_std, 
            #         self.goal_obs_mean, self.goal_obs_std]

            ops +=[self.st0_mean, self.st0_std]


            names +=['s_t0_rms_mean', 's_t0_rms_std']

            stats_ops = ops
            stats_names = names

            for op, name in zip(stats_ops, stats_names):
                stat_summary_list.append(tf.summary.scalar(name,op))

        ops2 = []
        names2 = []

        with tf.variable_scope('state_stat_summary'):

            self.st1_mean = tf.placeholder(shape=(), dtype=tf.float32, name='s_t1_rms_mean')
            self.st1_std= tf.placeholder(shape=(), dtype=tf.float32, name='s_t1_rms_std')

            self.st2_mean = tf.placeholder(shape=(), dtype=tf.float32, name='s_t2_rms_mean')
            self.st2_std= tf.placeholder(shape=(), dtype=tf.float32, name='s_t2_rms_std')

            self.st3_mean = tf.placeholder(shape=(), dtype=tf.float32, name='s_t3_rms_mean')
            self.st3_std= tf.placeholder(shape=(), dtype=tf.float32, name='s_t3_rms_std')

            self.st4_mean = tf.placeholder(shape=(), dtype=tf.float32, name='s_t4_rms_mean')
            self.st4_std= tf.placeholder(shape=(), dtype=tf.float32, name='s_t4_rms_std')
                
            # self.goal0_mean = tf.placeholder(shape=(), dtype=tf.float32, name='goal_s0_rms_mean')
            # self.goal0_std= tf.placeholder(shape=(), dtype=tf.float32, name='goal_s1_rms_std')

            # self.goal1_mean = tf.placeholder(shape=(), dtype=tf.float32, name='goal_s1_rms_mean')
            # self.goal1_std= tf.placeholder(shape=(), dtype=tf.float32, name='goal_s1_rms_std')

            ops2 +=[self.st1_mean, self.st1_std, 
                    self.st2_mean, self.st2_std, 
                    self.st3_mean, self.st3_std,
                    self.st4_mean, self.st4_std]

                    # self.goal0_mean, self.goal0_std, 
                    # self.goal1_mean, self.goal1_std]

            names2 +=['s_t1_rms_mean', 's_t1_rms_std', 
                    's_t2_rms_mean', 's_t2_rms_std', 
                    's_t3_rms_mean', 's_t3_rms_std',
                    's_t4_rms_mean', 's_t4_rms_std']
                    # 'goal_s0_rms_mean', 'goal_s0_rms_std',
                    # 'goal_s1_rms_mean', 'goal_s1_rms_std']


            for op2, name2 in zip(ops2, names2):
                stat_summary_list.append(tf.summary.scalar(name2,op2))


        self.stat_summary_op = tf.summary.merge(stat_summary_list)

    def update_stat_summary(self, step): # inputs are numpy ndarray type

        stat_summary = self.sess.run(self.stat_summary_op,
                                        feed_dict={

        self.st0_mean: np.mean(self.s_t0_rms.mean), 
        self.st0_std: np.mean(self.s_t0_rms.std), 
        self.st1_mean: np.mean(self.s_t1_rms.mean), 
        self.st1_std: np.mean(self.s_t1_rms.std), 
        self.st2_mean: np.mean(self.s_t2_rms.mean), 
        self.st2_std: np.mean(self.s_t2_rms.std),
        self.st3_mean: np.mean(self.s_t3_rms.mean), 
        self.st3_std: np.mean(self.s_t3_rms.std),
        self.st4_mean: np.mean(self.s_t4_rms.mean), 
        self.st4_std: np.mean(self.s_t4_rms.std), 

        })

        self.summary_writer.add_summary(stat_summary, step)


    def setup_train_summary(self): #setpup 
        tls = []

        with tf.variable_scope('train_summary'): # for PPO Actor
            self.ep_ret_ph = tf.placeholder(shape=(), dtype=tf.float32, name='episode_return')
            self.ep_len_ph = tf.placeholder(shape=(), dtype=tf.float32, name='episode_length')

        with tf.variable_scope('train_summary'):
            tls.append(tf.summary.scalar('episode_return', self.ep_ret_ph))
            tls.append(tf.summary.scalar('episode_length', self.ep_len_ph)) 


        self.epi_summary_op = tf.summary.merge(tls)



    def update_train_summary(self, epi_ret, epi_len, episode): # inputs are numpy ndarray type
        stat_summary = self.sess.run(self.epi_summary_op, feed_dict={
                                    self.ep_ret_ph: epi_ret, 
                                    self.ep_len_ph: epi_len, })

        self.summary_writer.add_summary(stat_summary, episode)

    def setup_sac_summary(self): #setpup
        pls = []

        # approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
        # approx_ent = tf.reduce_mean(-logp)

        with tf.variable_scope('sac_summary'): # for PPO Actor
            self.q1val_ph = tf.placeholder(shape=(), dtype=tf.float32, name='Q1Vals')
            self.q2val_ph = tf.placeholder(shape=(), dtype=tf.float32, name='Q2Vals')
            self.vval_ph = tf.placeholder(shape=(), dtype=tf.float32, name='VVals')
            self.log_pi_ph = tf.placeholder(shape=(), dtype=tf.float32, name='LogPi')
            self.pi_loss_ph = tf.placeholder(shape=(), dtype=tf.float32, name='LossPi')
            self.q1_loss_ph = tf.placeholder(shape=(), dtype=tf.float32, name='LossQ1')
            self.q2_loss_ph = tf.placeholder(shape=(), dtype=tf.float32, name='LossQ2')
            self.v_loss_ph = tf.placeholder(shape=(), dtype=tf.float32, name='LossV')
            self.alpha_ph = tf.placeholder(shape=(), dtype=tf.float32, name='Alpha')

        # with tf.variable_scope('full_states'): # for PPO Critic
        #     self.step_stt_pos_ph = tf.placeholder(shape=(self.BS,)+POS, dtype=tf.float32, name='state_pos')
        #     self.step_stt_vel_ph = tf.placeholder(shape=(self.BS,)+VEL, dtype=tf.float32, name='state_vel')
        #     self.step_stt_eff_ph = tf.placeholder(shape=(self.BS,)+EFF, dtype=tf.float32, name='state_eff')
        #     self.step_stt_obj_ph = tf.placeholder(shape=(self.BS,)+OBJ, dtype=tf.float32, name='state_obj')
       
        with tf.variable_scope('sac_summary'):
            pls.append(tf.summary.scalar('Q1Vals', self.q1val_ph))
            pls.append(tf.summary.scalar('Q2Vals', self.q2val_ph)) 
            pls.append(tf.summary.scalar('VVals', self.vval_ph)) 
            pls.append(tf.summary.scalar('LogPi', self.log_pi_ph)) 
            pls.append(tf.summary.scalar('LossPi', self.pi_loss_ph)) 
            pls.append(tf.summary.scalar('LossQ1', self.q1_loss_ph)) 
            pls.append(tf.summary.scalar('LossQ2', self.q2_loss_ph)) 
            pls.append(tf.summary.scalar('LossV', self.v_loss_ph)) 
            pls.append(tf.summary.scalar('Alpha', self.alpha_ph)) 

        self.sac_summary_op = tf.summary.merge(pls)



    def update_sac_summary(self, Q1Vals, Q2Vals, VVals, LogPi, LossPi, LossQ1, LossQ2, LossV, Alpha, episode): # inputs are numpy ndarray type
        stat_summary = self.sess.run(self.sac_summary_op, feed_dict={
                                    self.q1val_ph: Q1Vals, 
                                    self.q2val_ph: Q2Vals, 
                                    self.vval_ph: VVals, 
                                    self.log_pi_ph: LogPi, 
                                    self.pi_loss_ph: LossPi, 
                                    self.q1_loss_ph: LossQ1, 
                                    self.q2_loss_ph: LossQ2, 
                                    self.v_loss_ph: LossV, 
                                    self.alpha_ph: Alpha,})

        self.summary_writer.add_summary(stat_summary, episode)


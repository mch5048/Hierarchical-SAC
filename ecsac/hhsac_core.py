import numpy as np
import tensorflow as tf
import rospy
from collections import OrderedDict as od
from state_action_space import *

EPS = 1e-8
CRIT_L2_REG = 1e-3
init_scale=np.sqrt(2)
xavier = tf.contrib.layers.xavier_initializer()
ortho = tf.keras.initializers.Orthogonal(init_scale, seed=0)

def ortho_init(scale=1.0):
    """
    Orthogonal initialization for the policy weights

    :param scale: (float) Scaling factor for the weights.
    :return: (function) an initialization function for the weights
    """

    # _ortho_init(shape, dtype, partition_info=None)
    def _ortho_init(shape, *_, **_kwargs):
        """Intialize weights as Orthogonal matrix.

        Orthogonal matrix initialization [1]_. For n-dimensional shapes where
        n > 2, the n-1 trailing axes are flattened. For convolutional layers, this
        corresponds to the fan-in, so this makes the initialization usable for
        both dense and convolutional layers.

        References
        ----------
        .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.
               "Exact solutions to the nonlinear dynamics of learning in deep
               linear
        """
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        gaussian_noise = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(gaussian_noise, full_matrices=False)
        weights = u if u.shape == flat_shape else v  # pick the one with the correct shape
        weights = weights.reshape(shape)
        return (scale * weights[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

# def _ln(input_tensor, gain, bias, epsilon=1e-5, axes=None):
#     """
#     Apply layer normalisation. from stable baaseline

#     :param input_tensor: (TensorFlow Tensor) The input tensor for the Layer normalization
#     :param gain: (TensorFlow Tensor) The scale tensor for the Layer normalization
#     :param bias: (TensorFlow Tensor) The bias tensor for the Layer normalization
#     :param epsilon: (float) The epsilon value for floating point calculations
#     :param axes: (tuple, list or int) The axes to apply the mean and variance calculation
#     :return: (TensorFlow Tensor) a normalizing layer
#     """
#     if axes is None:
#         axes = [1]
#     mean, variance = tf.nn.moments(input_tensor, axes=axes, keep_dims=True)
#     input_tensor = (input_tensor - mean) / tf.sqrt(variance + epsilon)
#     input_tensor = input_tensor * gain + bias
#     return input_tensor


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None, kernel_regularizer=None, kernel_initializer=ortho):
    for h in hidden_sizes[:-1]:
        x = (tf.layers.dense(x, units=h, activation=activation, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer))
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)

def cnn_feature_extractor(x, activation=tf.nn.relu, kernel_regularizer=None, kernel_initializer=ortho):
    x = tf.layers.conv2d(x, filters=32,kernel_size=8, strides=(4,4), activation=activation, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, name='actor_conv1')
    x = tf.layers.conv2d(x, filters=64,kernel_size=4, strides=(2,2), activation=activation, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, name='actor_conv2')
    x = tf.layers.conv2d(x, filters=64,kernel_size=3, strides=(1,1), activation=activation, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, name='actor_conv3')
    return tf.layers.flatten(x, name='actor_conv2fc')

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def mlp_gaussian_policy(x, a, hidden_sizes, activation, output_activation):
    act_dim = a.shape.as_list()[-1]
    net = mlp(x, list(hidden_sizes), activation, activation)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)
    log_alpha = tf.get_variable(name='log_alpha', initializer=0.0, dtype=np.float32)

    """
    Because algorithm maximizes trade-off of reward and entropy,
    entropy must be unique to state---and therefore log_stds need
    to be a neural network output instead of a shared-across-states
    learnable parameter vector. But for deep Relu and other nets,
    simply sticking an activationless dense layer at the end would
    be quite bad---at the beginning of training, a randomly initialized
    net could produce extremely large values for the log_stds, which
    would result in some actions being either entirely deterministic
    or too random to come back to earth. Either of these introduces
    numerical instability which could break the algorithm. To 
    protect against that, we'll constrain the output range of the 
    log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is 
    slightly different from the trick used by the original authors of
    SAC---they used tf.clip_by_value instead of squashing and rescaling.
    I prefer this approach because it allows gradient propagation
    through log_std where clipping wouldn't, but I don't know if
    it makes much of a difference.
    """
    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

# from here are what's necessary for HIRO.

def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
    """ categorical policy for gripper control
    """
    act_dim = action_space.n
    logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi

def mlp_deterministic_policy(stt, sub_goal, activation=tf.nn.relu, hidden_sizes=(512,256,256), output_activation=tf.nn.tanh, kernel_initializer=ortho_init(init_scale), kernel_regularizer=None):
    """ policy for high-level manager, TD3 policy
    """
    batch_size = sub_goal.shape.as_list()[0]
    sg_dim = sub_goal.shape.as_list()[-1]
    net = mlp(stt, list(hidden_sizes), activation=activation, kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer)
    mu = tf.layers.dense(net, sg_dim, activation=output_activation, kernel_initializer=kernel_initializer)

    return mu, net, batch_size

def cnn_gaussian_policy_with_logits(obs, act, goal, activation=tf.nn.relu, output_activation=None, kernel_initializer=ortho_init(init_scale)):
    """ policy for low-level controller, SAC policy
    Note that the gripper is actuated by categorical policy.
    """
    grip_dim = 2 # on/off
    act_dim = act.shape.as_list()[-1]
    _feat = cnn_feature_extractor(obs, activation)
    _feat = tf.concat([_feat, goal], axis=-1)
    _feat = tf.layers.dense(_feat, units=256, activation=activation, kernel_initializer=kernel_initializer)
    _feat = tf.layers.dense(_feat, units=256, activation=activation, kernel_initializer=kernel_initializer)
    # logit action for controlling gripper (on/off)
    logits = tf.layers.dense(_feat, units=grip_dim, activation=activation, kernel_initializer=kernel_initializer)
    logp_g_all = tf.nn.log_softmax(logits)
    # pi_g = tf.squeeze(tf.multinomial(logits,1), axis=1)
    pi_g = tf.multinomial(logits,1)
    pi_g_f = tf.cast(pi_g, tf.float32)
    logp_pi_g = tf.reduce_sum(tf.one_hot(pi_g, depth=grip_dim) * logp_g_all, axis=1)

    mu = tf.layers.dense(_feat, units=act_dim - 1, activation=output_activation, kernel_initializer=kernel_initializer)
    mu = mu * a_scale[:-1] + a_mean[:-1]
    mu = tf.minimum(a_high[:-1], tf.maximum(a_low[:-1], mu))
    log_std = tf.layers.dense(_feat, act_dim - 1, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) #Squash & rescale log_std
    std = tf.exp(log_std) # retrieve std
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    pi = tf.minimum(a_high[:-1], tf.maximum(a_low[:-1], pi))
    logp_pi = gaussian_likelihood(pi, mu, log_std)

    return mu, pi, logp_pi, std, pi_g_f, logp_pi_g

def cnn_gaussian_policy(obs, act, goal, activation=tf.nn.relu, output_activation=None):
    """ policy for low-level controller, SAC policy
    """
    act_dim = act.shape.as_list()[-1]
    log_alpha = tf.get_variable(name='log_alpha', initializer=0.0, dtype=np.float32)
    _feat = cnn_feature_extractor(obs, activation)
    _feat = tf.concat([_feat, goal], axis=-1)
    _feat = tf.layers.dense(_feat, units=256, activation=activation, kernel_initializer=ortho_init(init_scale))
    _feat = tf.layers.dense(_feat, units=256, activation=activation, kernel_initializer=ortho_init(init_scale))
    # logit action for controlling gripper (on/off)
    
    mu = tf.layers.dense(_feat, act_dim, activation=output_activation)
    mu = mu * a_scale + a_mean
    mu = tf.minimum(a_high, tf.maximum(a_low, mu))
    log_std = tf.layers.dense(_feat, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) #Squash & rescale log_std
    std = tf.exp(log_std) # retrieve std
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    pi = tf.minimum(a_high, tf.maximum(a_low, pi))
    logp_pi = gaussian_likelihood(pi, mu, log_std)

    return mu, pi, logp_pi, std

def apply_squashing_func(mu, pi, logp_pi):
    """apply tanh activation"""
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


"""
Actor-Critics for manager class : mu_hi, deterministic policy
"""


def mlp_manager_actor_critic(meas_stt, sub_goal, aux_stt, action_space=None, hidden_sizes=(200,200), activation=tf.nn.relu, 
                     output_activation=tf.tanh, policy=mlp_deterministic_policy):
    """ actor-critic for TD3
        args: 
            action_space : dict for the upper/lower limit of thes subgoal space
            meas_stt : 21-dim [joint_pos, joint_vel, joint_eff + gripper_pos]
            aux_stt : 
            => sub_goal_space = dict(ee_pos=ee_pos, ee_quat=ee_quat, ee_rpy=ee_rpy,
                      joint_p=joint_p, joint_v=joint_v, joint_e=joint_e)       
    """
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=CRIT_L2_REG)

    # policy
    with tf.variable_scope('pi'): # '/manager/main/pi/' , high-level policy only receives measured states!
        mu, _pre_act, _ = policy(meas_stt, sub_goal, activation=activation, hidden_sizes=hidden_sizes,
                            output_activation=output_activation, kernel_initializer=ortho, kernel_regularizer=kernel_regularizer)

    # policy reg losses
    preact_reg = tf.norm(_pre_act) 

    # make sure actions are in correct range
    mu = mu * s_scale + s_mean
    mu = tf.minimum(s_high, tf.maximum(s_low, mu))

    # vfs for TD3
    vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None, kernel_regularizer=kernel_regularizer), axis=1)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(tf.concat([meas_stt, aux_stt, sub_goal], axis=-1)) # sub_goal -> off-policy data
    with tf.variable_scope('q1', reuse=True):
        q1_pi = vf_mlp(tf.concat([meas_stt, aux_stt, mu], axis=-1)) # mu -> on-policy data (tensor-connected)
    with tf.variable_scope('q2'):
        q2 = vf_mlp(tf.concat([meas_stt, aux_stt, sub_goal], axis=-1)) # sub_goal -> off-policy data

    return mu, q1, q2, q1_pi, preact_reg


"""
Actor-Critics for controller class : mu_lo, stochastic policy
"""

def cnn_controller_actor_critic(meas_stt, obs, goal, act, aux_stt, action_space=None, hidden_sizes=(400,300), activation=tf.nn.relu, 
                    output_activation=None, policy=cnn_gaussian_policy_with_logits):
    """ Define actor-critic for controller policy ; actor: cnn critic: mlp
    """
    kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=CRIT_L2_REG)
    # policy
    with tf.variable_scope('pi'):
        _mu, _pi, _logp_pi, std, pi_g, _ = policy(obs, act, goal, activation, output_activation, kernel_initializer=ortho)
        # poliy reg losses 
        preact_reg = tf.norm(_mu)
        std_reg = tf.norm(std)
        _mu, _pi, logp_pi = apply_squashing_func(_mu, _pi, _logp_pi)
        mu = tf.concat([_mu, pi_g], axis=-1)
        pi = tf.concat([_pi, pi_g], axis=-1)


    # value ftns
    vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None, kernel_regularizer=kernel_regularizer), axis=1)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(tf.concat([meas_stt, aux_stt, goal, act], axis=-1))
    with tf.variable_scope('q1', reuse=True): # same as TD3 value ftn for policy learning.
        q1_pi = vf_mlp(tf.concat([meas_stt, aux_stt, goal, pi], axis=-1))
    with tf.variable_scope('q2'):
        q2 = vf_mlp(tf.concat([meas_stt, aux_stt, goal, act], axis=-1))
    with tf.variable_scope('q2', reuse=True):
        q2_pi = vf_mlp(tf.concat([meas_stt, aux_stt, goal, pi], axis=-1))

    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, _, pi_g, {'preact_reg':preact_reg, 'std_reg':std_reg}


# def cnn_controller_actor_critic(stt, obs, goal, act, aux, action_space=None, hidden_sizes=(512,256,256), activation=tf.nn.relu, 
#                      output_activation=None, policy=cnn_gaussian_policy, isCartesian=True):
#     """ Define actor-critic for controller policy ; actor: cnn critic: mlp
#     """
#     kernel_regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

#     # policy
#     with tf.variable_scope('pi'):
#         _mu, _pi, _logp_pi, std = policy(obs, act, goal, activation, output_activation)
#         mu, pi, logp_pi = apply_squashing_func(_mu, _pi, _logp_pi)

#     # poliy reg losses 
#     preact_reg = tf.norm(_mu)
#     std_reg = tf.norm(std)
#     # value ftns
#     vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None, kernel_regularizer=kernel_regularizer), axis=1)
#     with tf.variable_scope('q1'):
#         q1 = vf_mlp(tf.concat([stt, goal, act, aux], axis=-1))
#     with tf.variable_scope('q1', reuse=True): # same as TD3 value ftn for policy learning.
#         q1_pi = vf_mlp(tf.concat([stt, goal, pi, aux], axis=-1))
#     with tf.variable_scope('q2'):
#         q2 = vf_mlp(tf.concat([stt, goal, act, aux], axis=-1))
#     with tf.variable_scope('q2', reuse=True):
#         q2_pi = vf_mlp(tf.concat([stt, goal, pi, aux], axis=-1))
#     with tf.variable_scope('v'):
#         v = vf_mlp(tf.concat([stt, goal, aux], axis=-1))

#     return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v, {'preact_reg':preact_reg, 'std_reg':std_reg}
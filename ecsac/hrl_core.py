import numpy as np
import tensorflow as tf
import rospy
EPS = 1e-8

ee_pos = {'x':{'lo': 0.3397,'mean': 0.5966,'hi': 0.9697},
            'y':{'lo': -0.3324,'mean': 0.0274,'hi': 0.3162},
            'z':{'lo': 0.0200,'mean': 0.1081,'hi': 0.7638}}

ee_quat = {'x':{'lo': -0.0734,'mean': 0.9710,'hi': 1.0000},
            'y':{'lo': -0.4876,'mean': -0.0196,'hi': 0.2272},
            'z':{'lo': -0.3726,'mean': -0.0222,'hi': 0.6030},
            'w':{'lo': -0.9128,'mean': 0.0462,'hi': 0.9398}}

# roll should be considered for its absolute value (2.8~3.14)
# should compare with absolute value of the roll
# TODO: if ee_quat is not effective, replace it with ee_rpy
ee_rpy = {'r':{'lo': 2.8000,'mean': 3.0000,'hi': 3.1400},
            'p':{'lo': -0.4000,'mean': 0.0,'hi': 0.4000},
            'y':{'lo': -0.4500,'mean': 0.0,'hi': 0.4500}}

joint_p = {'j1':{'lo': -0.6110,'mean': 0.000,'hi': 0.6110},
            'j2':{'lo': -1.1530,'mean': -0.7750,'hi': 0.0000},
            'j3':{'lo': -1.6550,'mean': -0.321,'hi': 0.0000},
            'j4':{'lo': 0.5186,'mean': 1.1511,'hi': 2.191},
            'j5':{'lo': -1.369,'mean': 0.0123,'hi': 1.4400},
            'j6':{'lo': -1.538,'mean': 0.7484,'hi': 1.3150},
            'j7':{'lo': -2.500,'mean': -1.804,'hi': -1.00}}
# joint vels and efforts when imobile
# velocity: [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001]
# effort: [0.156, -30.78, -6.676, -9.88, 2.444, 0.2, 0.12]

joint_v = {'j1':{'lo': -1.000,'mean': 0.000,'hi': 1.000},
            'j2':{'lo': -0.6000,'mean': 0.000,'hi': 0.6000},
            'j3':{'lo': -1.655,'mean': -0.321,'hi': 0.0000},
            'j4':{'lo': -1.000,'mean': 0.000,'hi': 1.000},
            'j5':{'lo': -1.200,'mean': 0.000,'hi': 1.200},
            'j6':{'lo': -1.500,'mean': 0.000,'hi': 1.000},
            'j7':{'lo': -2.200,'mean': 0.000,'hi': 1.900}}

joint_e = {'j1':{'lo': -2.500,'mean': 0.000,'hi': 2.500},
            'j2':{'lo': -42.00,'mean': -30.00,'hi': -14.83},
            'j3':{'lo': -15.63,'mean': -6.676,'hi': -1.564},
            'j4':{'lo': -15.74,'mean': -10.22,'hi': 0.200},
            'j5':{'lo': -2.412,'mean': 2.000,'hi': 3.312},
            'j6':{'lo': -1.100,'mean': 0.200,'hi': 2.520},
            'j7':{'lo': -0.700,'mean': 0.05,'hi': 1.228}}

grip_pos = {'pos':{'lo': 0.0,'mean': 0.022,'hi': 0.044}}

act_space = {'j1':{'lo': -0.850,'mean': 0.000,'hi': 0.850},
                'j2':{'lo': -0.800,'mean': -0.100,'hi': 0.650},
                'j3':{'lo': -0.600,'mean': -0.300,'hi': 0.630},
                'j4':{'lo': -0.870,'mean': 0.000,'hi': 0.800},
                'j5':{'lo': -1.200,'mean': 0.000,'hi': 1.200},
                'j6':{'lo': -1.500,'mean': 0.000,'hi': 1.500},
                'j7':{'lo': -1.500,'mean': 0.000,'hi': 1.500},
                'grip':{'lo': -1.500,'mean': 0.000,'hi': 1.500}}

min_tensor = [joint_p['j1']['lo'], joint_p['j2']['lo'], joint_p['j3']['lo'], joint_p['j4']['lo'], joint_p['j5']['lo'], joint_p['j6']['lo'], joint_p['j7']['lo'],
            joint_v['j1']['lo'], joint_v['j2']['lo'], joint_v['j3']['lo'], joint_v['j4']['lo'], joint_v['j5']['lo'], joint_v['j6']['lo'], joint_v['j7']['lo'],
            joint_e['j1']['lo'], joint_e['j2']['lo'], joint_e['j3']['lo'], joint_e['j4']['lo'], joint_e['j5']['lo'], joint_e['j6']['lo'], joint_e['j7']['lo'],
            grip_pos['pos']['lo'], ee_pos['x']['lo'], ee_pos['y']['lo'], ee_pos['z']['lo'], ee_quat['x']['lo'], ee_quat['y']['lo'], ee_quat['z']['lo'], ee_quat['w']['lo']]

max_tensor = [joint_p['j1']['hi'], joint_p['j2']['hi'], joint_p['j3']['hi'], joint_p['j4']['hi'], joint_p['j5']['hi'], joint_p['j6']['hi'], joint_p['j7']['hi'],
            joint_v['j1']['hi'], joint_v['j2']['hi'], joint_v['j3']['hi'], joint_v['j4']['hi'], joint_v['j5']['hi'], joint_v['j6']['hi'], joint_v['j7']['hi'],
            joint_e['j1']['hi'], joint_e['j2']['hi'], joint_e['j3']['hi'], joint_e['j4']['hi'], joint_e['j5']['hi'], joint_e['j6']['hi'], joint_e['j7']['hi'],
            grip_pos['pos']['hi'], ee_pos['x']['hi'], ee_pos['y']['hi'], ee_pos['z']['hi'], ee_quat['x']['hi'], ee_quat['y']['hi'], ee_quat['z']['hi'], ee_quat['w']['hi']] 

mean_tensor = [joint_p['j1']['mean'], joint_p['j2']['mean'], joint_p['j3']['mean'], joint_p['j4']['mean'], joint_p['j5']['mean'], joint_p['j6']['mean'], joint_p['j7']['mean'],
            joint_v['j1']['mean'], joint_v['j2']['mean'], joint_v['j3']['mean'], joint_v['j4']['mean'], joint_v['j5']['mean'], joint_v['j6']['mean'], joint_v['j7']['mean'],
            joint_e['j1']['mean'], joint_e['j2']['mean'], joint_e['j3']['mean'], joint_e['j4']['mean'], joint_e['j5']['mean'], joint_e['j6']['mean'], joint_e['j7']['mean'],
            grip_pos['pos']['mean'], ee_pos['x']['mean'], ee_pos['y']['mean'], ee_pos['z']['mean'], ee_quat['x']['mean'], ee_quat['y']['mean'], ee_quat['z']['mean'], ee_quat['w']['mean']] 

act_lo = [act_space['j1']['lo'], act_space['j2']['lo'], act_space['j3']['lo'], act_space['j4']['lo'], act_space['j5']['lo'], act_space['j6']['lo'], act_space['j7']['lo'], act_space['grip']['lo']]
act_hi = [act_space['j1']['hi'], act_space['j2']['hi'], act_space['j3']['hi'], act_space['j4']['hi'], act_space['j5']['hi'], act_space['j6']['hi'], act_space['j7']['hi'], act_space['grip']['hi']]
act_mean = [act_space['j1']['mean'], act_space['j2']['mean'], act_space['j3']['mean'], act_space['j4']['mean'], act_space['j5']['mean'], act_space['j6']['mean'], act_space['j6']['mean'], act_space['grip']['mean']]

scale_tensor = list()
act_scale = list()

for idx in range(len(min_tensor)):
    scale_tensor.append(max_tensor[idx] - min_tensor[idx])

for idx in range(len(act_lo)):
    act_scale.append(act_hi[idx] - act_lo[idx])
'''

def cnn_feature_extractor(img_obs): # let's apply bn , adopted from net_utils
    activ = tf.nn.relu
    layer_1 = activ(conv(img_obs, 'actor_conv1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2)))
    layer_2 = activ(conv(layer_1, 'actor_conv2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2)))
    layer_3 = activ(conv(layer_2, 'actor_conv3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2)))
    layer_4 = conv_to_fc(layer_3)
    return layer_4
'''

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


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    init_scale=np.sqrt(2)
    for h in hidden_sizes[:-1]:
        x = (tf.layers.dense(x, units=h, activation=activation))
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation, kernel_initializer=ortho_init(init_scale))

def cnn_feature_extractor(x, activation=tf.nn.relu):
    init_scale=np.sqrt(2)
    x = tf.layers.conv2d(x, filters=32,kernel_size=8, strides=(4,4), activation=activation, kernel_initializer=ortho_init(init_scale), name='actor_conv1')
    x = tf.layers.conv2d(x, filters=64,kernel_size=4, strides=(2,2), activation=activation, kernel_initializer=ortho_init(init_scale), name='actor_conv2')
    x = tf.layers.conv2d(x, filters=64,kernel_size=3, strides=(1,1), activation=activation, kernel_initializer=ortho_init(init_scale), name='actor_conv3')
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

# def mlp_categorical_policy(x, a, hidden_sizes, activation, output_activation, action_space):
#     """ categorical policy for gripper control
#     """
#     act_dim = action_space.n
#     logits = mlp(x, list(hidden_sizes)+[act_dim], activation, None)
#     logp_all = tf.nn.log_softmax(logits)
#     pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
#     logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
#     logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
#     return pi, logp, logp_pi

def mlp_deterministic_policy(stt, goal, sub_goal, aux, activation=tf.nn.relu, hidden_sizes=(512,256,256), output_activation=tf.nn.tanh):
    """ policy for high-level manager, TD3 policy
    """
    batch_size = sub_goal.shape.as_list()[0]
    sg_dim = sub_goal.shape.as_list()[-1]
    net = mlp(tf.concat([stt,goal], axis=-1), list(hidden_sizes), activation=activation)
    mu = tf.layers.dense(net, sg_dim, activation=output_activation)

    return mu, batch_size

def cnn_gaussian_policy(obs, act, goal, activation=tf.nn.relu, output_activation=None):
    """ policy for low-level controller, SAC policy
    """
    act_dim = act.shape.as_list()[-1]
    log_alpha = tf.get_variable(name='log_alpha', initializer=0.0, dtype=np.float32)
    _feat = cnn_feature_extractor(obs, activation)
    _feat = tf.concat([_feat, goal], axis=-1)
    _feat = tf.layers.dense(_feat, units=256, activation=activation)
    _feat = tf.layers.dense(_feat, units=256, activation=activation)
    # parameterized mean and stddev
    mu = tf.layers.dense(_feat, act_dim, activation=output_activation)
    mu = mu * act_scale + act_mean
    # pi = pi * act_scale + act_mean
    mu = tf.minimum(act_hi, tf.maximum(act_lo, mu))
    log_std = tf.layers.dense(_feat, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1) #Squash & rescale log_std
    std = tf.exp(log_std) # retrieve std
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    pi = tf.minimum(act_hi, tf.maximum(act_lo, pi))
    logp_pi = gaussian_likelihood(pi, mu, log_std)

    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


"""
Actor-Critics for manager class : mu_hi, deterministic policy
"""


def reverse_action(norm_action, space):
    """ relocate and rescale the aciton of manager policy (subgoal) to desirable values.    
        subgoal_policy = tensor of shape (1, 29(full_stt + ee + gripper) )
        subgoal_dim = dict(ee_pos=ee_pos, ee_quat=ee_quat, ee_rpy=ee_rpy,
                      joint_p=joint_p, joint_v=joint_v, joint_e=joint_e) 
    """
    assert NotImplementedError

def mlp_manager_actor_critic(stt, goal, sub_goal, aux, action_space, hidden_sizes=(400,300), activation=tf.nn.relu, 
                     output_activation=tf.tanh, policy=mlp_deterministic_policy):
    """ actor-critic for TD3
        args: 
            action_space : dict for the upper/lower limit of thes subgoal space
            => sub_goal_space = dict(ee_pos=ee_pos, ee_quat=ee_quat, ee_rpy=ee_rpy,
                      joint_p=joint_p, joint_v=joint_v, joint_e=joint_e)       
    """
    sg_dim = sub_goal.shape.as_list()[-1]
    # policy
    with tf.variable_scope('pi'): # '/manager/main/pi/'
        mu, batch_size = policy(stt, goal, sub_goal, aux, activation=activation, hidden_sizes=hidden_sizes, output_activation=output_activation)

    # make sure actions are in correct range
    # action_scale = action_space[1]
    # mu = normalize_action(action=mu, space=action_space, batch_size=batch_size)

    # low = list()
    # high = list()
    # mean = list()
    # scale = list()        
    # for key, value in action_space.items():
    #     for k, v in value.items():
    #         low.append(v['lo'])
    #         high.append(v['hi'])
    #         mean.append(v['mean'])
    #         scale.append((v['hi']-v['lo'])/2)
    # here, reloc is the mean
    mu = mu * scale_tensor + mean_tensor
    mu = tf.minimum(max_tensor, tf.maximum(min_tensor, mu))

    # vfs for TD3
    vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q1'):
        q1 = vf_mlp(tf.concat([stt, goal, sub_goal, aux], axis=-1))
    with tf.variable_scope('q1', reuse=True):
        q1_pi = vf_mlp(tf.concat([stt, goal, mu, aux], axis=-1))
    with tf.variable_scope('q2'):
        q2 = vf_mlp(tf.concat([stt, goal, sub_goal, aux], axis=-1))

    return mu, q1, q2, q1_pi



"""
Actor-Critics for controller class : mu_lo, stochastic policy
"""

def cnn_controller_actor_critic(stt, obs, goal, act, aux, action_space,hidden_sizes=(512,256,256), activation=tf.nn.relu, 
                     output_activation=None, policy=cnn_gaussian_policy, isCartesian=True):
    """ Define actor-critic for controller policy ; actor: cnn critic: mlp
    """
    # policy
    with tf.variable_scope('pi'):
        mu, pi, logp_pi = policy(obs, act, goal, activation, output_activation)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # value ftns
    vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q1'):
        q1 = tf.squeeze(mlp(tf.concat([stt, goal, act, aux], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q1', reuse=True): # same as TD3 value ftn for policy learning.
        q1_pi = tf.squeeze(mlp(tf.concat([stt, goal, pi, aux], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q2'):
        q2 = tf.squeeze(mlp(tf.concat([stt, goal, act, aux], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('q2', reuse=True):
        q2_pi = tf.squeeze(mlp(tf.concat([stt, goal, pi, aux], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('v'):
        v = tf.squeeze(mlp(tf.concat([stt, goal, aux], axis=-1), list(hidden_sizes)+[1], activation, None), axis=1)

    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v

# def mlp_manager_actor_critic(stt, act, action_space,hidden_sizes=(512,256,256), activation=tf.nn.relu, 
#                      output_activation=None, policy=mlp
#                      _gaussian_policy, isCartesian=True):
#     """ Define actor-critic for manager policy, both actor and critic consist of mlps
#     """   
#     # policy
#     with tf.variable_scope('pi'):
#         mu, pi, logp_pi, log_alpha = policy(obs, act, activation, output_activation)
#         mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

#     # retrieve learnable alpha
#     alpha = tf.exp(log_alpha)
#     # make sure actions are in correct range
#     action_scale = action_space[1]
#     # action_scale = 0.15 if isCartesian else 1.0
#     mu *= action_scale
#     pi *= action_scale

#     # vfs
#     vf_mlp = lambda x : tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
#     with tf.variable_scope('q1'):
#         q1 = vf_mlp(tf.concat([stt,act], axis=-1))
#     with tf.variable_scope('q1', reuse=True): # same as TD3
#         q1_pi = vf_mlp(tf.concat([stt, pi], axis=-1))
#     with tf.variable_scope('q2'):
#         q2 = vf_mlp(tf.concat([stt,act], axis=-1))
#     with tf.variable_scope('q2', reuse=True): # same as TD3
#         q2_pi = vf_mlp(tf.concat([stt, pi], axis=-1))
#     with tf.variable_scope('v'):
#         v = vf_mlp(stt)
#     return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v, log_alpha, alpha


from collections import OrderedDict as od

ee_pos = od()
ee_pos['x'] = {'lo': 0.3397,'mean': 0.5966,'hi': 0.9697}
ee_pos['y'] = {'lo': -0.3324,'mean': 0.0274,'hi': 0.3162}
ee_pos['z'] = {'lo': 0.0200,'mean': 0.1081,'hi': 0.7638}

ee_quat = od()
ee_quat['x'] = {'lo': -0.0734,'mean': 0.9710,'hi': 1.0000}
ee_quat['y'] = {'lo': -0.4876,'mean': -0.0196,'hi': 0.2272}
ee_quat['z'] = {'lo': -0.3726,'mean': -0.0222,'hi': 0.6030}
ee_quat['w'] = {'lo': -0.9128,'mean': 0.0462,'hi': 0.9398}

# roll should be considered for its absolute value (2.8~3.14)
# should compare with absolute value of the roll
# TODO: if ee_quat is not effective, replace it with ee_rpy

ee_rpy = od()
ee_rpy['r'] = {'lo': 2.8000,'mean': 3.0000,'hi': 3.1400}
ee_rpy['p'] = {'lo': -0.4000,'mean': 0.0,'hi': 0.4000}
ee_rpy['y'] = {'lo': -0.4500,'mean': 0.0,'hi': 0.4500}

joint_p = od()
joint_p['j1'] = {'lo': -0.6110,'mean': 0.000,'hi': 0.6110}
joint_p['j2'] = {'lo': -1.1530,'mean': -0.7750,'hi': 0.0000}
joint_p['j3'] = {'lo': -1.6550,'mean': -0.321,'hi': 0.0000}
joint_p['j4'] = {'lo': 0.5186,'mean': 1.1511,'hi': 2.191}
joint_p['j5'] = {'lo': -1.369,'mean': 0.0123,'hi': 1.4400}
joint_p['j6'] = {'lo': -1.538,'mean': 0.7484,'hi': 1.3150}
joint_p['j7'] = {'lo': -2.500,'mean': -1.804,'hi': -1.00}

# joint vels and efforts when imobile
# velocity: [-0.001, -0.001, -0.001, -0.001, -0.001, -0.001, -0.001]
# effort: [0.156, -30.78, -6.676, -9.88, 2.444, 0.2, 0.12]

joint_v = od()
joint_v['j1'] = {'lo': -1.000,'mean': 0.000,'hi': 1.000}
joint_v['j2'] = {'lo': -0.6000,'mean': 0.000,'hi': 0.6000}
joint_v['j3'] = {'lo': -1.655,'mean': -0.321,'hi': 0.0000}
joint_v['j4'] = {'lo': -1.000,'mean': 0.000,'hi': 1.000}
joint_v['j5'] = {'lo': -1.200,'mean': 0.000,'hi': 1.200}
joint_v['j6'] = {'lo': -1.500,'mean': 0.000,'hi': 1.000}
joint_v['j7'] = {'lo': -2.200,'mean': 0.000,'hi': 1.900}

joint_e = od()
joint_e['j1'] = {'lo': -2.500,'mean': 0.000,'hi': 2.500}
joint_e['j2'] = {'lo': -42.00,'mean': -30.00,'hi': -14.83}
joint_e['j3'] = {'lo': -15.63,'mean': -6.676,'hi': -1.564}
joint_e['j4'] = {'lo': -15.74,'mean': -10.22,'hi': 0.200}
joint_e['j5'] = {'lo': -2.412,'mean': 2.000,'hi': 3.312}
joint_e['j6'] = {'lo': -1.100,'mean': 0.200,'hi': 2.520}
joint_e['j7'] = {'lo': -0.700,'mean': 0.05,'hi': 1.228}
grip_pos = {'pos':{'lo': 0.0,'mean': 0.022,'hi': 0.044}}

act_space = od()
act_space['j1'] = {'lo': -0.850,'mean': 0.000,'hi': 0.850}
act_space['j2'] = {'lo': -0.800,'mean': -0.100,'hi': 0.650}
act_space['j3'] = {'lo': -0.600,'mean': -0.300,'hi': 0.630}
act_space['j4'] = {'lo': -0.870,'mean': 0.000,'hi': 0.800}
act_space['j5'] = {'lo': -1.200,'mean': 0.000,'hi': 1.200}
act_space['j6'] = {'lo': -1.500,'mean': 0.000,'hi': 1.500}
act_space['j7'] = {'lo': -1.500,'mean': 0.000,'hi': 1.500}
act_space['grip'] = {'lo': 0.000,'mean': 0.000,'hi': 1.000} # binary input



st_space = [joint_p, joint_v, joint_e, grip_pos]
st_low = list()
st_high = list()
st_mean = list()

s_low = list()
s_high = list()
s_mean = list()
s_scale = list()        
for item in st_space:
    for k, v in item.items():
        s_low.append(v['lo'])
        s_high.append(v['hi'])
        s_mean.append(v['mean'])
        s_scale.append((v['hi']-v['lo'])/2)

a_low = list()
a_high = list()
a_mean = list()
a_scale = list()
for k, v in act_space.items():
    a_low.append(v['lo'])
    a_high.append(v['hi'])
    a_mean.append(v['mean'])
    a_scale.append((v['hi']-v['lo'])/2)

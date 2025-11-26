import numpy as np
import gymnasium as gym
import mujoco


class TactileObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, k=10):
        super().__init__(env)
        self.k = k
        self.env = env
        self.action_space = env.action_space

        self.base = env.unwrapped
        self.rendered = mujoco.Renderer(self.base.model)

        self.touch = self.base.touch
        self.proprioception = env.proprioception
        self.model = self.base.model

        self.BODY_WITH_SENSORS_IDS = set(self.touch.sensing_bodies())

        self.NUMBER_SENSORS = {}
        self.BODY_IDS = set()

        count = 0

        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if 'left_hand' == name:
                self.LEFT_HAND_ID = i
            elif 'right_hand' == name:
                self.RIGHT_HAND_ID = i
            elif 'right_fingers' == name:
                self.RIGHT_FINGERS_ID = i 
            elif 'left_fingers' == name:
                self.LEFT_FINGERS_ID = i

            if i > 1:
                self.BODY_IDS.add(i)
            
            if i in self.BODY_WITH_SENSORS_IDS:
                act_count = self.touch.get_sensor_count(i)
                self.NUMBER_SENSORS[i] = (count, count + act_count)
                count += act_count

        self.NAMES_JOINTS_Q = ['robot:right_shoulder_horizontal', 'robot:right_shoulder_ad_ab', 'robot:right_shoulder_rotation', 'robot:right_elbow', 'robot:right_hand1', 'robot:right_hand2', 'robot:right_hand3', 'robot:right_fingers', 'robot:left_shoulder_horizontal', 'robot:left_shoulder_ad_ab', 'robot:left_shoulder_rotation', 'robot:left_elbow']
        self.NAMES_JOINTS_TORQUE  = ['proprio:right_shoulder', 'proprio:right_elbow', 'proprio:right_wrist', 'proprio:right_fingers', 'proprio:left_shoulder', 'proprio:left_elbow', 'proprio:left_wrist', 'proprio:left_fingers']

        joint_names = self.proprioception.sensor_names['qpos']
        self.ids_joints_q = [i for i,name in enumerate(joint_names) if name in self.NAMES_JOINTS_Q]

        joint_names = self.proprioception.sensor_names['torque']
        self.ids_joints_torques = [i for i,name in enumerate(joint_names) if name in self.NAMES_JOINTS_TORQUE]

        dummy = self.get_obs_vector()
        obs_dim = dummy.shape[0]

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def get_k_sensors_from_contact(self,touch,contact_id,body_id,k=10):
        pos_abs = touch.get_contact_position_world(contact_id)
        pos_rel = touch.get_contact_position_relative(contact_id,body_id)

        k_sensors = touch.get_k_nearest_sensors(pos_abs,body_id,k)
        return pos_rel,k_sensors
    
    def get_hands_k_features_mean(self,touch, left_ids, right_ids, body_ids, 
                              k=10):
        '''
        It returns features for left and right hand.
        The output are two array of dimension (2k + 6)
        The first 3 elements are the relative position of the left hand contact for k sensors
        The next k elements are the forces for those k sensors.
        The next 3 elements are the rel.. for left fingers
        The next k elements are the forces for those k sensor.

        Same for right hand and fingers
        
        :param touch: Description
        :param left_ids: left_hand_id, left_fingers_id
        :param right_ids: right_hand_id, right_fingers_id
        :param body_ids: Description
        :param k: Description
        '''

        contacts = touch.get_contacts()
        contacts_clean = []

        left_contact_ids = set()
        right_contact_ids = set()

        for c_id, b_id, f in contacts:
            if b_id in left_ids:
                left_contact_ids.add(c_id)
            elif b_id in right_ids:
                right_contact_ids.add(c_id)
            else:
                contacts_clean.append([c_id,b_id,f])

        left_positions_hand = []
        left_features_hand = []
        right_positions_hand = []
        right_features_hand = []

        left_positions_fingers = []
        left_features_fingers = []
        right_positions_fingers = []
        right_features_fingers = []

        
        for c_id, b_id, _ in contacts_clean:
            if b_id in body_ids:
                #9 right_lower_arm
                #13 left_lower_arm
                if c_id in left_contact_ids and b_id != 13:
                    hand_id, fingers_id = left_ids
                    pos_rel, k_sensors = self.get_k_sensors_from_contact(
                        touch, c_id, hand_id, k
                    )
                    left_positions_hand.append(pos_rel)
                    left_features_hand.append(k_sensors)

                    pos_rel, k_sensors = self.get_k_sensors_from_contact(
                        touch, c_id, fingers_id, k
                    )
                    left_positions_fingers.append(pos_rel)
                    left_features_fingers.append(k_sensors)

                if c_id in right_contact_ids and b_id != 9:
                    hand_id, fingers_id = right_ids

                    pos_rel, k_sensors = self.get_k_sensors_from_contact(
                        touch, c_id, hand_id, k
                    )
                    right_positions_hand.append(pos_rel)
                    right_features_hand.append(k_sensors)

                    pos_rel, k_sensors = self.get_k_sensors_from_contact(
                        touch, c_id, fingers_id, k
                    )
                    right_positions_fingers.append(pos_rel)
                    right_features_fingers.append(k_sensors)

        
        #if we have many we just get the mean
        if len(left_positions_hand) > 0:
            left_pos_mean = np.mean(np.vstack(left_positions_hand), axis=0)
            left_feat_mean = np.mean(np.vstack(left_features_hand), axis=0)
            left_vec_hand = np.concatenate([left_pos_mean, left_feat_mean])

            left_pos_mean = np.mean(np.vstack(left_positions_fingers), axis=0)
            left_feat_mean = np.mean(np.vstack(left_features_fingers), axis=0)
            left_vec_fingers = np.concatenate([left_pos_mean, left_feat_mean])

            left_vec = np.concatenate([left_vec_hand, left_vec_fingers])
        else:
            left_vec = np.zeros(2*(3 + k))

        if len(right_positions_hand) > 0:
            right_pos_mean = np.mean(np.vstack(right_positions_hand), axis=0)
            right_feat_mean = np.mean(np.vstack(right_features_hand), axis=0)
            right_vec_hand = np.concatenate([right_pos_mean, right_feat_mean])

            right_pos_mean = np.mean(np.vstack(right_positions_fingers), axis=0)
            right_feat_mean = np.mean(np.vstack(right_features_fingers), axis=0)
            right_vec_fingers = np.concatenate([right_pos_mean, right_feat_mean])

            right_vec = np.concatenate([right_vec_hand, right_vec_fingers])
        else:
            right_vec = np.zeros(2*(3 + k))

        return left_vec, right_vec



    def get_joints_values(self,sensor_outputs,ids_q,ids_torque):
        q_pos = np.array([sensor_outputs['qpos'][i] for i in ids_q])
        q_vel = np.array([sensor_outputs['qvel'][i] for i in ids_q])
        torque = np.array([sensor_outputs['torques'][i] for i in ids_torque])

        return q_pos,q_vel,torque
    
    def get_obs_vector(self):
        touch = self.touch
        sensor_outputs = self.proprioception.sensor_outputs

        left_ids = [self.LEFT_HAND_ID,self.LEFT_FINGERS_ID]
        right_ids = [self.RIGHT_HAND_ID,self.RIGHT_FINGERS_ID]
        body_ids = self.BODY_IDS
        
        left,right = self.get_hands_k_features_mean(touch, left_ids, right_ids, body_ids, self.k)
        
        ids_q = self.ids_joints_q
        ids_torque = self.ids_joints_torques

        qpos,qvel,torque = self.get_joints_values(sensor_outputs,ids_q,ids_torque)

        res = np.concatenate([left,right,qpos,qvel,torque])

        return res

    def observation(self, obs):
        return self.get_obs_vector()
    
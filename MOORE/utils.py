from collections import OrderedDict
from metaworld import Benchmark, _make_tasks
from metaworld.envs import reward_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import metaworld.envs.mujoco.env_dict as env_dict
from metaworld.envs.mujoco.env_dict import *

class MT28(Benchmark):
    def __init__(self, seed=None):
        super().__init__()
        MT28_V2 = OrderedDict((
            ('assembly-v2', SawyerNutAssemblyEnvV2),
            ('disassemble-v2', SawyerNutDisassembleEnvV2),
            ('coffee-pull-v2', SawyerCoffeePullEnvV2),
            ('coffee-push-v2', SawyerCoffeePushEnvV2),
            ('door-open-v2', SawyerDoorEnvV2),
            ('door-close-v2', SawyerDoorCloseEnvV2),
            ('door-lock-v2', SawyerDoorLockEnvV2),
            ('door-unlock-v2', SawyerDoorUnlockEnvV2),
            ('drawer-open-v2', SawyerDrawerOpenEnvV2),
            ('drawer-close-v2', SawyerDrawerCloseEnvV2),
            ('faucet-open-v2', SawyerFaucetOpenEnvV2),
            ('faucet-close-v2', SawyerFaucetCloseEnvV2),
            ('handle-press-v2', SawyerHandlePressEnvV2),
            ('handle-pull-v2', SawyerHandlePullEnvV2),
            ('handle-press-side-v2', SawyerHandlePressSideEnvV2),
            ('handle-pull-side-v2', SawyerHandlePullSideEnvV2),
            ('peg-insert-side-v2', SawyerPegInsertionSideEnvV2),
            ('peg-unplug-side-v2', SawyerPegUnplugSideEnvV2),
            ('plate-slide-v2', SawyerPlateSlideEnvV2),
            ('plate-slide-back-v2', SawyerPlateSlideBackEnvV2),
            ('plate-slide-side-v2', SawyerPlateSlideSideEnvV2),
            ('plate-slide-back-side-v2', SawyerPlateSlideBackSideEnvV2),
            ('push-v2', SawyerPushEnvV2),
            ('push-back-v2', SawyerPushBackEnvV2),
            ('stick-push-v2', SawyerStickPushEnvV2),
            ('stick-pull-v2', SawyerStickPullEnvV2),
            ('window-open-v2', SawyerWindowOpenEnvV2),
            ('window-close-v2', SawyerWindowCloseEnvV2),
        ))
        self._train_classes = MT28_V2
        MT28_V2_ARGS_KWARGS = {
            key: dict(args=[],
                    kwargs={'task_id': list(env_dict.ALL_V2_ENVIRONMENTS.keys()).index(key)})
            for key, _ in MT28_V2.items()
        }
        train_kwargs = MT28_V2_ARGS_KWARGS
        _MT_OVERRIDE = dict(partially_observable=False)
        self._train_tasks = _make_tasks(self._train_classes, train_kwargs,
                                        _MT_OVERRIDE,
                                        seed=seed)
        
        self._test_classes = OrderedDict()
        self._test_tasks = []

def state_reward(env_name, state, action):
    computed_reward = 0
    
    if env_name == 'window-open-v2':
        obj = state[4:7]
        tcp = state[0:3] + np.array([0, 0, -0.045]) #approx
        target = state[36:39]
        obj_init_pos = target - np.array([0.2, 0, 0])            
        target_to_obj = (obj[0] - target[0])
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (obj_init_pos[0] - target[0])
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        target_radius = 0.05
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, target_radius),
            margin=abs(target_to_obj_init - target_radius),
            sigmoid='long_tail',
        )

        handle_radius = 0.02
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = 0.326954932 #approx
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        tcp_opened = 0
        object_grasped = reach

        computed_reward = 10 * reward_utils.hamacher_product(reach, in_place)
        
    if env_name == 'window-close-v2':
        obj = state[4:7]
        tcp = state[0:3] + np.array([0, 0, -0.045]) #approx
        target = state[36:39]    
        obj_init_pos = target - np.array([0.2, 0, 0])            
        target_to_obj = (obj[0] - target[0])
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (obj_init_pos[0] - target[0])
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        target_radius = 0.05
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, target_radius),
            margin=abs(target_to_obj_init - target_radius),
            sigmoid='long_tail',
        )

        handle_radius = 0.02
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = 0.3911952 #approx
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='gaussian',
        )
        tcp_opened = 0
        object_grasped = reach

        computed_reward = 10 * reward_utils.hamacher_product(reach, in_place)

    if env_name == 'drawer-open-v2':
        maxDist = 0.2
        _target_pos = state[36:39]
        init_tcp = np.array([0.00600364, 0.6004762, 0.14935033])

        gripper = state[:3]
        handle = state[4:7]

        handle_error = np.linalg.norm(handle - _target_pos)

        reward_for_opening = reward_utils.tolerance(
            handle_error,
            bounds=(0, 0.02),
            margin=maxDist,
            sigmoid='long_tail'
        )

        handle_pos_init = _target_pos + np.array([.0, maxDist, .0])
        scale = np.array([3., 3., 1.])
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init - init_tcp) * scale

        reward_for_caging = reward_utils.tolerance(
            np.linalg.norm(gripper_error),
            bounds=(0, 0.01),
            margin=np.linalg.norm(gripper_error_init),
            sigmoid='long_tail'
        )

        computed_reward = reward_for_caging + reward_for_opening
        computed_reward *= 5.0

    if env_name == 'drawer-close-v2':
        TARGET_RADIUS = 0.05
        target = state[36:39]

        tcp = np.array([0.00599208, 0.60039715, 0.14930194])
        init_tcp = np.array([[0.00600364, 0.6004762, 0.14935033]])
        obj_init_pos = np.array([target[0], 0.57999998, 0.14])

        obj = state[4:7]
        
        target_to_obj = (obj - target)
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (obj_init_pos - target)
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, TARGET_RADIUS),
            margin=abs(target_to_obj_init - TARGET_RADIUS),
            sigmoid='long_tail',
        )

        handle_reach_radius = 0.005
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_reach_radius),
            margin=abs(tcp_to_obj_init-handle_reach_radius),
            sigmoid='gaussian',
        )
        gripper_closed = min(max(0, action[-1]), 1)

        reach = reward_utils.hamacher_product(reach, gripper_closed)
        tcp_opened = 0
        object_grasped = reach

        reward = reward_utils.hamacher_product(reach, in_place)
        if target_to_obj <= TARGET_RADIUS + 0.015:
            reward = 1.

        computed_reward = reward * 10

    if env_name == 'assembly-v2':
        hand = state[:3]
        wrench = state[4:7]
        wrench_center = state[4:7] + np.array([-0.129983, 0.00067193, 0.00199226])  # env._get_site_pos('RoundNut')
        
        wrench_threshed = wrench.copy()
        WRENCH_HANDLE_LENGTH = 0.02
        threshold = WRENCH_HANDLE_LENGTH / 2.0
        if abs(wrench[0] - hand[0]) < threshold:
            wrench_threshed[0] = hand[0]

        def _reward_quat(state):
            # Ideal laid-down wrench has quat [.707, 0, 0, .707]
            # Rather than deal with an angle between quaternions, just approximate:
            ideal = np.array([0.707, 0, 0, 0.707])
            error = np.linalg.norm(state[7:11] - ideal)
            return max(1.0 - error/0.4, 0.0)

        reward_quat = _reward_quat(state)

        tcp = state[0:3] + np.array([0, 0, -0.045]) #approx
        obj = state[4:7]
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = 0.326954932 #approx
        handle_radius = 0.02
        reward_grab = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='gaussian',
        )
        _target_pos = state[36:39]  # env._target_pos

        def _reward_pos(wrench_center, target_pos):
            pos_error = target_pos - wrench_center

            radius = np.linalg.norm(pos_error[:2])

            aligned = radius < 0.02
            hooked = pos_error[2] > 0.0
            success = aligned and hooked

            threshold = 0.02 if success else 0.01
            target_height = 0.0
            if radius > threshold:
                target_height = 0.02 * np.log(radius - threshold) + 0.2

            pos_error[2] = target_height - wrench_center[2]

            scale = np.array([1., 1., 3.])
            a = 0.1  # Relative importance of just *trying* to lift the wrench
            b = 0.9  # Relative importance of placing the wrench on the peg
            lifted = wrench_center[2] > 0.02 or radius < threshold
            in_place = a * float(lifted) + b * reward_utils.tolerance(
                np.linalg.norm(pos_error * scale),
                bounds=(0, 0.02),
                margin=0.4,
                sigmoid='long_tail',
            )

            return in_place, success

        reward_in_place, success = _reward_pos(
            wrench_center,
            _target_pos
        )

        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
        
        # Override reward on success
        if success:
            reward = 10.0
        
        computed_reward = reward

    if env_name == 'disassemble-v2':
        hand = state[:3]
        wrench = state[4:7]
        wrench_center = state[4:7] + np.array([-0.129983, 0, 0])  # env._get_site_pos('RoundNut')
        
        wrench_threshed = wrench.copy()
        WRENCH_HANDLE_LENGTH = 0.02
        threshold = WRENCH_HANDLE_LENGTH / 2.0
        if abs(wrench[0] - hand[0]) < threshold:
            wrench_threshed[0] = hand[0]

        def _reward_quat(state):
            # Ideal laid-down wrench has quat [.707, 0, 0, .707]
            # Rather than deal with an angle between quaternions, just approximate:
            ideal = np.array([0.707, 0, 0, 0.707])
            error = np.linalg.norm(state[7:11] - ideal)
            return max(1.0 - error/0.4, 0.0)

        reward_quat = _reward_quat(state)

        tcp = state[0:3] + np.array([0, 0, -0.045]) #approx
        obj = state[4:7]
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = 0.326954932 #approx
        handle_radius = 0.02
        reward_grab = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='gaussian',
        )
        _target_pos = state[36:39]  # env._target_pos

        def _reward_pos(wrench_center, target_pos):
            pos_error = target_pos + np.array([.0, .0, .1]) - wrench_center

            a = 0.1  # Relative importance of just *trying* to lift the wrench
            b = 0.9  # Relative importance of placing the wrench on the peg
            lifted = wrench_center[2] > 0.02
            in_place = a * float(lifted) + b * reward_utils.tolerance(
                np.linalg.norm(pos_error),
                bounds=(0, 0.02),
                margin=0.2,
                sigmoid='long_tail',
            )

            return in_place

        reward_in_place = _reward_pos(
            wrench_center,
            _target_pos
        )

        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
        # Override reward on success
        success = state[6] > _target_pos[2]
        if success:
            reward = 10.0

        computed_reward = reward

    if env_name == 'coffee-pull-v2':
        obj = state[4:7]
        target = state[36:39]

        obj_init_pos = np.array([0, 0.75, 0.])

        scale = np.array([2., 2., 1.])
        target_to_obj = (obj - target) * scale
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (obj_init_pos - target) * scale
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, 0.05),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )
        tcp_opened = state[3]
        tcp_center = state[0:3] - np.array([[0, 0, 4.49999989e-02]])
        tcp_to_obj = np.linalg.norm(obj - tcp_center)

        tcp = state[0:3] + np.array([0, 0, -0.045]) #approx
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - tcp) #approx
        handle_radius = 0.02
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='gaussian',
        )
        gripper_closed = min(max(0, action[-1]), 1)
        reach = reward_utils.hamacher_product(reach, gripper_closed)
        object_grasped = reach
        
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        if tcp_to_obj < 0.04 and tcp_opened > 0:
            reward += 1. + 5. * in_place
        if target_to_obj < 0.05:
            reward = 10.
        
        computed_reward = reward

    if env_name == 'coffee-push-v2':
        obj = state[4:7]
        target = state[36:39]

        obj_init_pos = np.array([0, 0.6, 0.])

        scale = np.array([2., 2., 1.])
        target_to_obj = (obj - target) * scale
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (obj_init_pos - target) * scale
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, 0.05),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )
        tcp_opened = state[3]
        tcp_center = state[0:3] - np.array([[0, 0, 4.49999989e-02]])
        tcp_to_obj = np.linalg.norm(obj - tcp_center)

        tcp = state[0:3] + np.array([0, 0, -0.045]) #approx
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - tcp) #approx
        handle_radius = 0.02
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='gaussian',
        )
        gripper_closed = min(max(0, action[-1]), 1)
        reach = reward_utils.hamacher_product(reach, gripper_closed)
        object_grasped = reach
        
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        if tcp_to_obj < 0.04 and tcp_opened > 0:
            reward += 1. + 5. * in_place
        if target_to_obj < 0.05:
            reward = 10.
        
        computed_reward = reward

    if env_name == 'door-close-v2':
        _TARGET_RADIUS = 0.05
        tcp = state[0:3] + np.array([0, 0, -0.045]) #approx
        obj = state[4:7]
        target = state[36:39]

        tcp_to_target = np.linalg.norm(tcp - target)
        tcp_to_obj = np.linalg.norm(tcp - obj)
        obj_to_target = np.linalg.norm(obj - target)

        obj_init_pos = np.array([0.1, 0.95, 0.15])
        in_place_margin = np.linalg.norm(obj_init_pos - target)
        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='gaussian',)

        hand_init_pos = np.array([-0.5, 0.6, 0.2])
        hand_margin = np.linalg.norm(hand_init_pos - obj) + 0.1
        hand_in_place = reward_utils.tolerance(tcp_to_target,
                                    bounds=(0, 0.25*_TARGET_RADIUS),
                                    margin=hand_margin,
                                    sigmoid='gaussian',)

        reward = 3 * hand_in_place + 6 * in_place

        if obj_to_target < _TARGET_RADIUS:
            reward = 10
            
        computed_reward = reward

    if env_name == 'door-open-v2':
        def _reward_grab_effort(actions):
            return (np.clip(actions[3], -1, 1) + 1.0) / 2.0

        def _reward_pos(obs, theta):
            hand = obs[:3]
            door = obs[4:7] + np.array([-0.05, 0, 0])

            threshold = 0.12
            radius = np.linalg.norm(hand[:2] - door[:2])
            if radius <= threshold:
                floor = 0.0
            else:
                floor = 0.04 * np.log(radius - threshold) + 0.4
            above_floor = 1.0 if hand[2] >= floor else reward_utils.tolerance(
                floor - hand[2],
                bounds=(0.0, 0.01),
                margin=floor / 2.0,
                sigmoid='long_tail',
            )
            in_place = reward_utils.tolerance(
                np.linalg.norm(hand - door - np.array([0.05, 0.03, -0.01])),
                bounds=(0, threshold / 2.0),
                margin=0.5,
                sigmoid='long_tail',
            )
            ready_to_open = reward_utils.hamacher_product(above_floor, in_place)

            door_angle = -theta
            a = 0.2  # Relative importance of just *trying* to open the door at all
            b = 0.8  # Relative importance of fully opening the door
            opened = a * float(theta < -np.pi/90.) + b * reward_utils.tolerance(
                np.pi/2. + np.pi/6 - door_angle,
                bounds=(0, 0.5),
                margin=np.pi/3.,
                sigmoid='long_tail',
            )

            return ready_to_open, opened
        
        obj_to_target = np.linalg.norm(state[4:7] - state[36:39])
        x = 0.55
        theta = (obj_to_target - x) / x * (np.pi / 2) #approx

        reward_grab = _reward_grab_effort(action)
        reward_steps = _reward_pos(state, theta)

        reward = sum((
            2.0 * reward_utils.hamacher_product(reward_steps[0], reward_grab),
            8.0 * reward_steps[1],
        ))

        _target_pos = state[36:39]

        if abs(state[4] - _target_pos[0]) <= 0.08:
            reward = 10.0
        
        computed_reward = reward

    if env_name == 'door-lock-v2':
        obj = state[4:7]
        tcp = state[0:3] + np.array([[0.00057068, 0.01744805, 0.00082062]]) #approx
        init_left_pad = np.array([-0.02224193, 0.69970212, 0.19125289]) #approx
        
        scale = np.array([0.25, 1., 0.5])
        tcp_to_obj = np.linalg.norm((obj - tcp) * scale)
        tcp_to_obj_init = np.linalg.norm((obj - init_left_pad) * scale)

        _target_pos = state[36:39]
        obj_to_target = abs(_target_pos[2] - obj[2])

        tcp_opened = max(state[3], 0.0)
        near_lock = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.01),
            margin=tcp_to_obj_init,
            sigmoid='long_tail',
        )
        _lock_length = 0.1
        lock_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=_lock_length,
            sigmoid='long_tail',
        )

        reward = 2 * reward_utils.hamacher_product(tcp_opened, near_lock)
        reward += 8 * lock_pressed

        computed_reward = reward

    if env_name == 'door-unlock-v2':
        gripper = state[:3]
        lock = state[4:7]

        offset = np.array([.0, .055, .07])

        scale = np.array([0.25, 1., 0.5])
        obj_init_pos = np.array([0, 0.85, 0.15])
        init_tcp =  np.array([0.00600364, 0.6004762, 0.14935033])
        shoulder_to_lock = (gripper + offset - lock) * scale
        shoulder_to_lock_init = (
            init_tcp + offset - obj_init_pos
        ) * scale

        ready_to_push = reward_utils.tolerance(
            np.linalg.norm(shoulder_to_lock),
            bounds=(0, 0.02),
            margin=np.linalg.norm(shoulder_to_lock_init),
            sigmoid='long_tail',
        )

        _target_pos = state[36:39]
        obj_to_target = abs(_target_pos[2] - lock[2])
        _lock_length = 0.1
        pushed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=_lock_length,
            sigmoid='long_tail',
        )

        reward = 2 * ready_to_push + 8 * pushed

        computed_reward = reward

    if env_name == 'faucet-open-v2':
        obj = state[4:7]
        tcp = state[0:3] + np.array([0, 0, -0.045]) #approx
        target = state[36:39]
        _target_radius = 0.05
        obj_init_pos = np.array([0, 0.8, 0.0])
        init_tcp = np.array([0.00588028, 0.39976012, 0.14998482])
        
        target_to_obj = (obj - target)[1]
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (obj_init_pos - target)
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, _target_radius),
            margin=abs(target_to_obj_init - _target_radius),
            sigmoid='long_tail',
        )

        faucet_reach_radius = 0.01
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, faucet_reach_radius),
            margin=abs(tcp_to_obj_init - faucet_reach_radius),
            sigmoid='gaussian',
        )

        tcp_opened = 0
        object_grasped = reach

        reward = 2 * reach + 3 * in_place

        reward *= 2

        reward = 10 if target_to_obj <= _target_radius else reward

        computed_reward = reward

    if env_name == 'faucet-close-v2':
        obj = state[4:7]
        tcp = state[0:3] + np.array([0, 0, -0.045]) #approx
        target = state[36:39]
        _target_radius = 0.05
        obj_init_pos = np.array([0, 0.8, 0.0])
        init_tcp = np.array([0.00588028, 0.39976012, 0.14998482])

        target_to_obj = (obj - target)[1]
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (obj_init_pos - target)
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, _target_radius),
            margin=abs(target_to_obj_init - _target_radius),
            sigmoid='long_tail',
        )

        faucet_reach_radius = 0.01
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, faucet_reach_radius),
            margin=abs(tcp_to_obj_init - faucet_reach_radius),
            sigmoid='gaussian',
        )

        tcp_opened = 0
        object_grasped = reach

        reward = 2 * reach + 3 * in_place
        reward *= 2
        reward = 10 if target_to_obj <= _target_radius else reward

        computed_reward = reward

    if env_name == 'handle-press-v2':
        objPos = state[4:7]
        obj = state[4:7]
        tcp = state[0:3] + np.array([0, 0, -0.045]) #approx
        init_tcp = np.array([0.00527384, 0.60098165, 0.14945339]) #approx
        target = state[36:39]
        _handle_init_pos = np.array([obj[0], obj[1], 0.172]) #approx
        
        target_to_obj = (obj[2] - target[2])
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (_handle_init_pos[2] - target[2])
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        TARGET_RADIUS = 0.02
        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, TARGET_RADIUS),
            margin=abs(target_to_obj_init - TARGET_RADIUS),
            sigmoid='long_tail',
        )

        handle_radius = 0.02
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(_handle_init_pos - init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        tcp_opened = 0
        object_grasped = reach

        reward = reward_utils.hamacher_product(reach, in_place)
        reward = 1 if target_to_obj <= TARGET_RADIUS else reward
        reward *= 10

        computed_reward = reward

    if env_name == 'handle-pull-v2':
        obj = state[4:7]
        TARGET_RADIUS = 0.02
        target = state[36:39]
        obj_init_pos = np.array([obj[0] - 0.05, obj[1] + 0.216, 0])
        tcp = state[0:3] + np.array([[0, 0.005, 4.49999989e-02]]) #approx

        target_to_obj = abs(target[2] - obj[2])
        target_to_obj_init = abs(target[2] - obj_init_pos[2])

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )

        handle_radius = 0.02
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - tcp) #approx
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        object_grasped = reach
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        tcp_opened = state[3]
        tcp_to_obj = np.linalg.norm(obj - tcp)

        if tcp_to_obj < 0.035 and tcp_opened > 0 and \
                obj[2] - 0.01 > obj_init_pos[2]:
            reward += 1. + 5. * in_place
        if target_to_obj < TARGET_RADIUS:
            reward = 10.

        computed_reward = reward

    if env_name == 'handle-press-side-v2':
        objPos = state[4:7]
        obj = state[4:7]
        tcp = state[0:3] + np.array([0, 0, -0.045]) #approx
        init_tcp = np.array([0.00600364, 0.60047618, 0.14935032]) #approx
        target = state[36:39]
        _handle_init_pos = np.array([obj[0], obj[1], 0.172]) #approx
        
        target_to_obj = (obj[2] - target[2])
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (_handle_init_pos[2] - target[2])
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        TARGET_RADIUS = 0.02

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, TARGET_RADIUS),
            margin=abs(target_to_obj_init - TARGET_RADIUS),
            sigmoid='long_tail',
        )

        handle_radius = 0.02
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(_handle_init_pos - init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        tcp_opened = 0
        object_grasped = reach

        reward = reward_utils.hamacher_product(reach, in_place)
        reward = 1 if target_to_obj <= TARGET_RADIUS else reward
        reward *= 10

        computed_reward = reward

    if env_name == 'handle-pull-side-v2':
        obj = state[4:7]
        TARGET_RADIUS = 0.02
        target = state[36:39]
        obj_init_pos = np.array([obj[0], obj[1], 0.05])
        tcp = state[0:3] + np.array([[0, 0.005, -4.49999989e-02]]) #approx
        
        # Emphasize Z error
        scale = np.array([0., 0., 1.])
        target_to_obj = (obj - target) * scale
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (obj_init_pos - target) * scale
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )

        handle_radius = 0.02
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - tcp) #approx
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        object_grasped = reach
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        tcp_opened = state[3]
        tcp_to_obj = np.linalg.norm(obj - tcp)

        if tcp_to_obj < 0.035 and tcp_opened > 0 and obj[2] - 0.01 > obj_init_pos[2]:
            reward += 1. + 5. * in_place
        if target_to_obj < TARGET_RADIUS:
            reward = 10.

        computed_reward = reward

    if env_name == 'peg-insert-side-v2':
        tcp = state[0:3] + np.array([[0, 0.005, -4.49999989e-02]]) #approx
        obj = state[4:7]
        obj_head = state[4:7] - np.array([0.13, 0., 0.01])
        tcp_opened = state[3]
        target = state[36:39]
        tcp_to_obj = np.linalg.norm(obj - tcp)
        scale = np.array([1., 2., 2.])
        obj_to_target = np.linalg.norm((obj_head - target) * scale)

        peg_head_pos_init = np.array([-0.1, 0.6, 0.01478467])
        TARGET_RADIUS = 0.07
        in_place_margin = np.linalg.norm((peg_head_pos_init - target) * scale)
        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)
        
        pad_success_margin = 0.03
        object_reach_radius=0.01
        x_z_margin = 0.005
        obj_radius = 0.0075

        obj_init_pos = np.array([0, 0.6, 0.02])
        
        handle_radius = 0.02
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - tcp) #approx
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        object_grasped = reach
        # gripper_closed = min(max(0, action[-1]), 1)
        # object_grasped = reward_utils.hamacher_product(reach, gripper_closed)
        if tcp_to_obj < 0.08 and (tcp_opened > 0) and (obj[2] - 0.01 > obj_init_pos[2]):
            object_grasped = 1.
        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped, in_place)
        reward = in_place_and_object_grasped

        if tcp_to_obj < 0.08 and (tcp_opened > 0) and (obj[2] - 0.01 > obj_init_pos[2]):
            reward += 1. + 5 * in_place

        if obj_to_target <= 0.07:
            reward = 10.

        computed_reward = reward

    if env_name == 'peg-unplug-side-v2':
        tcp = state[0:3] + np.array([[0, 0.005, -4.49999989e-02]]) #approx
        obj = state[4:7]
        tcp_opened = state[3]
        target = state[36:39]
        tcp_to_obj = np.linalg.norm(obj - tcp)
        obj_to_target = np.linalg.norm(obj - target)
        pad_success_margin = 0.05
        object_reach_radius = 0.01
        x_z_margin = 0.005
        obj_radius = 0.025

        obj_init_pos = np.array([-0.1, 0.7, 0.13]) #approx
        
        handle_radius = 0.02
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - tcp) #approx
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        object_grasped = reach
        in_place_margin = np.linalg.norm(obj_init_pos - target)
        
        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.05),
            margin=in_place_margin,
            sigmoid='long_tail',
        )
        grasp_success = (tcp_opened > 0.5 and (obj[0] - obj_init_pos[0] > 0.015))
        
        reward = 2 * object_grasped

        if grasp_success and tcp_to_obj < 0.035:
            reward = 1 + 2 * object_grasped + 5 * in_place

        if obj_to_target <= 0.05:
            reward = 10.

        computed_reward = reward

    if env_name == 'plate-slide-v2':
        _TARGET_RADIUS = 0.05
        tcp = state[0:3] + np.array([0, 0.005, -4.49999989e-02]) #approx
        obj = state[4:7]
        tcp_opened = state[3]
        target = state[36:39]
        
        obj_init_pos = np.array([0., 0.6, 0.])
        obj_to_target = np.linalg.norm(obj - target)
        in_place_margin = np.linalg.norm(obj_init_pos - target)

        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)
        
        init_tcp = np.array([0.00600364, 0.6004762, 0.14935033]) #approx
        tcp_to_obj = np.linalg.norm(tcp - obj)
        obj_grasped_margin = np.linalg.norm(init_tcp - obj_init_pos)

        object_grasped = reward_utils.tolerance(tcp_to_obj,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=obj_grasped_margin,
                                    sigmoid='long_tail',)
        
        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped, in_place)
        reward = 8 * in_place_and_object_grasped

        if obj_to_target < _TARGET_RADIUS:
            reward = 10.

        computed_reward = reward
    
    if env_name == 'plate-slide-back-v2':
        _TARGET_RADIUS = 0.05
        tcp = state[0:3] + np.array([0, 0.005, -4.49999989e-02]) #approx
        obj = state[4:7]
        tcp_opened = state[3]
        target = state[36:39]

        obj_init_pos = np.array([0., 0.85, 0.])
        obj_to_target = np.linalg.norm(obj - target)
        in_place_margin = np.linalg.norm(obj_init_pos - target)
        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin - _TARGET_RADIUS,
                                    sigmoid='long_tail',)

        init_tcp = np.array([0.00600364, 0.6004762, 0.14935033]) #approx
        tcp_to_obj = np.linalg.norm(tcp - obj)
        obj_grasped_margin = np.linalg.norm(init_tcp - obj_init_pos)
        object_grasped = reward_utils.tolerance(tcp_to_obj,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=obj_grasped_margin - _TARGET_RADIUS,
                                    sigmoid='long_tail',)

        reward = 1.5 * object_grasped

        if tcp[2] <= 0.03 and tcp_to_obj < 0.07:
            reward = 2 + (7 * in_place)

        if obj_to_target < _TARGET_RADIUS:
            reward = 10.

        computed_reward = reward

    if env_name == 'plate-slide-side-v2':
        _TARGET_RADIUS = 0.05
        tcp = state[0:3] + np.array([0, 0.005, -4.49999989e-02]) #approx
        obj = state[4:7]
        tcp_opened = state[3]
        target = state[36:39]

        obj_init_pos = np.array([0., 0.6, 0.])
        obj_to_target = np.linalg.norm(obj - target)
        in_place_margin = np.linalg.norm(obj_init_pos - target)
        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin - _TARGET_RADIUS,
                                    sigmoid='long_tail',)

        init_tcp = np.array([0.00600364, 0.6004762, 0.14935033]) #approx
        tcp_to_obj = np.linalg.norm(tcp - obj)
        obj_grasped_margin = np.linalg.norm(init_tcp - obj_init_pos)
        object_grasped = reward_utils.tolerance(tcp_to_obj,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=obj_grasped_margin - _TARGET_RADIUS,
                                    sigmoid='long_tail',)

        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped, in_place)
        reward = 1.5 * object_grasped

        if tcp[2] <= 0.03 and tcp_to_obj < 0.07:
            reward = 2 + (7 * in_place)

        if obj_to_target < _TARGET_RADIUS:
            reward = 10.

        computed_reward = reward

    if env_name == 'plate-slide-back-side-v2':
        _TARGET_RADIUS = 0.05
        tcp = state[0:3] + np.array([0, 0.005, -4.49999989e-02]) #approx
        obj = state[4:7]
        tcp_opened = state[3]
        target = state[36:39]

        obj_init_pos = np.array([-0.25, 0.6, 0.])
        obj_to_target = np.linalg.norm(obj - target)
        in_place_margin = np.linalg.norm(obj_init_pos - target)
        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin - _TARGET_RADIUS,
                                    sigmoid='long_tail',)

        init_tcp = np.array([0.00600364, 0.6004762, 0.14935033]) #approx
        tcp_to_obj = np.linalg.norm(tcp - obj)
        obj_grasped_margin = np.linalg.norm(init_tcp - obj_init_pos)
        object_grasped = reward_utils.tolerance(tcp_to_obj,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=obj_grasped_margin - _TARGET_RADIUS,
                                    sigmoid='long_tail',)

        reward = 1.5 * object_grasped

        if tcp[2] <= 0.03 and tcp_to_obj < 0.07:
            reward = 2 + (7 * in_place)

        if obj_to_target < _TARGET_RADIUS:
            reward = 10.
        
        computed_reward = reward

    if env_name == 'push-v2':
        TARGET_RADIUS = 0.05
        
        obj = state[4:7]
        tcp_opened = state[3]
        tcp = state[0:3] + np.array([0, 0.005, -4.49999989e-02]) #approx
        tcp_to_obj = np.linalg.norm(obj - tcp)
        target = state[36:39]
        target_to_obj = np.linalg.norm(obj - target)
        obj_init_pos = np.array([0, 0.6, 0.02])
        target_to_obj_init = np.linalg.norm(obj_init_pos - target)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )

        tcp_to_obj_init = np.linalg.norm(obj_init_pos - tcp) #approx
        handle_radius = 0.02
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        gripper_closed = min(max(0, action[-1]), 1)
        reach = reward_utils.hamacher_product(reach, gripper_closed)
        object_grasped = reach
        reward = 2 * object_grasped

        if tcp_to_obj < 0.02 and tcp_opened > 0:
            reward += 1. + reward + 5. * in_place
        if target_to_obj < TARGET_RADIUS:
            reward = 10.

        computed_reward = reward
    
    if env_name == 'push-back-v2':
        TARGET_RADIUS = 0.05

        obj = state[4:7]
        tcp_opened = state[3]
        tcp = state[0:3] + np.array([0, 0.005, -4.49999989e-02]) #approx
        tcp_to_obj = np.linalg.norm(obj - tcp)
        target = state[36:39]
        target_to_obj = np.linalg.norm(obj - target)
        obj_init_pos = np.array([0, 0.8, 0.02])
        target_to_obj_init = np.linalg.norm(obj_init_pos - target)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - tcp) #approx
        handle_radius = 0.02
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        gripper_closed = min(max(0, action[-1]), 1)
        reach = reward_utils.hamacher_product(reach, gripper_closed)
        object_grasped = reach

        reward = reward_utils.hamacher_product(object_grasped, in_place)

        if (tcp_to_obj < 0.01) and (0 < tcp_opened < 0.55) and (target_to_obj_init - target_to_obj > 0.01):
            reward += 1. + 5. * in_place
        if target_to_obj < TARGET_RADIUS:
            reward = 10.
        
        computed_reward = reward

    if env_name == 'push-v2':
        obj = state[4:7]
        tcp_opened = state[3]
        tcp = state[0:3] + np.array([0, 0.005, -4.49999989e-02]) #approx
        tcp_to_obj = np.linalg.norm(obj - tcp)
        target = state[36:39]
        target_to_obj = np.linalg.norm(obj - target)
        obj_init_pos = np.array([0, 0.6, 0.02])
        target_to_obj_init = np.linalg.norm(obj_init_pos - target)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )
        tcp_to_obj_init = np.linalg.norm(obj_init_pos - tcp) #approx
        handle_radius = 0.02
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        # gripper_closed = min(max(0, action[-1]), 1)
        # reach = reward_utils.hamacher_product(reach, gripper_closed)
        object_grasped = reach

        reward = 2 * object_grasped

        if tcp_to_obj < 0.02 and tcp_opened > 0:
            reward += 1. + reward + 5. * in_place
        if target_to_obj < TARGET_RADIUS:
            reward = 10.

    if env_name == 'stick-push-v2':
        _TARGET_RADIUS = 0.12
        obj = state[4:7]
        tcp = state[0:3] + np.array([0, 0.005, -4.49999989e-02]) #approx
        stick = state[4:7] + np.array([.015, .0, .0])
        container = state[11:14]
        tcp_opened = state[3]
        target = state[36:39]

        tcp_to_stick = np.linalg.norm(stick - tcp)
        stick_to_target = np.linalg.norm(stick - target)
        stick_init_pos = np.array([-0.05, 0.6, 0.02])
        stick_in_place_margin = (np.linalg.norm(stick_init_pos - target)) - _TARGET_RADIUS
        stick_in_place = reward_utils.tolerance(stick_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=stick_in_place_margin,
                                    sigmoid='long_tail',)

        obj_init_pos = np.array([0.2, 0.6, 0.0])
        container_to_target = np.linalg.norm(container - target)
        container_in_place_margin = np.linalg.norm(obj_init_pos - target) - _TARGET_RADIUS
        container_in_place = reward_utils.tolerance(container_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=container_in_place_margin,
                                    sigmoid='long_tail',)

        tcp_to_obj_init = np.linalg.norm(obj_init_pos - tcp) #approx
        tcp_to_obj = np.linalg.norm(obj - tcp)
        handle_radius = 0.02
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        gripper_closed = min(max(0, action[-1]), 1)
        reach = reward_utils.hamacher_product(reach, gripper_closed)
        object_grasped = reach

        reward = object_grasped

        if tcp_to_stick < 0.02 and (tcp_opened > 0) and (stick[2] - 0.01 > stick_init_pos[2]):
            object_grasped = 1
            reward = 2. + 5. * stick_in_place + 3. * container_in_place

            if container_to_target <= _TARGET_RADIUS:
                reward = 10.

        computed_reward = reward

    if env_name == 'stick-pull-v2':
        _TARGET_RADIUS = 0.05
        obj = state[4:7]
        tcp = state[0:3] + np.array([0, 0.005, -4.49999989e-02]) #approx
        stick = state[4:7]
        end_of_stick = stick + np.array([0.05, 0., 0.])
        container = state[11:14] + np.array([0.05, 0., 0.])
        obj_init_pos = np.array([0.2, 0.6, 0.0])
        container_init_pos = obj_init_pos + np.array([0.05, 0., 0.])
        handle = state[11:14]
        tcp_opened = state[3]
        target = state[36:39]
        tcp_to_stick = np.linalg.norm(stick - tcp)
        handle_to_target = np.linalg.norm(handle - target)

        yz_scaling = np.array([1., 1., 2.])
        stick_to_container = np.linalg.norm((stick - container) * yz_scaling)
        stick_init_pos = np.array([0, 0.63, 0.02])
        stick_in_place_margin = (np.linalg.norm((stick_init_pos - container_init_pos) * yz_scaling))
        stick_in_place = reward_utils.tolerance(
            stick_to_container,
            bounds=(0, _TARGET_RADIUS),
            margin=stick_in_place_margin,
            sigmoid='long_tail',
        )

        stick_to_target = np.linalg.norm(stick - target)
        stick_in_place_margin_2 = np.linalg.norm(stick_init_pos - target)
        stick_in_place_2 = reward_utils.tolerance(
            stick_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=stick_in_place_margin_2,
            sigmoid='long_tail',
        )

        container_to_target = np.linalg.norm(container - target)
        container_in_place_margin = np.linalg.norm(obj_init_pos - target)
        container_in_place = reward_utils.tolerance(
            container_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=container_in_place_margin,
            sigmoid='long_tail',
        )

        tcp_to_obj_init = np.linalg.norm(obj_init_pos - tcp) #approx
        tcp_to_obj = np.linalg.norm(obj - tcp)
        handle_radius = 0.02
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        gripper_closed = min(max(0, action[-1]), 1)
        reach = reward_utils.hamacher_product(reach, gripper_closed)
        object_grasped = reach

        grasp_success = (tcp_to_stick < 0.02 and (tcp_opened > 0) and (stick[2] - 0.01 > stick_init_pos[2]))
        object_grasped = 1 if grasp_success else object_grasped

        in_place_and_object_grasped = reward_utils.hamacher_product(
            object_grasped, stick_in_place)
        reward = in_place_and_object_grasped

        if grasp_success:
            reward = 1. + in_place_and_object_grasped + 5. * stick_in_place

            def _stick_is_inserted(handle, end_of_stick):
                return (end_of_stick[0] >= handle[0]) and (np.abs(end_of_stick[1] - handle[1]) <= 0.040) and (np.abs(end_of_stick[2] - handle[2]) <= 0.060)

            if _stick_is_inserted(handle, end_of_stick):
                reward = 1. + in_place_and_object_grasped + 5. + \
                         2. * stick_in_place_2 + 1. * container_in_place

                if handle_to_target <= 0.12:
                    reward = 10.
        
        computed_reward = reward
    
    if env_name == 'reach-v2':
        _TARGET_RADIUS = 0.05
        tcp = state[0:3] + np.array([0, 0.005, -4.49999989e-02]) #approx
        obj = state[4:7]
        tcp_opened = state[3]
        target = state[36:39]

        tcp_to_target = np.linalg.norm(tcp - target)
        obj_to_target = np.linalg.norm(obj - target)

        hand_init_pos = np.array([0., 0.6, 0.2])
        in_place_margin = (np.linalg.norm(hand_init_pos - target))
        in_place = reward_utils.tolerance(tcp_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)
        reward = 10 * in_place
        computed_reward = reward

    if env_name == 'pick-place-v2':
        _TARGET_RADIUS = 0.05
        tcp = state[0:3] + np.array([0, 0.005, -4.49999989e-02]) #approx
        obj = state[4:7]
        tcp_opened = state[3]
        target = state[36:39]

        obj_to_target = np.linalg.norm(obj - target)
        tcp_to_obj = np.linalg.norm(obj - tcp)
        obj_init_pos = np.array([0, 0.6, 0.02])
        in_place_margin = (np.linalg.norm(obj_init_pos - target))

        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)

        handle_radius = 0.02
        tcp_to_obj_init = 0.326954932 #approx
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        tcp_opened = 0
        gripper_closed = min(max(0, action[-1]), 1)
        object_grasped = reward_utils.hamacher_product(reach, gripper_closed)

        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
                                                                    in_place)
        reward = in_place_and_object_grasped

        if tcp_to_obj < 0.02 and (tcp_opened > 0) and (obj[2] - 0.01 > obj_init_pos[2]):
            reward += 1. + 5. * in_place
        if obj_to_target < _TARGET_RADIUS:
            reward = 10.
        
        computed_reward = reward

    return computed_reward

def reverse_transition(transition, reversible_dict, env_names):
    # print('transition', transition)
    # input: [[env_index, state], action, reward, [env_index, next_state], absorbing, last]
    # output: [[reverse_env_index, reverse_state], action, reverse_reward, [reverse_env_index, reverse_next_state], absorbing, last]
    env_index = transition[0][0]
    state = transition[0][1]
    action = transition[1]
    reward = transition[2]
    next_state = transition[3][1]
    absorbing = transition[4]
    last = transition[5]

    reverse_env_index = reversible_dict[env_index]
    reverse_env_name = env_names[reverse_env_index]
    # state: curr_s, prev_s, goal; next_state: next_s, curr_s, goal
    prev_s, curr_s, next_s, goal = state[18:36], state[0:18], next_state[0:18], state[36:39]
    reverse_prev_s, reverse_curr_s, reverse_next_s = next_s, curr_s, prev_s
    # new_goal
    if reverse_env_name == 'window-open-v2':
        new_goal = np.array([0.21, goal[1], 0.16])
    if reverse_env_name == 'window-close-v2':
        new_goal = np.array([0, goal[1], 0.2])
    if reverse_env_name == 'drawer-open-v2':
        new_goal = np.array([goal[0], 0.53999998, 0.09])
    if reverse_env_name == 'drawer-close-v2':
        new_goal = np.array([goal[0], 0.73999998, 0.09])
    if reverse_env_name == 'assembly-v2':
        new_goal = np.array([0, 0.8, 0.1])
    if reverse_env_name == 'disassemble-v2':
        goal_low = np.array([0, 0.6, 0.1749])
        goal_high = np.array([0.1, 0.75, 0.1751])
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'coffee-pull-v2':
        goal_low = (-0.1, 0.55, -.001)
        goal_high = (0.1, 0.65, +.001)
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'coffee-push-v2':
        goal_low = (-0.05, 0.7, -.001)
        goal_high = (0.05, 0.75, +.001)
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'door-close-v2':
        goal_low = (.2, 0.65, 0.1499)
        goal_high = (.3, 0.75, 0.1501)
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'door-open-v2':
        goal_low = (-.3, 0.4, 0.1499)
        goal_high = (-.2, 0.5, 0.1501)
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'door-lock-v2':
        new_goal = np.array([0, 0.67, 0.111])
    if reverse_env_name == 'door-unlock-v2':
        new_goal = np.array([0.1, 0.67, 0.211])
    if reverse_env_name == 'faucet-open-v2':
        new_goal = np.array([0.15, 0.825, 0.125])
    if reverse_env_name == 'faucet-close-v2':
        new_goal = np.array([-0.2, 0.825, 0.125])
    if reverse_env_name == 'handle-press-v2':
        new_goal = np.array([0, 0.625, 0.075])
    if reverse_env_name == 'handle-pull-v2':
        new_goal = np.array([0, 0.625, 0.1725])
    if reverse_env_name == 'handle-press-side-v2':
        new_goal = np.array([-0.05, 0.7, 0.075])
    if reverse_env_name == 'handle-pull-side-v2':
        new_goal = np.array([-0.05, 0.7, 0.1725])
    if reverse_env_name == 'peg-insert-side-v2':
        new_goal = np.array([-0.3, 0.55, 0.13])
    if reverse_env_name == 'peg-unplug-side-v2':
        new_goal = np.array([-0.2, 0.7, 0]) + np.array([.044, .0, .131]) + np.array([.15, .0, .0])
    if reverse_env_name == 'plate-slide-v2':
        goal_low = np.array([-0.1, 0.85, 0.])
        goal_high = np.array([0.1, 0.9, 0.])
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'plate-slide-back-v2':
        goal_low = np.array([-0.1, 0.6, 0.015])
        goal_high = np.array([0.1, 0.6, 0.015])
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'plate-slide-side-v2':
        goal_low = np.array([-0.3, 0.54, 0.])
        goal_high = np.array([-0.25, 0.66, 0.])
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'plate-slide-back-side-v2':
        goal_low = np.array([-0.05, 0.6, 0.015])
        goal_high = np.array([0.15, 0.6, 0.015])
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'push-v2':
        goal_low = np.array([-0.1, 0.8, 0.01])
        goal_high = np.array([0.1, 0.9, 0.02])
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'push-back-v2':
        goal_low = np.array([-0.1, 0.6, 0.0199])
        goal_high = np.array([0.1, 0.7, 0.0201])
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'stick-push-v2':
        new_goal = np.array([0.4, 0.575, 0.132])
    if reverse_env_name == 'stick-pull-v2':
        goal_low = np.array([0.35, 0.45, 0.0199])
        goal_high = np.array([0.45, 0.55, 0.0201])
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'reach-v2':
        goal_low = np.array([-0.1, 0.8, 0.05])
        goal_high = np.array([0.1, 0.9, 0.3])
        new_goal = np.random.uniform(goal_low, goal_high)
    if reverse_env_name == 'pick-place-v2':
        goal_low = np.array([-0.1, 0.8, 0.05])
        goal_high = np.array([0.1, 0.9, 0.3])
        new_goal = np.random.uniform(goal_low, goal_high)
    # reverse_state, reverse_next_state 
    reverse_state = np.concatenate([reverse_curr_s, reverse_prev_s, new_goal])
    reverse_next_state = np.concatenate([reverse_next_s, reverse_curr_s, new_goal])
    reverse_reward = state_reward(reverse_env_name, reverse_next_state, action)

    reverse_transition = [[reverse_env_index, reverse_state], action, reverse_reward, [reverse_env_index, reverse_next_state], absorbing, last]

    # print(reverse_transition)
    # print('-'*30)
    return reverse_transition

def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
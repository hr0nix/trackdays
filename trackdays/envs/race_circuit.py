import math

from gym.envs.registration import register

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import CircularLane, StraightLane, SineLane, LineType
from highway_env.vehicle.kinematics import Vehicle


class Circuit(object):
    def __init__(self, circuit_config, np_random):
        self._circuit_config = circuit_config
        self._road_network = RoadNetwork()
        self._road = Road(network=self._road_network, np_random=np_random, record_history=True)
        self._route = list()
        self._circuit_length = 0
        self._create_circuit()

    @property
    def speed_limit(self):
        return self._circuit_config['speed_limit']

    @property
    def circuit_width(self):
        return self._circuit_config['circuit_width']

    @property
    def circuit_length(self):
        return self._circuit_length

    @property
    def road(self):
        return self._road

    @property
    def start_lane_index(self):
        return self._route and self._route[0]

    def get_route(self):
        return list(self._route)

    def _create_circuit(self):
        circular_radius = 50
        circular_lane_1 = CircularLane(
            center=(0.0, 0.0), radius=circular_radius, start_phase=0, end_phase=math.pi * 0.5,
            speed_limit=self.speed_limit, width=self.circuit_width,
        )
        circular_lane_2 = CircularLane(
            center=(0.0, 0.0), radius=circular_radius, start_phase=math.pi * 0.5, end_phase=math.pi,
            speed_limit=self.speed_limit, width=self.circuit_width,
        )
        circular_lane_start = circular_lane_1.position(0, 0)
        circular_lane_end = circular_lane_2.position(circular_lane_2.length, 0)
        sine_lane = SineLane(
            circular_lane_end, circular_lane_start,
            amplitude=10,
            pulsation=2 * math.pi / (circular_radius * 2),
            phase=0,
            speed_limit=self.speed_limit, width=self.circuit_width,
        )

        self._add_lane('start', 'int1', circular_lane_1)
        self._add_lane('int1', 'int2', circular_lane_2)
        self._add_lane('int2', 'end', sine_lane)

    def _add_lane(self, src, dest, lane):
        self._road_network.add_lane(src, dest, lane)
        self._circuit_length += lane.length
        self._route.append((src, dest, 0))

    def get_circuit_pos(self, pos):
        current_lane_index = self._road_network.get_closest_lane_index(pos)
        assert current_lane_index is not None

        abs_pos = 0.0
        for lane_index in self._route:
            lane = self._road_network.get_lane(lane_index)
            if lane_index != current_lane_index:
                abs_pos += lane.length
            else:
                abs_pos += lane.local_coordinates(pos)[0]
                break

        # The position can go out of bounds if we aren't exactly within a lane, so we need to normalize
        while abs_pos < 0:
            abs_pos += self.circuit_length
        while abs_pos > self.circuit_length:
            abs_pos -= self.circuit_length

        return abs_pos

    def get_lap_progress(self, prev_pos, cur_pos):
        prev_circuit_pos = self.get_circuit_pos(prev_pos)
        cur_circuit_pos = self.get_circuit_pos(cur_pos)

        # Somewhat heuristic: large jump is pos means that finish line has been crossed
        finish_line_crossed = abs(cur_circuit_pos - prev_circuit_pos) > 50
        if finish_line_crossed:
            if cur_circuit_pos < prev_circuit_pos:
                # Going the right way
                return cur_circuit_pos + self.circuit_length - prev_circuit_pos, True
            else:
                # Going backwards
                return cur_circuit_pos - self.circuit_length - prev_circuit_pos, True
        else:
            return cur_circuit_pos - prev_circuit_pos, False


class RaceCircuitEnv(AbstractEnv):
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'circuit_config': {
                'speed_limit': 1000.0,
                'circuit_width': 10,
            },
            'race_config': {
                'max_lap_count': 2,
                'max_steps': 100,
                'out_of_circuit_stop_margin': 10,
            },
            'reward_config': {
                'out_of_circuit_reward_scale': -2,
                'tick': -1,
                'lap_progress': 1,
            },
            'screen_width': 200,
            'screen_height': 200,
            'scaling': 3,
            'centering_position': [0.5, 0.5],
            'simulation_frequency': 6,
            'policy_frequency': 3,
            'observation': {
                'type': 'GrayscaleObservation',
                'weights': [0.33, 0.33, 0.33],
                'stack_size': 4,
                'observation_shape': (200, 200),
            },
            'action': {
                'type': 'Continuous'
            }
        })
        return config

    def race_config(self, key):
        return self.config['race_config'][key]

    def reward_config(self, key):
        return self.config['reward_config'][key]

    def reset(self):
        self._make_road()
        self._make_vehicles()
        self._lap_number = 0
        self._steps = 0
        return super().reset()

    def step(self, action):
        self._vehicle_pos_before_update = self.vehicle.position.copy()
        result = super().step(action)
        self._update_progress()
        return result

    def _update_progress(self):
        last_progress, finish_line_crossed = self._circuit.get_lap_progress(
            self._vehicle_pos_before_update, self.vehicle.position
        )
        if finish_line_crossed:
            self._lap_number += math.copysign(1, last_progress)
        self._steps += 1

    def _make_road(self):
        self._circuit = Circuit(self.config['circuit_config'], self.np_random)
        self.road = self._circuit.road

    def _make_vehicles(self):
        route = self._circuit.get_route()
        start_lane_index = route[self.np_random.choice(len(route))]
        start_lane = self.road.network.get_lane(start_lane_index)
        self.vehicle = Vehicle.make_on_lane(
            self.road,
            lane_index=start_lane_index,
            longitudinal=self.np_random.uniform(0.0, start_lane.length),
            velocity=self.np_random.uniform(0.0, 15.0),
        )
        self._prev_pos = self.vehicle.position.copy()
        self.road.vehicles.append(self.vehicle)

    def _is_terminal(self):
        if self._lap_number >= self.race_config('max_lap_count'):
            return True

        if self._steps >= self.race_config('max_steps'):
            return True

        if self._out_of_lane_degree() > self.race_config('out_of_circuit_stop_margin'):
            return True

        return False

    def _out_of_lane_degree(self):
        long, lat = self.vehicle.lane.local_coordinates(self.vehicle.position)
        long_dist = max(max(-long, 0), max(long - self.vehicle.lane.length, 0))
        lat_dist = max(max(-lat - self.vehicle.lane.width * 0.5, 0), max(lat - self.vehicle.lane.width * 0.5, 0))
        return max(long_dist, lat_dist)

    def _reward(self, action):
        #circuit_pos = self._circuit.get_circuit_pos(self.vehicle.position)

        last_progress, last_finish_line_crossed = self._circuit.get_lap_progress(
            self._vehicle_pos_before_update, self.vehicle.position)
        # print('Last progress: %.1fm, last finish line crossed: %s' % (last_progress, last_finish_line_crossed))
        # print('Lap number: %d, relative lap pos: %.2f' % (
        #     self._lap_number, circuit_pos / self._circuit.circuit_length
        # ))

        reward = self.reward_config('tick')
        self.vehicle.lane.local_coordinates(self.vehicle.position)
        reward += self._out_of_lane_degree() * self.reward_config('out_of_circuit_reward_scale')
        reward += self.reward_config('lap_progress') * last_progress

        # print('Action: {0} reward: {1}'.format(action, reward))
        return reward


register(
    id='racecircuit-v0',
    entry_point='trackdays.envs.race_circuit:RaceCircuitEnv'
)

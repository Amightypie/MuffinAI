import sc2
from sc2 import run_game, maps, Race, Difficulty, bot_ai
from sc2.player import Bot, Computer
from sc2.constants import *
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features

from sc2.ids import ability_id
import random
import math
import numpy as np
import pandas as pd
import os
from absl import app

EXTRACTOR = UnitTypeId.EXTRACTOR
HATCHERY = UnitTypeId.HATCHERY
DRONE = UnitTypeId.DRONE
LARVA = UnitTypeId.LARVA
OVERLORD = UnitTypeId.OVERLORD
ROACHWARREN = UnitTypeId.ROACHWARREN

do_nothing = 'donothing'
expand_now = 'expansionnow'
train_zergling = 'trainzergling'
train_drone = 'traindrone'
train_overlord = 'trainoverlord'
build_spawningpool = 'buildspawnpool'
build_extractor = 'buildextractor'
distribute_drones = 'distributedrones'
attack = 'attacknow'
DATA_FILE = 'muffin_ai_brain'

enemy_player = 4

smart_actions = [
    do_nothing,
    train_zergling,
    train_drone,
    train_overlord,
    build_spawningpool,
    build_extractor,
    distribute_drones,
    expand_now
]

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 2) % 8 == 0 and (mm_y + 2) % 8 == 0:
            smart_actions.append(attack + '_' + str(mm_x - 4) + '_' + str(mm_y - 4))


class MuffinAI(base_agent.BaseAgent, sc2.BotAI):
    def __init__(self):
        super(MuffinAI, self).__init__()
        # for location in bot_ai.BotAI.expansion_locations:
        #    smart_actions.append(expand_now + '_' + location)
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_action = None
        self.move_number = 0
        self.base_top_left = None
        self.previous_state = None
        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
            self.qlearn.q_table.to_csv(DATA_FILE + '.csv')

    # UTILITY COMMANDS
    def split_action(self, action_id):
        smart_action = smart_actions[action_id]
        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')
        return smart_action, x, y

    def transformlocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]
        return [x, y]

    def make_supply(self, larvae):
        if self.can_afford(OVERLORD) and larvae.exists:
            self.do(larvae.random.train(OVERLORD))

    def build_workers(self, larvae):
        if self.can_afford(DRONE):
            self.do(larvae.random.train(DRONE))

    def expand(self):

        self.expand_now()

    def build_extractor(self):
        mineralworkerdeficit = 0
        gasworkerdeficit = 0
        for hatchery in self.units(HATCHERY).ready:
            vespenes = self.state.vespene_geyser.closer_than(25, hatchery)
            for vespene in vespenes:
                if not self.can_afford(UnitTypeId.EXTRACTOR):
                    break
                worker = self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.units(UnitTypeId.EXTRACTOR).closer_than(1.0, vespene).exists:
                    if self.can_afford(UnitTypeId.EXTRACTOR):
                        self.do(worker.build(UnitTypeId.EXTRACTOR, vespene))

    def on_step(self, obs):
        if obs.last():
            reward = self.reward
            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
            self.previous_action = None
            self.previous_state = None
            self.move_number = 0
            return actions.FUNCTIONS.no_op()

        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            player_x = player_x.mean()
            player_y = player_y.mean()
            xmean = player_x.mean()
            ymean = player_y.mean()
            self.base_top_left = 1 if player_y.any() and ymean <= 31 else 0

        owned_expansions = self.owned_expansions
        larvae = self.units(LARVA)
        overlord_count = self.units(OVERLORD).amount
        spawningpool_count = self.units(UnitTypeId.SPAWNINGPOOL)
        supply_free= self.supply_left
        supply_cap=self.supply_cap
        army_supply=self.supply_used - self.workers.amount
        worker_supply=self.workers.amount
        extractor_count = self.units(UnitTypeId.EXTRACTOR).amount
        hatchery_count = self.units(UnitTypeId.HATCHERY).amount

        if self.move_number == 0:
            self.move_number += 1

            current_state = np.zeros(134)  # 64 - hotsqaures + 64 - green squares + # of stats i want
            current_state[0] = overlord_count
            current_state[1] = spawningpool_count
            current_state[2] = supply_free
            current_state[3] = supply_cap
            current_state[4] = army_supply
            current_state[5] = worker_supply

            hot_squares = np.zeros(64)
            enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == enemy_player).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 8))
                x = int(math.ceil((enemy_x[i] + 1) / 8))

                hot_squares[((y - 1) * 8) + (x - 1)] = 1
            if not self.base_top_left:  # orientates hot-squares to top-left start
                hot_squares = hot_squares[::-1]
            for i in range(0, 64):
                current_state[i + 6] = hot_squares[i]

            green_squares = np.zeros(64)
            friendly_y, friendly_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF) \
                .nonzero()
            for i in range(0, len(friendly_y)):
                y = int(math.ceil((friendly_y[i] + 1) / 8))
                x = int(math.ceil((friendly_x[i] + 1) / 8))

                green_squares[((y - 1) * 8) + (x - 1)] = 1

            if not self.base_top_left:
                green_squares = green_squares[::-1]

            for i in range(0, 64):
                current_state[i + 70] = green_squares[i]

            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            excluded_actions = []
            # actions to be ignored if not possible this iteration/step

            if army_supply == 0:
                for action in smart_actions:
                    if attack in action:
                        excluded_actions.append(smart_actions.index(action))
            if supply_free == 0:
                excluded_actions.append(1)
                excluded_actions.append(2)
            if not self.can_afford(UnitTypeId.ZERGLING): # workers and Zerglings cost the same
                excluded_actions.append(1)              # so cant build either cant build both
                excluded_actions.append(2)
            if supply_free >5 or not self.can_afford(OVERLORD):
                excluded_actions.append(3)

            if spawningpool_count == 1 or  not self.can_afford(UnitTypeId.SPAWNINGPOOL):
                excluded_actions.append(4)

            if extractor_count >= hatchery_count*2:
                excluded_actions.append(5)

            if not self.can_afford(HATCHERY):
                excluded_actions.append(7)

            rl_action = self.qlearn.choose_action(str(current_state), excluded_actions)
            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action, x, y = self.split_action(self.previous_action)

            if smart_action == do_nothing:  # return if we want to do nothing
                self.move_number = 0
                return actions.FUNCTIONS.no_op()

            if smart_action == train_overlord:
                if self.supply_left < 5 and larvae.exists:
                    self.make_supply(larvae)

            if smart_action == train_drone:
                mineralworkerdeficit=0
                gasworkerdeficit=0
                if self.geysers:  # if there are any geysers (never know!)
                    for g in self.geysers:  # count how many workers down for gas harvesting
                        actual = g.assigned_harvesters
                        ideal = g.ideal_harvesters
                        gasworkerdeficit = ideal - actual
                if owned_expansions:  # if we own any bases
                    for location, townhall in owned_expansions.items():  # gather how many workers down for mining
                        actual = townhall.assigned_harvesters
                        ideal = townhall.ideal_harvesters
                        mineralworkerdeficit = ideal - actual

                if (mineralworkerdeficit + gasworkerdeficit) > 3 * len(
                        self.units(HATCHERY)):  # only care if we're down 3 per hatch
                    workerdown = 0
                    while workerdown < (mineralworkerdeficit + gasworkerdeficit):  # stock up on drones
                        self.build_workers(larvae)

            if smart_action == distribute_drones:
                if self.workers.idle:
                    self.distribute_workers()  # send them to work

            if smart_action == expand_now:
                if self.units(HATCHERY).amount < 2 and self.can_afford(HATCHERY):
                    self.expand()

            if smart_action == build_extractor:
                self.build_extractor()
            if smart_action == attack:
                self.move_number = 0
                if self.units.not_structure.exclude_type(DRONE).amount > 15:
                    army = self.units.not_structure.exclude_type(DRONE)
                    target_location = self.transformlocation(int(x), int(y))
                    return actions.FUNCTIONS.Attack_minimap('now', target_location)
        if self.move_number == 1:
            self.move_number = 0


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.disallowed_actions = {}

    def choose_action(self, observation, excluded_actions=[]):
        self.check_state_exist(observation)
        self.disallowed_actions[observation] = excluded_actions
        state_action = self.q_table.ix[observation, :]
        for excluded_action in excluded_actions:
            del state_action[excluded_action]

        if np.random.uniform() < self.epsilon:
            # choose best action
            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        if s == s_:
            return
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]
        s_rewards = self.q_table.ix[s_, :]

        if s_ in self.disallowed_actions:
            for excluded_action in self.disallowed_actions[s_]:
                del s_rewards[excluded_action]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal
        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


# def main(unused_argv):
#     agent=MuffinAI()
#     try:
#         while True:
#             run_game(maps.get("(2)AcidPlantLE"), [
#             Bot(Race.Zerg, agent),
#             Computer(Race.Zerg, Difficulty.Medium)
#         ], realtime=True)
#             while True:
#                 agent.on_step(0)
#     except KeyboardInterrupt:
#         pass

def main(unused_argv):
    agent = MuffinAI()
    print(agent.units(OVERLORD))
    try:
        while True:
            with sc2_env.SC2Env(map_name="AcidPlant",
                                players=[sc2_env.Agent(sc2_env.Race.zerg),
                                         sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.easy)],
                                agent_interface_format=features.AgentInterfaceFormat(
                                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                                    use_feature_units=True),
                                step_mul=16,
                                game_steps_per_episode=0, visualize=True)as env:
                agent.setup(env.observation_spec(), env.action_spec())

                timesteps = env.reset()
                agent.reset()
                while True:
                    step_actions = [agent.on_step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.on_step(step_actions)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)

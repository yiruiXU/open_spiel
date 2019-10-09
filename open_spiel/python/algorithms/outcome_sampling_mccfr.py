# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python implementation for Monte Carlo Counterfactual Regret Minimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pyspiel

# Indices in the information sets for the regrets and average policy sums.
#信息集中的表示遗憾和平均保单金额的指数。
_REGRET_INDEX = 0
_AVG_POLICY_INDEX = 1


class OutcomeSamplingSolver(object):
  """An implementation of outcome sampling MCCFR.

  Uses stochastically-weighted averaging.
  结果采样MCCFR的实现
   使用随机加权平均
  For details, see Chapter 4 (p. 49) of:
  http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf
  (Lanctot, 2013. "Monte Carlo Sampling and Regret Minimization for Equilibrium
  Computation and Decision-Making in Games")
  """

  def __init__(self, game):
    self._game = game
    self._infostates = {}  # infostate keys -> [regrets, avg strat]
    self._num_players = game.num_players()
    # This is the epsilon exploration factor. When sampling episodes, the
    # updating player will sampling according to expl * uniform + (1 - expl) *
    # current_policy.
    #这是epsilon探索因素。 采样剧集时，更新播放器将根据expl * uniform + (1 - expl) *current_policy进行更新
    self._expl = 0.6

    assert game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL, (
        "MCCFR requires sequential games. If you're trying to run it " +
        "on a simultaneous (or normal-form) game, please first transform it " +
        "using turn_based_simultaneous_game.")

  def iteration(self):
    """Performs one iteration of outcome sampling.

    An iteration consists of one episode for each player as the update player.
    """
    #执行一次结果采样迭代。
    #迭代由每个玩家作为更新玩家的一个情节组成。
    for update_player in range(self._num_players):
      state = self._game.new_initial_state()
      self._episode(
          state, update_player, my_reach=1.0, opp_reach=1.0, sample_reach=1.0)

  def _lookup_infostate_info(self, info_state_key, num_legal_actions):
    """Looks up an information set table for the given key.
        查找给定键的信息集表。
    Args:
      info_state_key: information state key (string identifier).
      num_legal_actions: number of legal actions at this information state.

    Returns:
      A list of:
        - the average regrets as a numpy array of shape [num_legal_actions]
        - the average strategy as a numpy array of shape [num_legal_actions].
          The average is weighted using `my_reach`
    """
    retrieved_infostate = self._infostates.get(info_state_key, None)
    if retrieved_infostate is not None:
      return retrieved_infostate

    # Start with a small amount of regret and total accumulation, to give a
    # uniform policy: this will get erased fast.
    #从少量的遗憾和总的积累开始，制定统一的政策：这将很快被消除。
    self._infostates[info_state_key] = [
        np.ones(num_legal_actions, dtype=np.float64) / 1000.0,
        np.ones(num_legal_actions, dtype=np.float64) / 1000.0,
    ]
    return self._infostates[info_state_key]

  def _add_regret(self, info_state_key, action_idx, amount):
    self._infostates[info_state_key][_REGRET_INDEX][action_idx] += amount

  def _add_avstrat(self, info_state_key, action_idx, amount):
    self._infostates[info_state_key][_AVG_POLICY_INDEX][action_idx] += amount

  def callable_avg_policy(self):
    """Returns the average joint policy as a callable.

    The callable has a signature of the form string (information
    state key) -> list of (action, prob).
    将平均联合保单返回为可赎回价格。
   可调用对象具有形式字符串的签名（信息状态密钥）->（动作，概率）列表。
    """

    def wrap(state):
      info_state_key = state.information_state(state.current_player())
      legal_actions = state.legal_actions()
      infostate_info = self._lookup_infostate_info(info_state_key,
                                                   len(legal_actions))
      avstrat = (
          infostate_info[_AVG_POLICY_INDEX] /
          infostate_info[_AVG_POLICY_INDEX].sum())
      return [(legal_actions[i], avstrat[i]) for i in range(len(legal_actions))]

    return wrap

  def _regret_matching(self, regrets, num_legal_actions):
    """Applies regret matching to get a policy.

    Args:
      regrets: numpy array of regrets for each action.
      num_legal_actions: number of legal actions at this state.

    Returns:
      numpy array of the policy indexed by the index of legal action in the
      list.
      应用后悔匹配以获取策略。

     精氨酸：
       遗憾：每一个动作都会感到遗憾。
       num_legal_actions：在此状态下法律诉讼的数量。

     返回值：
       由列表中的法律行动索引索引的策略的numpy数组。
    """
    positive_regrets = np.maximum(regrets,
                                  np.zeros(num_legal_actions, dtype=np.float64))
    sum_pos_regret = positive_regrets.sum()
    if sum_pos_regret <= 0:
      return np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions
    else:
      return positive_regrets / sum_pos_regret

  def _episode(self, state, update_player, my_reach, opp_reach, sample_reach):
    """Runs an episode of outcome sampling.

    Args:
      state: the open spiel state to run from (will be modified in-place).
      update_player: the player to update regrets for (the other players update
        average strategies)
      my_reach: reach probability of the update player
      opp_reach: reach probability of all the opponents (including chance)
      sample_reach: reach probability of the sampling (behavior) policy

    Returns:
      A tuple of (util, reach_tail), where:
        - util is the utility of the update player divided by the sample reach
          of the trajectory, and
        - reach_tail is the product of all players' reach probabilities
          to the terminal state (from the state that was passed in).
          
Runs an episode of outcome sampling.

    Args:
      state: the open spiel state to run from (will be modified in-place).
      update_player: the player to update regrets for (the other players update average strategies)
      my_reach: reach probability of the update player
      opp_reach: reach probability of all the opponents (including chance)
      sample_reach: reach probability of the sampling (behavior) policy

    Returns:
      A tuple of (util, reach_tail), where:
        - util is the utility of the update player divided by the sample reach of the trajectory, and
        - reach_tail is the product of all players' reach probabilities to the terminal state (from the state that was passed in).
715/5000
运行结果抽样情节。

     精氨酸：
       状态：要运行的开放spiel状态（将被就地修改）。
       update_player：要更新遗憾的玩家（其他玩家更新平均策略）
       my_reach：达到更新播放器的概率
       opp_reach：所有对手的达到概率（包括机会）
       sample_reach：达到采样（行为）策略的概率

     返回值：
       （util，reach_tail）的元组，其中：
         -util是更新播放器的实用程序除以轨迹的样本范围，并且
         -reach_tail是所有玩家到达终端状态（从传入状态开始）的到达概率的乘积。
    """
    if state.is_terminal():
      return state.player_return(update_player) / sample_reach, 1.0

    if state.is_chance_node():
      outcomes, probs = zip(*state.chance_outcomes())
      outcome = np.random.choice(outcomes, p=probs)
      state.apply_action(outcome)
      return self._episode(state, update_player, my_reach, opp_reach,
                           sample_reach)

    cur_player = state.current_player()
    info_state_key = state.information_state(cur_player)
    legal_actions = state.legal_actions()
    num_legal_actions = len(legal_actions)
    infostate_info = self._lookup_infostate_info(info_state_key,
                                                 num_legal_actions)
    policy = self._regret_matching(infostate_info[_REGRET_INDEX],
                                   num_legal_actions)
    if cur_player == update_player:
      uniform_policy = (
          np.ones(num_legal_actions, dtype=np.float64) / num_legal_actions)
      sampling_policy = (
          self._expl * uniform_policy + (1.0 - self._expl) * policy)
    else:
      sampling_policy = policy
    sampled_action_idx = np.random.choice(
        np.arange(num_legal_actions), p=sampling_policy)
    if cur_player == update_player:
      new_my_reach = my_reach * policy[sampled_action_idx]
      new_opp_reach = opp_reach
    else:
      new_my_reach = my_reach
      new_opp_reach = opp_reach * policy[sampled_action_idx]
    new_sample_reach = sample_reach * sampling_policy[sampled_action_idx]
    state.apply_action(legal_actions[sampled_action_idx])
    util, reach_tail = self._episode(state, update_player, new_my_reach,
                                     new_opp_reach, new_sample_reach)
    new_reach_tail = policy[sampled_action_idx] * reach_tail
    # The following updates are based on equations 4.9 - 4.15 (Sec 4.2) of
    # http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf
    if cur_player == update_player:
      # update regrets. Note the w here already includes the sample reach of the
      # trajectory (from root to terminal) in util due to the base case above.
      w = util * opp_reach
      for action_idx in range(num_legal_actions):
        if action_idx == sampled_action_idx:
          self._add_regret(info_state_key, action_idx,
                           w * (reach_tail - new_reach_tail))
        else:
          self._add_regret(info_state_key, action_idx, -w * new_reach_tail)
    else:
      # update avg strat
      for action_idx in range(num_legal_actions):
        self._add_avstrat(info_state_key, action_idx,
                          opp_reach * policy[action_idx] / sample_reach)
    return util, new_reach_tail

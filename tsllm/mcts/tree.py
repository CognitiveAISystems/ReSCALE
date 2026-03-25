"""
The Node and MCTS class for AlphaZero.
"""
#
import copy
import json
import math

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
from tsllm.distributed.utils import print_rank_0, print_with_rank
from tsllm.envs.base_env import CoTEnv
import pdb
from tqdm import tqdm
import heapq


class Node(object):
    """
    Overview:
        The node base class for tree_search.
    """

    def __init__(
        self, parent: "Node" = None, prior_p: float = 1.0, initial_value: float = 0.0
    ) -> None:
        self._parent = parent
        self._children = {}
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = prior_p
        self.prior_p_ori = prior_p

        self._initial_value = initial_value
        self.max_value = initial_value
        self._terminated = False

    def __lt__(self, other):
        return self._initial_value < other._initial_value

    @property
    def terminated(self):
        return self._terminated

    def set_as_terminate_node(self):
        self._terminated = True

    @property
    def true_value(self):
        if self._visit_count == 0:
            return -math.inf

        return self.value

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        if self._visit_count == 0:
            return self._initial_value
        return self._value_sum / self._visit_count

    def update(self, value: float, max_value: float) -> None:
        """
        Overview:
            Updata the current node information, such as visit_count and value_sum.
        Arguments:
            - value (:obj:`Int`): The value of the node.
        """
        self._visit_count += 1
        self._value_sum += value
        self.max_value = max(self.max_value, max_value)

    def update_recursive(self, leaf_value: float, mcts_mode: str, max_value: float) -> None:
        """
        Overview:
            Update node information recursively.
        Arguments:
            - leaf_value (:obj:`Int`): The value of the node.
        """
        if mcts_mode == "self_play_mode":
            self.update(leaf_value, max_value)
            if self.is_root():
                return
            self._parent.update_recursive(-leaf_value, mcts_mode, max_value)
        if mcts_mode == "play_with_bot_mode":
            self.update(leaf_value, max_value)
            if self.is_root():
                return
            self._parent.update_recursive(leaf_value, mcts_mode, max_value)

    def is_leaf(self) -> Dict:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Dict`): Dict type children node.
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
        """
        return self._parent is None

    @property
    def parent(self) -> None:
        return self._parent

    @property
    def children(self) -> None:
        return self._children

    @property
    def visit_count(self) -> None:
        return self._visit_count

    def get_info(self):
        # return [
        #     "visit_cnt: {}, value: {:.6f}, prior: {:.6f}".format(
        #         self.visit_count, self.value, self.prior_p)
        # ]
        return {
            "visit_cnt": self.visit_count,
            "value": self.value,
            "prior_p": float(self.prior_p_ori),
            "initial_value": self._initial_value,
            "terminated": self.terminated,
        }

    def clear(self):
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = self.prior_p_ori

    def to_json(self):
        childrens = {}
        for name, child_node in self.children.items():
            childrens[name] = child_node.to_json()

        rets = {"children": childrens, "info": self.get_info()}
        return rets


class LanguageNode(Node):
    text_state: Optional[str] = None
    last_action: Optional[str] = None
    prm_value: Optional[float] = None
    num_generated_token: Optional[int] = None

    def __init__(
        self,
        parent: Node = None,
        prior_p: float = 1.0,
        prm_value: Optional[float] = None,
        text_state: Optional[str] = None,
        last_action: Optional[str] = None,
        initial_value: float = 0.0,
        num_generated_token: Optional[int] = None,
    ) -> None:
        super().__init__(parent, prior_p, initial_value)
        self.text_state = text_state
        self.last_action = last_action
        self.prm_value = prm_value

        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False

    def get_path(self):
        ans = []
        node = self
        while not node.is_root():
            ans.append(node.last_action)
            node = node.parent
        return "\n".join(reversed(ans))

    def get_info(self):
        info_dict = super().get_info()
        if not self.is_root():
            info_dict["last_action"] = self.last_action
            info_dict["prm_value"] = self.prm_value
        else:
            info_dict["text_state"] = self.text_state
        return info_dict

    def to_dict(self, level=0):
        if self.is_root():
            action = self.text_state
        else:
            action = self.last_action

        info = {'action': action, 'num_tokens': self.num_generated_token, 'prior_p': self.prior_p,
                'value': self.value, 'num_visits': self.visit_count, 'terminated': self.terminated,
                'level': level, 'on_path': False}

        if self.is_leaf():
            info['children'] = []
        else:
            info['children'] = [child.to_dict(level=level + 1) for child in self.children.values()]

        return info


class GumbelNode(LanguageNode):
    def __init__(
        self,
        parent: Node = None,
        prior_p: float = 1.0,
        prm_value: Optional[float] = None,
        text_state: Optional[str] = None,
        last_action: Optional[str] = None,
        initial_value: float = 0.0,
        num_generated_token: Optional[int] = None,
    ) -> None:
        super().__init__(parent, prior_p, prm_value, text_state, last_action, initial_value, num_generated_token)
        if prior_p > 0:
            self.prior_log_p = math.log(prior_p)
        elif prior_p == 0:
            self.prior_log_p = -math.inf
        else:
            raise ValueError(f'Unexpected prior_p: type: {type(prior_p)} value: {prior_p}')

    def get_mixed_value_approximation(self) -> float:
        sum_visits = sum([child.visit_count for child in self.children.values()])
        sum_visited_pi = 0
        sum_visited_pi_q = 0
        for action in self.children:
            node = self.children[action]
            if node.visit_count > 0:
                pi = node.prior_p
                sum_visited_pi += pi
                sum_visited_pi_q += pi * node.value

        mixed_value = self._initial_value
        if sum_visited_pi != 0:
            mixed_value += (sum_visits / sum_visited_pi) * sum_visited_pi_q
        mixed_value /= 1. + sum_visits

        return mixed_value

    def get_estimated_q_tensor(self) -> Tuple[np.ndarray, np.ndarray]:
        values = []
        is_unvisited = []
        for child in self.children.values():
            values.append(child.true_value)
            is_unvisited.append(child.visit_count == 0)
        return np.array(values), np.array(is_unvisited)

    def get_completed_q_values(self) -> np.ndarray:
        completed_q, unvisited_children = self.get_estimated_q_tensor()
        value_approximation = self.get_mixed_value_approximation()
        completed_q[unvisited_children] = value_approximation

        return completed_q

    @property
    def children_prior_logits(self):
        prior_logits = [child.prior_log_p for child in self.children.values()]
        return np.asarray(prior_logits, dtype=np.float64)

    def get_altered_visit_count_distribution_tensor(self):
        visit_counts = np.asarray([child.visit_count for child in self.children.values()], dtype=np.float64)
        return visit_counts / (1 + visit_counts.sum())


def get_root(node: Node):
    while not node.is_root():
        node = node.parent
    return node


def tree_to_dict(node: LanguageNode):
    path = []
    while not node.is_root():
        path.append(node.last_action)
        node = node.parent

    tree_dict = node.to_dict()
    tree_dict['on_path'] = True
    tree_dict['response'] = '\n'.join(path[::-1])
    current_node_dict = tree_dict
    for action in path[::-1]:
        found = False
        for child in current_node_dict['children']:
            if child['action'] == action:
                current_node_dict = child
                current_node_dict['on_path'] = True
                found = True
                break

        if not found:
            raise ValueError(f'Cannot find child for action: {action}')

    return tree_dict


class MCTS(object):
    """
    Overview:
        MCTS search process.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg

        self._num_simulations = self._cfg.get("num_simulations", 20)

        # UCB formula
        self._pb_c_base = self._cfg.get("pb_c_base", 19652)  # 19652
        self._pb_c_init = self._cfg.get("pb_c_init", 1.25)  # 1.25

        # Root prior exploration noise.
        self._root_dirichlet_alpha = self._cfg.get(
            "root_dirichlet_alpha", 0.3
        )  # 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self._root_noise_weight = self._cfg.get("root_noise_weight", 0.25)  # 0.25

        self._prm_factor = self._cfg.get("prm_factor", 0.5)
        self.root = None

        self.answers = set()
        self.wrong_answers = set()
        self.visited_paths = None

        self.no_terminal_reward = self._cfg.get("no_terminal_reward", True)
        self.mask_non_terminal_node_value = self._cfg.get(
            "mask_non_terminal_node_value", False
        )

        self._init_critic_value = self._cfg.get("init_critic_value", True)

        self._num_generated_token = 0

        self._prune_node_under_v = self._cfg.get("prune_node_under_v", None)
        self._final_action_strategy = self._cfg["final_action_strategy"]
        self._sequential_halving_start_nodes = self._cfg["sequential_halving_start_nodes"]
        self.gumbel_c_visit = self._cfg.get("gumbel_c_visit", 50)
        self.gumbel_c_scale = self._cfg.get("gumbel_c_scale", 1)

        self.non_root_child_selection_mode = self._cfg.get("non_root_child_selection_mode", "ucb")
        self._use_gumbel_noise_in_alpha_mcts = self._cfg.get("use_gumbel_noise_in_alpha_mcts", False)
        self._use_gumbel_noise_in_gumbel_mcts = self._cfg.get("use_gumbel_noise_in_gumbel_mcts", True)

        if self._sequential_halving_start_nodes > self._num_simulations:
            raise ValueError(f"Sequential Halving start nodes can't be greater than number of simulations: {self._sequential_halving_start_nodes} > {self._num_simulations}")

    @property
    def num_generated_token(self):
        return self._num_generated_token

    def clear_node(self, node):
        assert node is not None
        node.clear()
        for child in node.children.values():
            self.clear_node(child)

    def get_final_action_by_visits(self, temperature, sample, root, simulate_env):
        action_visits = []
        for action_dict in simulate_env.legal_actions:
            action = action_dict["action"]
            if action in root.children:
                action_visits.append((action, root.children[action].visit_count))
            else:
                action_visits.append((action, 0))

        actions, visits = zip(*action_visits)
        action_probs = nn.functional.softmax(
            1.0
            / temperature
            * np.log(torch.as_tensor(visits, dtype=torch.float32) + 1e-10),
            dim=0,
        ).numpy()
        if sample:
            action = np.random.choice(actions, p=action_probs)
            self.reset_prior(root)
        else:
            action = actions[np.argmax(action_probs)]

        return action, action_probs

    def get_final_action_by_expected_value(self, temperature, sample, root, simulate_env):
        action_values = []
        for action_dict in simulate_env.legal_actions:
            action = action_dict["action"]
            if action in root.children:
                action_values.append((action, root.children[action].value))
            else:
                action_values.append((action, -math.inf))

        actions, values = zip(*action_values)
        action_probs = nn.functional.softmax(
            1.0
            / temperature
            * torch.as_tensor(values, dtype=torch.float32),
            dim=0,
        ).numpy()
        if sample:
            action = np.random.choice(actions, p=action_probs)
            self.reset_prior(root)
        else:
            action = actions[np.argmax(action_probs)]

        return action, action_probs

    def get_final_action_by_max_value(self, temperature, sample, root, simulate_env):
        action_values = []
        for action_dict in simulate_env.legal_actions:
            action = action_dict["action"]
            if action in root.children:
                action_values.append((action, root.children[action].max_value))
            else:
                action_values.append((action, -math.inf))

        actions, values = zip(*action_values)
        action_probs = nn.functional.softmax(
            1.0
            / temperature
            * torch.as_tensor(values, dtype=torch.float32),
            dim=0,
        ).numpy()
        if sample:
            action = np.random.choice(actions, p=action_probs)
            self.reset_prior(root)
        else:
            action = actions[np.argmax(action_probs)]

        return action, action_probs

    def sigma_q(self, node: Node, q_values: np.ndarray):
        visits = [child.visit_count for child in node.children.values()]
        max_visit = 0 if len(visits) == 0 else max(visits)
        return (self.gumbel_c_visit + max_visit) * self.gumbel_c_scale * q_values

    def get_next_action(
        self,
        simulate_env: Type[CoTEnv],
        policy_forward_fn: Optional[Callable] = None,
        temperature: int = 1.0,
        sample: bool = True,
        return_tree=False,
    ) -> Tuple[int, List[float]]:
        """
        Overview:
            calculate the move probabilities based on visit counts at the root node.
        Arguments:
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - policy_forward_fn (:obj:`Function`): The Callable to compute the action probs and state value.
            - temperature (:obj:`Int`): Temperature is a parameter that controls the "softness" of the probability distribution.
            - sample (:obj:`Bool`): The value of the node.
        Returns:
            - action (:obj:`Bool`): Select the action with the most visits as the final action.
            - action_probs (:obj:`List`): The output probability of each action.
        """
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, policy_forward_fn)
            self.root = root
        else:
            root = self.root

        if root.is_leaf():
            # if root is leaf node, expand it
            # We have updated the environment legal action when we test the node is leaf node
            # So the expansion won't have bugs
            self._expand_leaf_node(root, simulate_env, policy_forward_fn)

        if sample:
            self._add_exploration_noise(root)

        if self._use_gumbel_noise_in_alpha_mcts:
            self._add_gumbel_noise(root)

        for n in range(self._num_simulations):
            simulate_env_copy = simulate_env.copy()
            simulate_env_copy.battle_mode = simulate_env_copy.mcts_mode
            self._simulate(root, simulate_env_copy, policy_forward_fn)

        # for debugging
        # print('after simulation')
        # print('value= {}'.format([(k, v.value) for k,v in root.children.items()]))
        # print('visit_count= {}'.format([(k, v.visit_count) for k,v in root.children.items()]))

        if self._final_action_strategy == "visits":
            action, action_probs = self.get_final_action_by_visits(temperature, sample, root, simulate_env)
        elif self._final_action_strategy == "expected_value":
            action, action_probs = self.get_final_action_by_expected_value(temperature, sample, root, simulate_env)
        elif self._final_action_strategy == "max_value":
            action, action_probs = self.get_final_action_by_max_value(temperature, sample, root, simulate_env)
        else:
            assert False, f'Unexpected final action strategy: {self._final_action_strategy}'

        self.root = root
        if return_tree:
            return action, action_probs, root
        return action, action_probs

    @staticmethod
    def get_sequential_halving_simulations_for_levels(num_actions_at_root: int, simulation_budget: int):
        num_simulations_per_action = []
        actions_on_levels = []

        # number of levels if simulations
        num_levels = math.floor(math.log2(num_actions_at_root))

        remaining_actions = num_actions_at_root
        remaining_budget = simulation_budget
        for level in range(num_levels):
            if level > 0:
                remaining_actions = max(2, math.floor(remaining_actions / 2))

            if remaining_budget < remaining_actions:
                break

            actions_on_levels.append(remaining_actions)
            num_simulations_per_action.append(
                max(1, math.floor(remaining_budget / ((num_levels - level) * remaining_actions)))
            )
            remaining_budget -= num_simulations_per_action[-1] * actions_on_levels[-1]

        if remaining_budget > 0:
            num_simulations_per_action[-1] += remaining_budget // actions_on_levels[-1]

        return actions_on_levels, num_simulations_per_action

    def get_next_action_gumbel(
        self,
        simulate_env: Type[CoTEnv],
        policy_forward_fn: Optional[Callable] = None,
        temperature: int = 1.0,
        sample: bool = True,
        return_tree=False,
    ) -> Tuple[int, List[float]]:
        """
        Overview:
            calculate the move probabilities based on visit counts at the root node.
        Arguments:
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - policy_forward_fn (:obj:`Function`): The Callable to compute the action probs and state value.
            - temperature (:obj:`Int`): Temperature is a parameter that controls the "softness" of the probability distribution.
            - sample (:obj:`Bool`): The value of the node.
        Returns:
            - action (:obj:`Bool`): Select the action with the most visits as the final action.
            - action_probs (:obj:`List`): The output probability of each action.
        """
        if self.root is None:
            root = GumbelNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, policy_forward_fn)
            self.root = root
        else:
            root = self.root

        if root.is_leaf():
            # if root is leaf node, expand it
            # We have updated the environment legal action when we test the node is leaf node
            # So the expansion won't have bugs
            self._expand_leaf_node(root, simulate_env, policy_forward_fn)

        possible_actions = list(self.root.children)
        if len(possible_actions) == 1:
            if return_tree:
                return possible_actions[0], np.ones((1,), dtype=np.float32), root
            else:
                return possible_actions[0], np.ones((1,), dtype=np.float32)

        if self._use_gumbel_noise_in_gumbel_mcts:
            gumbel_logits = np.random.gumbel(size=len(possible_actions))
        else:
            gumbel_logits = np.zeros((len(possible_actions),), dtype=np.float64)

        gumbel_logits += np.array([child.prior_log_p for child in self.root.children.values()])
        num_actions_on_levels, num_simulations_per_action_on_levels = self.get_sequential_halving_simulations_for_levels(
            min(self._sequential_halving_start_nodes, len(possible_actions)), self._num_simulations)

        for level, (num_actions, num_simulations_per_action) in enumerate(zip(num_actions_on_levels, num_simulations_per_action_on_levels)):
            if level == 0:
                selected_child_ids = np.argpartition(gumbel_logits, len(possible_actions) - num_actions)[-num_actions:]
            else:
                estimated_q = self.root.get_estimated_q_tensor()[0]
                updated_gumbels = gumbel_logits + self.sigma_q(self.root, estimated_q)
                selected_gumbels = updated_gumbels[selected_child_ids]
                selected_child_ids = selected_child_ids[np.argpartition(selected_gumbels, selected_gumbels.shape[0] - num_actions)[-num_actions:]]

            for child_id in selected_child_ids:
                for _ in range(num_simulations_per_action):
                    simulate_env_copy = simulate_env.copy()
                    simulate_env_copy.battle_mode = simulate_env_copy.mcts_mode
                    self._simulate(root, simulate_env_copy, policy_forward_fn, first_action=possible_actions[child_id])

        if selected_child_ids.shape[0] > 1:
            estimated_q = self.root.get_estimated_q_tensor()[0]
            updated_gumbels = gumbel_logits + self.sigma_q(self.root, estimated_q)
            selected_gumbels = updated_gumbels[selected_child_ids]
            selected_action_id = selected_child_ids[np.argmax(selected_gumbels)]
        else:
            selected_action_id = selected_child_ids[0]

        action = possible_actions[selected_action_id]
        action_probs = np.zeros(len(possible_actions), dtype=np.float32)
        action_probs[selected_action_id] = 1

        # for debugging
        # print('after simulation')
        # print('value= {}'.format([(k, v.value) for k,v in root.children.items()]))
        # print('visit_count= {}'.format([(k, v.visit_count) for k,v in root.children.items()]))

        self.root = root
        if return_tree:
            return action, action_probs, root
        return action, action_probs

    @torch.inference_mode()
    def try_search_right_answer(
        self,
        simulate_env: Type[CoTEnv],
        policy_forward_fn: Optional[Callable] = None,
        sample: bool = True,
        save_path: Optional[str] = None,
    ) -> Tuple[int, List[float]]:
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self.root = root
            self._expand_leaf_node(root, simulate_env, policy_forward_fn)
        if sample:
            self._add_exploration_noise(root)

        def save_tree():
            if save_path is not None:
                json.dump(root.to_json(), open(save_path, "w"), indent=2)

        for n in range(self._num_simulations):
            simulate_env_copy = simulate_env.copy()
            simulate_env_copy.battle_mode = simulate_env_copy.mcts_mode
            self._simulate(root, simulate_env_copy, policy_forward_fn)

            if len(self.answers) > 0:
                save_tree()
                return True

        save_tree()
        return False

    def rollout(
        self,
        simulate_env: Type[CoTEnv],
        num_paths: int,
        policy_forward_fn: Optional[Callable] = None,
        *,
        max_num_simulation: Optional[int] = 200,
        max_token: Optional[int] = 25482,
        sample: bool = True,
        return_tree: bool = False
    ) -> List[Dict]:
        assert (max_num_simulation is None) ^ (max_token is None)

        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, policy_forward_fn)
            self.root = root
        else:
            root = self.root

        self.visited_paths = []
        cnt = 0
        traj_list = []
        visit_path_num = 0

        while len(self.visited_paths) < num_paths:
            cnt += 1
            if max_num_simulation is not None and cnt > max_num_simulation:
                print_with_rank(
                    "exit for max num simulation, #current_paths: {}".format(
                        len(self.visited_paths)
                    )
                )
                break
            elif max_token is not None and self._num_generated_token > max_token:
                print_with_rank(
                    "exit for exceed max generated token {}>{}, #current_paths: {}".format(
                        self._num_generated_token, max_token, len(self.visited_paths)
                    )
                )
                break
            simulate_env_copy = simulate_env.copy()
            simulate_env_copy.battle_mode = simulate_env_copy.mcts_mode
            self._simulate(root, simulate_env_copy, policy_forward_fn)

            if len(self.visited_paths) > len(traj_list):
                assert len(self.visited_paths) == len(traj_list) + 1
                # which means include new path
                new_visit_path = self.visited_paths[-1]
                traj_data = {
                    "path_idx": len(traj_list),
                    "text": new_visit_path["text"],
                    "value": new_visit_path["value"],
                    "num_generated_token": self._num_generated_token,
                }
                traj_list.append(traj_data)

        if return_tree:
            return traj_list, cnt, self.root
        return traj_list, cnt

    def rap(
        self,
        simulate_env: Type[CoTEnv],
        num_paths: int,
        policy_forward_fn: Optional[Callable] = None,
        select_by_prior: bool = False,
    ) -> List[Dict]:
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, policy_forward_fn)
            self.root = root

        traj_list = []
        for i_path in range(num_paths):
            node = self.root
            env_copy = simulate_env.copy()
            done = False
            while not done:
                if select_by_prior:
                    # select action node by the logp_score of LLM itself
                    action, node = self._select_by_prior(node, env_copy)
                else:
                    # select by PUCT
                    action, node = self._select_child(node, env_copy)

                _, _, terminated, truncated, info = env_copy.step(
                    action, update_legal_action=node.is_leaf()
                )
                done = terminated or truncated
                if not done and node.is_leaf():
                    self._expand_leaf_node(node, env_copy, policy_forward_fn)

            if not self.no_terminal_reward:  
                winner = info["winner"]
                if "reward" in info.keys(): # handle rlhf special case
                    leaf_value = info["reward"]
                else:
                    if winner == -1:
                        leaf_value = 0
                    elif winner == 1:
                        leaf_value = 1
                    elif winner == 2:
                        leaf_value = -1
            else:
                if node.visit_count > 0:
                    leaf_value = node.value
                else:
                    if self._init_critic_value:
                        leaf_value = node._initial_value
                    else:
                        leaf_value = policy_forward_fn(env_copy.get_state()).item()

            node.update_recursive(leaf_value, env_copy.mcts_mode, node.max_value)

            traj_data = {
                "path_idx": i_path,
                "text": env_copy.answer,
                "value": leaf_value,
                "num_generated_token": self._num_generated_token,
            }

            traj_list.append(traj_data)

        return traj_list

    def beam_search(
        self,
        simulate_env: Type[CoTEnv],
        beam_size: int,
        max_step: int,
        policy_forward_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, policy_forward_fn)
            self.root = root

        end_nodes, top_k_nodes = [], [(-root._initial_value, root, simulate_env.copy())]
        k = beam_size

        for _ in range(max_step + 1):
            cur_nodes_to_search = top_k_nodes
            top_k_nodes = []
            for cur_neg_v, cur_node, cur_env in cur_nodes_to_search:
                if cur_node.terminated:
                    end_nodes.append((cur_neg_v, cur_node, cur_env))
                    k -= 1
                elif k > 0:
                    # select at most topk children add push to heap
                    assert (
                        len(cur_node.children) > 0
                    ), "in beam search you should expand this non-terminal node at first."
                    self._num_generated_token += sum(
                        c.num_generated_token for c in cur_node.children.values()
                    )
                    top_k_children = sorted(
                        [
                            (action, child, child._initial_value)
                            for action, child in cur_node.children.items()
                        ],
                        key=lambda x: x[2],
                        reverse=True,
                    )[:k]
                    for c_act, c_node, c_value in top_k_children:
                        # new_env = cur_env.copy()
                        # _, _, terminated, truncated, info = new_env.step(
                        #     c_act, update_legal_action=True
                        # )
                        # if terminated or truncated:
                        #     c_node.set_as_terminate_node()
                        # else:
                        #     self._expand_leaf_node(c_node, new_env, policy_forward_fn)
                        new_env = cur_env.copy()
                        heapq.heappush(top_k_nodes, (-c_value, c_node, new_env))
            # nsmallest since we negate the value
            top_k_nodes = heapq.nsmallest(k, top_k_nodes)

            # expand selected nodes
            for value, node, new_env in top_k_nodes:
                _, _, terminated, truncated, info = new_env.step(
                    node.last_action, update_legal_action=True
                )
                if terminated or truncated:
                    node.set_as_terminate_node()
                else:
                    self._expand_leaf_node(node, new_env, policy_forward_fn)

            if len(end_nodes) == beam_size:
                assert k == 0
                break

        traj_list = []
        for i, (neg_e_v, e_node, e_env) in enumerate(end_nodes):
            traj_list.append(
                {
                    "path_idx": i,
                    "text": e_env.answer,
                    "value": -neg_e_v,
                    "num_generated_token": None,
                    # num_generated_token is hard to compute, since we
                    #  allow beam size to be larger than max_action of a node.
                }
            )
        traj_list[-1]["num_generated_token"] = self._num_generated_token
        return traj_list

    def dfs(
        self,
        simulate_env: Type[CoTEnv],
        num_paths: int,
        policy_forward_fn: Optional[Callable] = None,
        prune_value: Optional[float] = None,
        prune_ratio: Optional[float] = None,
    ) -> List[Dict]:
        if prune_ratio:
            assert 1 > prune_ratio > 0

        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state())
            self._expand_leaf_node(root, simulate_env, policy_forward_fn)
            self.root = root

        # end_nodes = []
        traj_list = []
        # num_visited_node = 0

        def execute_dfs(cur_node, cur_env):
            if cur_node.terminated:
                # end_nodes.append((cur_node._initial_value, cur_node, cur_env))
                traj_list.append(
                    {
                        "path_idx": len(traj_list),
                        "text": cur_env.answer,
                        "value": cur_node._initial_value,
                        "num_generated_token": self._num_generated_token,
                    }
                )
            else:
                assert len(cur_node.children.values()) > 0, "node must have children"
                self._num_generated_token += sum(
                    c.num_generated_token for c in cur_node.children.values()
                )

                # you can select only top k children to expand here with [:k].
                for i, child in enumerate(
                    sorted(
                        cur_node.children.values(),
                        key=lambda x: x._initial_value,
                        reverse=True,
                    )
                ):
                    # # pruned by enough visited nodes.
                    # if num_visited_node >= step_limit:
                    #     return
                    # else:
                    #     num_visited_node += 1

                    # sample at most num_paths answers
                    if len(traj_list) >= num_paths:
                        return

                    if prune_value is not None and child._initial_value < prune_value:
                        # if we don't have any answer yet, we will not prune.
                        # since we sorted w.r.t. _initial_value
                        return

                    if prune_ratio is not None and i > (1 - prune_ratio) * len(
                        cur_node.children
                    ):
                        return
                    copy_env = cur_env.copy()
                    _, _, terminated, truncated, info = copy_env.step(
                        child.last_action, update_legal_action=True
                    )

                    if terminated or truncated:
                        child.set_as_terminate_node()
                    else:
                        self._expand_leaf_node(child, copy_env, policy_forward_fn)
                    execute_dfs(child, copy_env)

        execute_dfs(self.root, simulate_env.copy())

        
        return traj_list

    def _simulate(
        self,
        node: Node,
        simulate_env: Type[CoTEnv],
        policy_forward_fn: Optional[Callable] = None,
        first_action=None,
    ) -> None:
        """
        Overview:
            Run a single playout from the root to the leaf, getting a value at the leaf and propagating it back through its parents.
            State is modified in-place, so a deepcopy must be provided.
        Arguments:
            - node (:obj:`Class Node`): Current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): The class of simulate env.
            - policy_forward_fn (:obj:`Function`): The Callable to compute the action probs and state value.
        """
        # XXX: fix the bug temporally, better implementation is required.
        winner = None
        done = False
        while not node.is_leaf():
            if first_action is None:
                if self.non_root_child_selection_mode == 'ucb':
                    action, node = self._select_child(node, simulate_env)
                elif self.non_root_child_selection_mode == 'gumbel':
                    action, node = self._select_child_gumbel(node, simulate_env)
                else:
                    assert False, f'Unexpected non_root_child_selection_mode={self.non_root_child_selection_mode}'
            else:
                if not node.has_collected_token_num:
                    self._num_generated_token += sum(
                        c.num_generated_token for c in node.children.values()
                    )
                    node.has_collected_token_num = True

                node = node.children[first_action]
                action = first_action
                first_action = None

            _, _, terminated, truncated, info = simulate_env.step(
                action, update_legal_action=(node.is_leaf() and node.visit_count == 1)
            )
            done = terminated or truncated

            # In original AlphaZero, the leaf node will be expanded once it is reached
            # In our setting, computing legal action is computational inefficient
            # Thus when we reach a leaf node, we will not directly expand it
            # Until the next time, when this node's children are required to be selected
            # In this case, node is leaf node and the visit count number of node is 1
            # Then we expand it

            if not done and node.is_leaf() and node.visit_count == 1:
                # Once we expand the node, the node will not be leaf node any more
                # And the while won't break
                self._expand_leaf_node(node, simulate_env, policy_forward_fn)

            winner = info["winner"]
        """
        in ``self_play_mode``, the leaf_value is calculated from the perspective of player ``simulate_env.current_player``.
        in ``play_with_bot_mode``, the leaf_value is calculated from the perspective of player 1.
        """
        if not done:
            # leaf_value = self._expand_leaf_node(node, simulate_env,
            #                                     policy_forward_fn)

            if not done and self.mask_non_terminal_node_value:
                leaf_value = 0.0
            else:
                if not self._init_critic_value:
                    leaf_value = policy_forward_fn(simulate_env.get_state()).item()
                else:
                    leaf_value = node._initial_value
        else:
            if not self.no_terminal_reward:
                if winner is not None:
                    if winner == 1:
                        self.answers.add(simulate_env.answer)
                    else:
                        self.wrong_answers.add(simulate_env.answer)

                # if simulate_env.mcts_mode == 'self_play_mode':
                #     if winner == -1:
                #         leaf_value = 0
                #     else:
                #         leaf_value = 1 if simulate_env.current_player == winner else -1

                if simulate_env.mcts_mode == "play_with_bot_mode":
                    # in ``play_with_bot_mode``, the leaf_value should be transformed to the perspective of player 1.
                    if "reward" in info.keys():
                        leaf_value = info["reward"]
                    else:
                        if winner == -1:
                            leaf_value = 0
                        elif winner == 1:
                            leaf_value = 1
                        elif winner == 2:
                            leaf_value = -1
            else:
                if node.visit_count > 0:
                    # because leaf value has been calculated and backpropogated
                    leaf_value = node.value
                else:
                    if self._init_critic_value:
                        leaf_value = node._initial_value
                    else:
                        leaf_value = policy_forward_fn(simulate_env.get_state()).item()

        if done:
            node.set_as_terminate_node()
            if self.visited_paths is not None:
                self.visited_paths.append(
                    {
                        "text": simulate_env.answer,
                        "correct": winner == 1,
                        "value": leaf_value,
                    }
                )

        # Update value and visit count of nodes in this traversal.
        if simulate_env.mcts_mode == "play_with_bot_mode":
            node.update_recursive(leaf_value, simulate_env.mcts_mode, node.max_value)

        elif simulate_env.mcts_mode == "self_play_mode":
            # NOTE: e.g.
            #       to_play: 1  ---------->  2  ---------->  1  ----------> 2
            #         state: s1 ---------->  s2 ---------->  s3 ----------> s4
            #                                     action    node
            #                                            leaf_value
            # leaf_value is calculated from the perspective of player 1, leaf_value = value_func(s3),
            # but node.value should be the value of E[q(s2, action)], i.e. calculated from the perspective of player 2.
            # thus we add the negative when call update_recursive().
            node.update_recursive(-leaf_value, simulate_env.mcts_mode, node.max_value)

    def _select_child_gumbel(self, node: GumbelNode, simulate_env: Type[CoTEnv]):
        if not node.has_collected_token_num:
            self._num_generated_token += sum(
                c.num_generated_token for c in node.children.values()
            )
            node.has_collected_token_num = True

        improved_policy = self.get_improved_policy(node)
        action_id = np.argmax(improved_policy - node.get_altered_visit_count_distribution_tensor())
        action, child = list(node.children.items())[action_id]

        return action, child

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def get_improved_policy(self, node: GumbelNode):
        completed_q_values = node.get_completed_q_values()
        sigma_q_values = self.sigma_q(node, completed_q_values)
        improved_policy = self.softmax(node.children_prior_logits + sigma_q_values)
        return improved_policy

    def _select_child(
        self, node: LanguageNode, simulate_env: Type[CoTEnv]
    ) -> Tuple[Union[int, float], Node]:
        """
        Overview:
            Select the child with the highest UCB score.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        Returns:
            - action (:obj:`Int`): choose the action with the highest ucb score.
            - child (:obj:`Node`): the child node reached by executing the action with the highest ucb score.
        """
        if not node.has_collected_token_num:
            self._num_generated_token += sum(
                c.num_generated_token for c in node.children.values()
            )
            node.has_collected_token_num = True

        action = None
        child = None
        best_score = -9999999

        scores = {}

        for action_tmp, child_tmp in node.children.items():
            # print(a, simulate_env.legal_actions)
            # if action_tmp in simulate_env.legal_actions:
            ucb_score = self._ucb_score(node, child_tmp)
            prm_value = 0.0 if child_tmp.prm_value is None else child_tmp.prm_value
            score = ucb_score + self._prm_factor * prm_value
            if score > best_score:
                best_score = score
                action = action_tmp
                child = child_tmp
            scores[action_tmp] = (score, ucb_score, child_tmp.prm_value)

        if child is None:
            child = node  # child==None, node is leaf node in play_with_bot_mode.

        # print("score: {}\n\n\tchoose_action: {}\n".format(
        #     json.dumps(scores, indent=2), action))
        return action, child

    def _select_by_prior(self, node: Node, simulate_env):
        data_tmp = [
            (x_action, x_node.prior_p) for x_action, x_node in node.children.items()
        ]
        action_list, prior_list = list(zip(*data_tmp))
        chosen_action = np.random.choice(action_list, p=np.array(prior_list))
        chosen_node = node.children[chosen_action]

        #  For select by prior, we should only calculate the token that
        #  is actually selected
        if not chosen_node.has_collected_token_num:
            self._num_generated_token += chosen_node.num_generated_token
            chosen_node.has_collected_token_num = True

        return chosen_action, chosen_node

    def _expand_leaf_node_without_value(
        self, node: Node, simulate_env: Type[CoTEnv]
    ) -> None:
        """
        Overview:
            expand the node without the policy_forward_fn.
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - policy_forward_fn (:obj:`Function`): the Callable to compute the action probs and state value.
        Returns:
            - leaf_value (:obj:`Bool`): the leaf node's value.
        """
        text_state = simulate_env.get_state()
        for i, action_dict in enumerate(simulate_env.legal_actions):
            action, prob = action_dict["action"], action_dict["prob"]
            node.children[action] = LanguageNode(
                parent=node,
                prior_p=prob,
                #  prm_value=prm_value,
                text_state=text_state,
                last_action=action,
                num_generated_token=action_dict["num_token"],
            )

    def _expand_leaf_node(
        self,
        node: Node,
        simulate_env: Type[CoTEnv],
        policy_forward_fn: Optional[Callable] = None,
    ) -> float:
        """
        Overview:
            expand the node with the policy_forward_fn.
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - policy_forward_fn (:obj:`Function`): the Callable to compute the action probs and state value.
        Returns:
            - leaf_value (:obj:`Bool`): the leaf node's value.
        """
        """
        action_probs_dict, leaf_value = policy_forward_fn(simulate_env)
        for action, prior_p in action_probs_dict.items():
            if action in simulate_env.legal_actions:
                node.children[action] = Node(parent=node, prior_p=prior_p)
        """
        # To implement for leaf value calcuation

        # if policy_forward_fn is not None:
        #     q_str = simulate_env.math_problem["question"]
        #     prefix = node.get_path()

        text_state = simulate_env.get_state()
        if not self._init_critic_value:
            leaf_value = policy_forward_fn(text_state).item()
        else:
            leaf_value = node._initial_value
            assert len(simulate_env.legal_actions) > 0
            child_values = policy_forward_fn(
                [
                    text_state + x["action"] + simulate_env.sep
                    for x in simulate_env.legal_actions
                ]
            ).tolist()

        assert len(node.children) == 0
        for i, action_dict in enumerate(simulate_env.legal_actions):
            action, prob = action_dict["action"], action_dict["prob"]
            # self._num_generated_token += action_dict["num_token"]

            # if policy_forward_fn is None:
            #     prm_value = None
            # else:
            #     prm_value = policy_forward_fn(q_str,
            #                                   prefix + "\n" + action + "\n")

            if self._init_critic_value:
                child_value = child_values[i]
            else:
                child_value = 0.0

            if self._prune_node_under_v is not None:
                assert (
                    self._init_critic_value
                ), "currently only support prune for init_critic_value setting."
                if child_value < self._prune_node_under_v:
                    # print_rank_0("Prune node of value {:.4f} < {:.4f}".format(
                    #     child_value, self._prune_node_under_v
                    # ))
                    continue

            node.children[action] = GumbelNode(
                parent=node,
                prior_p=prob,
                #  prm_value=prm_value,
                text_state=text_state,
                last_action=action,
                initial_value=child_value,
                num_generated_token=action_dict["num_token"],
            )
        if len(node.children) == 0:
            print_rank_0(
                "Prune all current children at node {}".format(node.last_action)
            )

        return leaf_value

    def _ucb_score(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute UCB score. The score for a node is based on its value, plus an exploration bonus based on the prior.
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
        Returns:
            - score (:obj:`Bool`): The UCB score.
        """
        pb_c = (
            math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base)
            + self._pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior_p
        value_score = child.value

        return prior_score + value_score
        # return value_score

    def reset_prior(self, node: Node) -> None:
        """
        Overview:
            Reset prior probability
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        for a in node.children.keys():
            node.children[a].prior_p = node.children[a].prior_p_ori

    def _add_exploration_noise(self, node: Node) -> None:
        """
        Overview:
            Add exploration noise.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        # Get a list of actions corresponding to the child nodes.
        actions = list(node.children.keys())
        # Create a list of alpha values for Dirichlet noise.
        alpha = [self._root_dirichlet_alpha] * len(actions)
        # Generate Dirichlet noise using the alpha values.
        noise = np.random.dirichlet(alpha)
        # Compute the weight of the exploration noise.
        frac = self._root_noise_weight
        # Update the prior probability of each child node with the exploration noise.
        for a, n in zip(actions, noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - frac) + n * frac

    def _add_gumbel_noise(self, node: Node):
        actions = list(node.children.keys())
        probs = np.array([node.children[a].prior_p for a in actions])
        noise_log_probs = np.log(probs) + np.random.gumbel(size=len(actions))
        noise_probs = self.softmax(noise_log_probs)
        for a, p in zip(actions, noise_probs):
            node.children[a].prior_p = p

    @classmethod
    def from_json(cls, cfg: dict, json_path: str, reset_visit_info: bool):
        tree_json = json.load(open(json_path, "r"))

        def build_tree(tree_dict: dict) -> Node:
            node_info = tree_dict["info"]
            current_node = LanguageNode(
                text_state=node_info.get("text_state", None),
                last_action=node_info.get("last_action", None),
                prior_p=node_info["prior_p"],
                prm_value=node_info.get("prm_value", None),
                initial_value=node_info.get("initial_value", 0.0),
            )

            if not reset_visit_info:
                current_node._visit_count = node_info["visit_cnt"]
                current_node._value_sum = node_info["value"] * current_node.visit_count
            if node_info.get("terminated", False):
                current_node.set_as_terminate_node()

            for name, child_dict in tree_dict["children"].items():
                child_node = build_tree(child_dict)
                current_node._children[name] = child_node
                child_node._parent = current_node

            return current_node

        root_node = build_tree(tree_dict=tree_json)

        obj = cls(cfg)
        obj.root = root_node
        return obj


if __name__ == "__main__":
    mcts_cfg = {
        "num_simulations": 200,
        "pb_c_base": 19652,
        "pb_c_init": 10,
        "root_dirichlet_alpha": 0.3,
        "root_noise_weight": 0.25,
    }

    tree_path = "./tree.json"

    mcts = MCTS.from_json(mcts_cfg, tree_path)

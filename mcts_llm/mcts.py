from __future__ import annotations

import math
import random
from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Generic, TypeVar, Union

State = TypeVar("State")
Action = TypeVar("Action")
Reward = TypeVar("Reward", bound=Union[float, Any])


class MCTSMeta(ABCMeta):
    pass


class MCTSNode(Generic[State, Reward]):
    __slots__ = ["S", "parent", "children", "N", "Q", "G"]

    def __init__(self, S: State, parent: MCTSNode | None = None):
        self.S = S
        self.parent = parent
        self.children = []
        self.N = 0
        self.Q = 0.0
        self.G = []

    def add_child(self, child_node: MCTSNode):
        self.children.append(child_node)

    def update(self, reward: Reward):
        self.N += 1
        self.G.append(reward)
        self.Q += (reward - self.Q) / self.N

    def is_fully_expanded(self, legal_actions: list[Any]) -> bool:
        return len(self.children) == len(legal_actions)


class MCTS(ABC, Generic[State, Action, Reward], metaclass=MCTSMeta):
    def __init__(
        self, max_rollouts: int = 4, c: float = 1.414, default_uct_score: float = float("inf"), *args, **kwargs
    ):
        self.max_rollouts = max_rollouts
        self.c = c
        self.default_uct_score = default_uct_score

    @abstractmethod
    def get_actions(self, S: State) -> list[Action]:
        """Return a list of legal actions for the given state."""
        pass

    @abstractmethod
    def get_next_state(self, S: State, action: Action) -> State:
        """Return the next state after taking the given action."""
        pass

    @abstractmethod
    def is_terminal(self, S: State) -> bool:
        """Check if the given state is terminal."""
        pass

    @abstractmethod
    def get_reward(self, S: State) -> Reward:
        """Return the reward for the given state."""
        pass

    def initialize(self, S: State) -> MCTSNode:
        return MCTSNode(S=S)

    def search(self, S: State) -> State:
        root = self.initialize(S)
        for _ in range(self.max_rollouts):
            leaf = self.select(root)
            child = self.expand(leaf)
            result = self.simulate(child)
            self.backpropagate(child, result)
        return self._best_child(root).S

    def select(self, node: MCTSNode) -> MCTSNode:
        while not self.is_terminal(node.S):
            if not node.is_fully_expanded(self.get_actions(node.S)):
                return node
            node = self._select_child(node)
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        actions = self.get_actions(node.S)
        unexpanded_actions = [
            action
            for action in actions
            if not any(child.S == self.get_next_state(node.S, action) for child in node.children)
        ]
        if unexpanded_actions:
            action = random.choice(unexpanded_actions)
            S_next = self.get_next_state(node.S, action)
            child = MCTSNode(S_next, parent=node)
            node.add_child(child)
            return child
        return node

    def simulate(self, node: MCTSNode) -> Reward:
        S_next = node.S
        while not self.is_terminal(S_next):
            action = self._simulate_policy(S_next)
            S_next = self.get_next_state(S_next, action)
        return self.get_reward(S_next)

    def _simulate_policy(self, S: State) -> Action:
        return random.choice(self.get_actions(S))  # pragma: no cover

    def backpropagate(self, node: MCTSNode, reward: Reward):
        while node:
            node.update(reward)
            node = node.parent

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=lambda child: self._uct(child))

    def _uct(self, node: MCTSNode) -> float:
        if node.N == 0:
            return self.default_uct_score
        return (node.Q / node.N) + self.c * math.sqrt(math.log(node.parent.N) / node.N)

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=lambda child: child.N)

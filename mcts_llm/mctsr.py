"""Implementation of MCTSr as described in https://arxiv.org/abs/2406.07394. Uses the default MCTS class, but overrides
the select, expand, simulate, and backpropagate methods. Several differences from the original MCTS:

- Selection: Uses UCT value to rank all answers that were not fully expanded and selects the
    highest-valued node for further exploration and refinement using a greedy strategy or via
    importance sampling.
- Expansion: Uses self-refinement to produced a refined answer as the new state.
- Simulation: Generates multiple rewards on a single node and updates the node's Q value
    by averaging the rewards' minimum and mean, balancing worst-case and average outcome.
- Backpropagation: Updates the visit count instead of being done in the node's update method and the Q value.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Generator
from enum import Enum

import dspy
import numpy as np

from mcts_llm.mcts import MCTS, MCTSMeta, MCTSNode
from mcts_llm.utils import parse_integer_answer


class InitializeStrategy(Enum):
    ZERO_SHOT = 1
    DUMMY_ANSWER = 2


class Policy(Enum):
    GREEDY = 1
    IMPORTANCE_SAMPLING = 2


class ZeroShotAnswer(dspy.Signature):
    problem: str = dspy.InputField()
    answer: str = dspy.OutputField()


class CritiqueAnswer(dspy.Signature):
    problem: str = dspy.InputField()
    current_answer: str = dspy.InputField()
    critique: str = dspy.OutputField()


class RefineAnswer(dspy.Signature):
    """[[ ## proposed_instruction ## ]] Given a mathematical problem, a current answer, and a critique of that answer,
    refine the current answer to provide a more accurate and well-reasoned solution. Begin by carefully analyzing the
    problem and the critique, then think step by step to derive the correct answer. Ensure that your reasoning is clear
    and logical, and that the final answer is justified by the steps taken.

    [[ ## completed ## ]]
    """

    problem: str = dspy.InputField()
    current_answer: str = dspy.InputField()
    critique: str = dspy.InputField()
    answer: str = dspy.OutputField()


class EvaluateAnswer(dspy.Signature):
    problem: str = dspy.InputField()
    answer: str = dspy.InputField()
    score: int = dspy.OutputField(ge=-100, le=100)


class ZeroShotCoT(dspy.Module):
    def __init__(self):
        self.cot = dspy.TypedChainOfThought(ZeroShotAnswer)

    def forward(self, problem) -> dspy.Prediction:
        return dspy.Prediction(answer=self.cot(problem=problem).answer)


class MultipleTurnSelfRefine(dspy.Module):
    def __init__(self, num_turns: int = 1):
        super().__init__()
        self.zero_shot_cot = ZeroShotCoT()
        self.critique_answer = dspy.TypedChainOfThought(CritiqueAnswer)
        self.refine_answer = dspy.TypedChainOfThought(RefineAnswer)
        self.num_turns = num_turns

    def forward(self, problem) -> dspy.Prediction:
        current_answer = self.zero_shot_cot(problem=problem).answer

        for _ in range(self.num_turns):
            critique_result = self.critique_answer(problem=problem, current_answer=current_answer)
            refined_result = self.refine_answer(
                problem=problem, current_answer=current_answer, critique=critique_result.critique
            )
            current_answer = refined_result.answer

        return dspy.Prediction(answer=current_answer)


class MCTSrState:
    __slots__ = ["problem", "answer"]

    def __init__(self, problem: str, answer: str):
        self.problem = problem
        self.answer = answer


class MCTSrNode(MCTSNode):
    def __init__(self, S: MCTSrState, parent: MCTSrNode | None = None):
        super().__init__(S, parent)

    def update(self, reward: int):
        self.G.append(reward)
        self.Q = (min(self.G) + np.mean(self.G)) / 2


class ModuleMeta(type(dspy.Module)):
    pass


class CombinedMeta(MCTSMeta, ModuleMeta):
    pass


class MCTSr(MCTS, dspy.Module, metaclass=CombinedMeta):
    def __init__(
        self,
        max_rollouts: int = 4,
        c: float = math.sqrt(2),
        max_children: int = 2,
        eps: float = 1e-8,
        reward_ub: int = 95,
        reward_penalty: int = 50,
        default_uct_score: float = 1000,
        dummy_answer: str = "I don't know.",
        policy: Policy = Policy.GREEDY,
        initialize_strategy: InitializeStrategy = InitializeStrategy.DUMMY_ANSWER,
        num_turns: int = 1,
        samples_per_node: int = 3,
    ):
        MCTS.__init__(self, max_rollouts=max_rollouts, c=c, default_uct_score=default_uct_score)
        dspy.Module.__init__(self)
        self.max_children = max_children
        self.eps = eps
        self.reward_ub = reward_ub
        self.reward_penalty = reward_penalty
        self.dummy_answer = dummy_answer
        self.policy = policy
        self.num_turns = num_turns
        self.initialize_strategy = initialize_strategy
        self.samples_per_node = samples_per_node

        self.zero_shot = ZeroShotCoT()
        self.critique = dspy.TypedChainOfThought(CritiqueAnswer)
        self.evaluate = dspy.TypedChainOfThought(EvaluateAnswer)
        self.refine = dspy.TypedChainOfThought(RefineAnswer)

    def initialize(self, S: MCTSrState) -> MCTSrNode:
        if self.initialize_strategy == InitializeStrategy.ZERO_SHOT:
            root = MCTSrNode(S=MCTSrState(problem=S.problem, answer=self.zero_shot.forward(problem=S.problem).answer))
        elif self.initialize_strategy == InitializeStrategy.DUMMY_ANSWER:
            root = MCTSrNode(S=MCTSrState(problem=S.problem, answer=self.dummy_answer))
        else:
            raise ValueError(f"Initialize Strategy `{self.initialize_strategy}` does not exist")

        return root

    def forward(self, problem) -> dspy.Prediction:
        S_best = self.search(S=MCTSrState(problem=problem, answer=self.dummy_answer))
        return dspy.Prediction(answer=S_best.answer)

    def is_terminal(self, node: MCTSNode) -> bool:
        return len(node.children) >= self.max_children or any(child.Q > node.Q for child in node.children)

    def get_actions(self, S: MCTSrState) -> list[str]:
        pass

    def get_next_state(self, S: MCTSrState, action=None) -> MCTSrState:
        current_answer = S.answer
        for _ in range(self.num_turns):
            critique = self.critique(problem=S.problem, current_answer=current_answer).critique
            refined = self.refine(problem=S.problem, current_answer=current_answer, critique=critique)
            current_answer = refined.answer
        S_next = MCTSrState(problem=S.problem, answer=current_answer)
        return S_next

    def get_reward(self, S: MCTSrState) -> int:
        reward = self.evaluate(problem=S.problem, answer=S.answer).score
        reward = parse_integer_answer(reward) if not isinstance(reward, int) else reward
        return min(reward, self.reward_ub) - self.reward_penalty if reward > self.reward_ub else reward

    def select(self, root: MCTSrNode) -> MCTSrNode:
        children = [child for child in self._traverse_tree(root) if not self.is_terminal(child)]
        if not children:
            return root

        uct_scores = np.array([self._uct(child) for child in children])

        if self.policy == Policy.GREEDY:
            node = children[np.argmax(uct_scores)]
        elif self.policy == Policy.IMPORTANCE_SAMPLING:
            probabilities = uct_scores / np.sum(uct_scores)
            node = np.random.choice(children, p=probabilities)
        else:
            raise ValueError(f"Selection Policy `{self.policy}` does not exist")

        return node

    def expand(self, node: MCTSrNode) -> MCTSrNode:
        S_next = self.get_next_state(S=node.S)
        child = MCTSrNode(S=S_next, parent=node)
        node.add_child(child)
        return child

    def simulate(self, node: MCTSrNode) -> list[int]:
        rewards = [self.get_reward(S=node.S) for _ in range(self.samples_per_node)]
        node.update(np.mean(rewards))
        return rewards

    def backpropagate(self, node: MCTSrNode, rewards: list[int] = None):
        while node.parent:
            node.parent.N += 1
            node.parent.Q = (node.parent.Q + max(child.Q for child in node.parent.children)) / 2
            node = node.parent

    def _uct(self, node: MCTSrNode) -> float:
        if not node.parent:
            return self.default_uct_score
        return node.Q + self.c * math.sqrt(math.log(node.parent.N + 1) / (node.N + self.eps))

    def _traverse_tree(self, root: MCTSNode) -> Generator[MCTSNode, None, None]:
        queue = deque([root])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    def _best_child(self, root: MCTSNode) -> MCTSNode:
        return max(self._traverse_tree(root), key=lambda node: node.Q)

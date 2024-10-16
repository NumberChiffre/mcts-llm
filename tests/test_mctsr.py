import math
from unittest.mock import Mock, patch

import dspy
import numpy as np
import pytest

from mcts_llm.mctsr import (
    InitializeStrategy,
    MCTSr,
    MCTSrNode,
    MCTSrState,
    MultipleTurnSelfRefine,
    Policy,
    ZeroShotCoT,
)


@pytest.fixture
def mock_parse_integer_answer():
    with patch("mcts_llm.mctsr.parse_integer_answer", return_value=50) as mock:
        yield mock


@pytest.fixture
def mock_chain_of_thought():
    with patch("dspy.TypedChainOfThought") as mock:
        mock_instance = Mock()
        mock_instance.return_value = Mock(answer="Mocked answer", critique="Mock critique")
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mctsr(mock_parse_integer_answer, mock_chain_of_thought):
    with patch("mcts_llm.mctsr.ZeroShotCoT") as mock_zero_shot, patch(
        "mcts_llm.mctsr.CritiqueAnswer"
    ) as mock_critique, patch("mcts_llm.mctsr.RefineAnswer") as mock_refine, patch(
        "mcts_llm.mctsr.EvaluateAnswer"
    ) as mock_evaluate:
        mock_zero_shot_instance = Mock()
        mock_zero_shot_instance.forward.return_value = dspy.Prediction(answer="Mock zero-shot answer")
        mock_zero_shot.return_value = mock_zero_shot_instance

        mock_critique.return_value = Mock(critique="Mock critique")
        mock_refine.return_value = Mock(answer="Mock refined answer")
        mock_evaluate.return_value = Mock(score=50)

        mcts = MCTSr(max_rollouts=2, max_children=2, samples_per_node=1)

        mcts.zero_shot = mock_zero_shot_instance
        mcts.critique = mock_critique
        mcts.refine = mock_refine
        mcts.evaluate = mock_evaluate

        yield mcts


def test_zero_shot_cot():
    with patch("mcts_llm.mctsr.dspy.TypedChainOfThought") as mock_chain_of_thought:
        mock_cot = Mock()
        mock_chain_of_thought.return_value = mock_cot
        mock_cot.return_value = dspy.Prediction(answer="Test answer")

        zero_shot = ZeroShotCoT()
        result = zero_shot.forward("Test problem")

    assert isinstance(result, dspy.Prediction)
    assert result.answer == "Test answer"
    mock_cot.assert_called_once_with(problem="Test problem")


@pytest.mark.parametrize("num_turns", [1, 3])
def test_multiple_turn_self_refine(num_turns):
    with patch("mcts_llm.mctsr.dspy.TypedChainOfThought") as mock_chain_of_thought:
        mock_zero_shot = Mock()
        mock_critique = Mock()
        mock_refine = Mock()

        def side_effect(*args, **kwargs):
            if "ZeroShotAnswer" in str(args[0]):
                return mock_zero_shot
            elif "CritiqueAnswer" in str(args[0]):
                return mock_critique
            elif "RefineAnswer" in str(args[0]):
                return mock_refine

        mock_chain_of_thought.side_effect = side_effect

        mock_zero_shot.return_value = dspy.Prediction(answer="Initial answer")
        mock_critique.return_value = dspy.Prediction(critique="Test critique")
        mock_refine.return_value = dspy.Prediction(answer="Refined answer")

        mtsf = MultipleTurnSelfRefine(num_turns=num_turns)
        result = mtsf.forward("Test problem")

    assert isinstance(result, dspy.Prediction)
    assert result.answer == "Refined answer"
    mock_zero_shot.assert_called_once_with(problem="Test problem")
    assert mock_critique.call_count == num_turns
    assert mock_refine.call_count == num_turns


def test_policy_enum():
    assert Policy.GREEDY.value == 1
    assert Policy.IMPORTANCE_SAMPLING.value == 2


def test_mctsr_initialization():
    mcts = MCTSr(
        max_rollouts=5,
        c=1.5,
        max_children=3,
        eps=1e-6,
        reward_ub=90,
        reward_penalty=40,
        default_uct_score=500,
        dummy_answer="No answer",
        policy=Policy.IMPORTANCE_SAMPLING,
        initialize_strategy=InitializeStrategy.ZERO_SHOT,
        num_turns=2,
        samples_per_node=4,
    )

    assert mcts.max_rollouts == 5
    assert mcts.c == 1.5
    assert mcts.max_children == 3
    assert mcts.eps == 1e-6
    assert mcts.reward_ub == 90
    assert mcts.reward_penalty == 40
    assert mcts.default_uct_score == 500
    assert mcts.dummy_answer == "No answer"
    assert mcts.policy == Policy.IMPORTANCE_SAMPLING
    assert mcts.initialize_strategy == InitializeStrategy.ZERO_SHOT
    assert mcts.num_turns == 2
    assert mcts.samples_per_node == 4


@pytest.mark.parametrize("strategy", [InitializeStrategy.ZERO_SHOT, InitializeStrategy.DUMMY_ANSWER])
def test_initialize(mctsr, strategy):
    mctsr.initialize_strategy = strategy
    state = MCTSrState(problem="Test problem", answer="Test answer")
    root = mctsr.initialize(state)

    assert isinstance(root, MCTSrNode)
    assert root.S.problem == "Test problem"

    if strategy == InitializeStrategy.ZERO_SHOT:
        mctsr.zero_shot.forward.assert_called_once_with(problem="Test problem")
        assert root.S.answer == "Mock zero-shot answer"
    else:
        assert root.S.answer == mctsr.dummy_answer


def test_get_next_state(mctsr):
    initial_state = MCTSrState(problem="Test problem", answer="Initial answer")
    next_state = mctsr.get_next_state(initial_state)

    assert isinstance(next_state, MCTSrState)
    assert next_state.problem == "Test problem"
    assert next_state.answer != "Initial answer"

    mctsr.critique.assert_called_once_with(problem="Test problem", current_answer="Initial answer")
    mctsr.refine.assert_called_once()

    assert next_state.answer == mctsr.refine.return_value.answer


def test_get_reward(mctsr):
    state = MCTSrState(problem="Test problem", answer="Test answer")
    reward = mctsr.get_reward(state)

    mctsr.evaluate.assert_called_once_with(problem="Test problem", answer="Test answer")
    assert reward == 50


def test_select_return_root(mctsr):
    root = MCTSrNode(MCTSrState("Test problem", "Root answer"))
    child1 = MCTSrNode(MCTSrState("Test problem", "Child 1 answer"), parent=root)
    child2 = MCTSrNode(MCTSrState("Test problem", "Child 2 answer"), parent=root)
    root.add_child(child1)
    root.add_child(child2)

    root.Q = 50
    child1.Q = 40
    child2.Q = 40

    mctsr.max_children = 2

    with patch.object(mctsr, "is_terminal", return_value=True):
        selected_node = mctsr.select(root)
        assert selected_node == root, "Should return root when all children are terminal"


@pytest.mark.parametrize("policy", [Policy.GREEDY, Policy.IMPORTANCE_SAMPLING])
def test_select(mctsr, policy):
    mctsr.policy = policy
    root = MCTSrNode(MCTSrState("Test problem", "Root answer"))
    child1 = MCTSrNode(MCTSrState("Test problem", "Child 1 answer"), parent=root)
    child2 = MCTSrNode(MCTSrState("Test problem", "Child 2 answer"), parent=root)
    root.add_child(child1)
    root.add_child(child2)

    child1.Q = 10
    child2.Q = 20

    selected_node = mctsr.select(root)

    if policy == Policy.GREEDY:
        assert selected_node == child2
    else:
        assert selected_node in [child1, child2]


def test_expand(mctsr):
    parent = MCTSrNode(MCTSrState("Test problem", "Parent answer"))
    child = mctsr.expand(parent)

    assert isinstance(child, MCTSrNode)
    assert child.parent == parent
    assert parent.children == [child]
    assert child.S.problem == "Test problem"
    assert child.S.answer == "Mock refined answer"


def test_simulate(mctsr):
    node = MCTSrNode(MCTSrState("Test problem", "Test answer"))
    rewards = mctsr.simulate(node)

    assert len(rewards) == 1
    assert all(reward == 50 for reward in rewards)
    assert node.Q == 50


def test_backpropagate(mctsr):
    root = MCTSrNode(MCTSrState("Test problem", "Root answer"))
    child1 = MCTSrNode(MCTSrState("Test problem", "Child 1 answer"), parent=root)
    child2 = MCTSrNode(MCTSrState("Test problem", "Child 2 answer"), parent=root)
    root.add_child(child1)
    root.add_child(child2)

    child1.Q = 10
    child2.Q = 20

    mctsr.backpropagate(child2)

    assert root.N == 1
    assert root.Q == 10


def test_uct(mctsr):
    parent = MCTSrNode(MCTSrState("Test problem", "Parent answer"))
    child = MCTSrNode(MCTSrState("Test problem", "Child answer"), parent=parent)
    parent.add_child(child)

    parent.N = 10
    child.N = 5
    child.Q = 20

    uct_score = mctsr._uct(child)
    expected_score = 20 + mctsr.c * np.sqrt(np.log(11) / (5 + mctsr.eps))

    assert np.isclose(uct_score, expected_score)


def test_best_child(mctsr):
    root = MCTSrNode(MCTSrState("Test problem", "Root answer"))
    child1 = MCTSrNode(MCTSrState("Test problem", "Child 1 answer"), parent=root)
    child2 = MCTSrNode(MCTSrState("Test problem", "Child 2 answer"), parent=root)
    root.add_child(child1)
    root.add_child(child2)

    child1.Q = 10
    child2.Q = 20

    best_child = mctsr._best_child(root)
    assert best_child == child2


def test_forward(mctsr):
    with patch.object(mctsr, "search") as mock_search:
        mock_best_state = MCTSrState(problem="Test problem", answer="Best answer")
        mock_search.return_value = mock_best_state

        result = mctsr("Test problem")

        mock_search.assert_called_once()
        assert isinstance(result, dspy.Prediction)
        assert result.answer == "Best answer"


@pytest.mark.parametrize(
    "num_children, child_q, max_children, expected",
    [
        (1, 40, 2, False),
        (2, 40, 2, True),
        (1, 60, 2, True),
        (2, 60, 2, True),
        (1, 40, 1, True),
    ],
)
def test_is_terminal(mctsr, num_children, child_q, max_children, expected):
    mctsr.max_children = max_children
    node = MCTSrNode(MCTSrState("Test problem", "Test answer"))
    node.Q = 50
    for _ in range(num_children):
        child = MCTSrNode(MCTSrState("Test problem", "Child answer"), parent=node)
        child.Q = child_q
        node.add_child(child)

    assert mctsr.is_terminal(node) == expected


def test_get_reward_exceeds_upper_bound(mctsr):
    mctsr.reward_ub = 80
    mctsr.reward_penalty = 10
    mctsr.evaluate.return_value = Mock(score=90)

    state = MCTSrState(problem="Test problem", answer="Test answer")
    reward = mctsr.get_reward(state)

    assert reward == 70


def test_uct_cases(mctsr):
    root = MCTSrNode(MCTSrState("Test problem", "Root answer"))
    uct_score = mctsr._uct(root)
    assert uct_score == mctsr.default_uct_score, "Root node should return default UCT score"

    parent = MCTSrNode(MCTSrState("Test problem", "Parent answer"))
    child = MCTSrNode(MCTSrState("Test problem", "Child answer"), parent=parent)
    parent.add_child(child)
    parent.N = 1

    uct_score = mctsr._uct(child)
    expected_score = mctsr.c * math.sqrt(math.log(2) / mctsr.eps)
    assert math.isclose(uct_score, expected_score, rel_tol=1e-9), f"Expected {expected_score}, but got {uct_score}"

    child.N = 1
    child.Q = 10
    uct_score = mctsr._uct(child)
    expected_score = 10 + mctsr.c * math.sqrt(math.log(2) / (1 + mctsr.eps))
    assert math.isclose(uct_score, expected_score, rel_tol=1e-9), f"Expected {expected_score}, but got {uct_score}"


def test_initialize_invalid_strategy():
    mcts = MCTSr()
    mcts.initialize_strategy = "INVALID"
    with pytest.raises(ValueError, match="Initialize Strategy `INVALID` does not exist"):
        mcts.initialize(MCTSrState("Test problem", "Test answer"))


def test_get_actions():
    mcts = MCTSr()
    assert mcts.get_actions(MCTSrState("Test problem", "Test answer")) is None


@pytest.mark.parametrize("policy", ["INVALID", None])
def test_select_invalid_policy(policy):
    mcts = MCTSr()
    mcts.policy = policy
    root = MCTSrNode(MCTSrState("Test problem", "Root answer"))
    with pytest.raises(ValueError, match=f"Selection Policy `{policy}` does not exist"):
        mcts.select(root)


def test_traverse_tree():
    mctsr = MCTSr()
    root = MCTSrNode(MCTSrState("Test problem", "Root answer"))
    child1 = MCTSrNode(MCTSrState("Test problem", "Child 1 answer"), parent=root)
    child2 = MCTSrNode(MCTSrState("Test problem", "Child 2 answer"), parent=root)
    root.add_child(child1)
    root.add_child(child2)

    traversed_nodes = list(mctsr._traverse_tree(root))

    assert len(traversed_nodes) == 3
    assert traversed_nodes[0] == root
    assert set(traversed_nodes[1:]) == {child1, child2}

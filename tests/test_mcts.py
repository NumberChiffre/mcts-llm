import random

import pytest

from mcts_llm.mcts import MCTS, MCTSNode


class TicTacToeState:
    def __init__(self, board: list[str] = None, player: str = "X"):
        self.board = board or [" " for _ in range(9)]
        self.player = player

    def copy(self):
        return TicTacToeState(self.board.copy(), self.player)

    def __eq__(self, other):
        return isinstance(other, TicTacToeState) and self.board == other.board and self.player == other.player


class TicTacToeMCTS(MCTS):
    def __init__(self, max_rollouts: int = 1000, c: float = 1.414):
        super().__init__(max_rollouts, c)

    def get_actions(self, S: TicTacToeState) -> list[int]:
        return [i for i, cell in enumerate(S.board) if cell == " "]

    def get_next_state(self, S: TicTacToeState, action: int) -> TicTacToeState:
        next_state = S.copy()
        next_state.board[action] = S.player
        next_state.player = "O" if S.player == "X" else "X"
        return next_state

    def is_terminal(self, S: TicTacToeState) -> bool:
        return self.get_winner(S) is not None or " " not in S.board

    def get_reward(self, S: TicTacToeState) -> float:
        winner = self.get_winner(S)
        if winner == "X":
            return 1
        elif winner == "O":
            return -1
        return 0

    def get_winner(self, S: TicTacToeState) -> str | None:
        winning_combinations = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for combo in winning_combinations:
            if S.board[combo[0]] == S.board[combo[1]] == S.board[combo[2]] != " ":
                return S.board[combo[0]]
        return None

    def _simulate_policy(self, S: TicTacToeState) -> int:
        return random.choice(self.get_actions(S))


@pytest.fixture
def empty_state():
    return TicTacToeState()


@pytest.fixture
def mcts():
    return TicTacToeMCTS(max_rollouts=1000, c=1.414)


def test_tictactoe_state_initialization(empty_state):
    assert empty_state.board == [" "] * 9
    assert empty_state.player == "X"


def test_tictactoe_state_copy(empty_state):
    copy_state = empty_state.copy()
    assert copy_state.board == empty_state.board
    assert copy_state.player == empty_state.player
    assert copy_state is not empty_state


def test_tictactoe_state_equality():
    state1 = TicTacToeState(["X", "O", " ", " ", "X", " ", " ", " ", "O"], "X")
    state2 = TicTacToeState(["X", "O", " ", " ", "X", " ", " ", " ", "O"], "X")
    state3 = TicTacToeState(["X", "O", " ", " ", "X", " ", " ", " ", "O"], "O")
    assert state1 == state2
    assert state1 != state3


def test_mcts_initialization(mcts):
    assert mcts.max_rollouts == 1000
    assert mcts.c == 1.414


def test_get_actions(mcts, empty_state):
    assert mcts.get_actions(empty_state) == list(range(9))

    state_with_moves = TicTacToeState(["X", "O", " ", " ", "X", " ", " ", " ", "O"], "X")
    assert mcts.get_actions(state_with_moves) == [2, 3, 5, 6, 7]


def test_get_next_state(mcts, empty_state):
    next_state = mcts.get_next_state(empty_state, 4)
    assert next_state.board == [" ", " ", " ", " ", "X", " ", " ", " ", " "]
    assert next_state.player == "O"


def test_is_terminal(mcts):
    non_terminal_state = TicTacToeState(["X", "O", " ", " ", "X", " ", " ", " ", "O"], "X")
    assert not mcts.is_terminal(non_terminal_state)

    terminal_state_win = TicTacToeState(["X", "X", "X", "O", "O", " ", " ", " ", " "], "O")
    assert mcts.is_terminal(terminal_state_win)

    terminal_state_draw = TicTacToeState(["X", "O", "X", "X", "O", "O", "O", "X", "X"], "X")
    assert mcts.is_terminal(terminal_state_draw)


def test_get_reward(mcts):
    x_win_state = TicTacToeState(["X", "X", "X", "O", "O", " ", " ", " ", " "], "O")
    assert mcts.get_reward(x_win_state) == 1

    o_win_state = TicTacToeState(["X", "X", "O", "X", "O", " ", "O", " ", " "], "X")
    assert mcts.get_reward(o_win_state) == -1

    draw_state = TicTacToeState(["X", "O", "X", "X", "O", "O", "O", "X", "X"], "X")
    assert mcts.get_reward(draw_state) == 0


def test_get_winner(mcts):
    x_win_state = TicTacToeState(["X", "X", "X", "O", "O", " ", " ", " ", " "], "O")
    assert mcts.get_winner(x_win_state) == "X"

    o_win_state = TicTacToeState(["X", "X", "O", "X", "O", " ", "O", " ", " "], "X")
    assert mcts.get_winner(o_win_state) == "O"

    no_winner_state = TicTacToeState(["X", "O", " ", " ", "X", " ", " ", " ", "O"], "X")
    assert mcts.get_winner(no_winner_state) is None


def test_mcts_simulate_policy(mcts, empty_state):
    action = mcts._simulate_policy(empty_state)
    assert action in mcts.get_actions(empty_state)


def test_mcts_search(mcts, empty_state):
    best_state = mcts.search(empty_state)
    assert isinstance(best_state, TicTacToeState)
    assert best_state != empty_state
    assert sum(1 for cell in best_state.board if cell != " ") == 1


def test_mcts_search_winning_move():
    state = TicTacToeState(["X", "X", " ", "O", "O", " ", " ", " ", " "], "X")
    mcts = TicTacToeMCTS(max_rollouts=1000, c=1.414)
    best_state = mcts.search(state)
    assert best_state.board[2] == "X"


def test_mcts_search_blocking_move():
    state = TicTacToeState(["O", "O", " ", "X", "X", " ", " ", " ", " "], "X")
    mcts = TicTacToeMCTS(max_rollouts=1000, c=1.414)
    best_state = mcts.search(state)
    assert best_state.board[2] == " "


def test_mcts_full_game():
    mcts = TicTacToeMCTS(max_rollouts=1000, c=1.414)
    state = TicTacToeState()

    moves = 0
    while not mcts.is_terminal(state):
        best_state = mcts.search(state)
        action = next(i for i, (a, b) in enumerate(zip(state.board, best_state.board)) if a != b)
        state = mcts.get_next_state(state, action)
        moves += 1

    assert moves <= 9
    assert mcts.is_terminal(state)


def test_mcts_initialize(mcts, empty_state):
    root = mcts.initialize(empty_state)
    assert isinstance(root, MCTSNode)
    assert root.S == empty_state
    assert root.parent is None


def test_mcts_default_uct_score(mcts):
    node = MCTSNode(TicTacToeState())
    assert mcts._uct(node) == float("inf")

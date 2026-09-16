from src.constants import Move
from src.game_machine import GameMachine
from src.player_classes import AlwaysCooperatePlayer, AlwaysDefectPlayer


def test_deterministic_payoffs():
    gm = GameMachine(error_rate=0.0)

    p1 = AlwaysCooperatePlayer()
    p2 = AlwaysCooperatePlayer()
    gm.play_game(p1, p2)
    assert p1.score == 3
    assert p2.score == 3

    p3 = AlwaysDefectPlayer()
    p4 = AlwaysCooperatePlayer()
    gm.play_game(p3, p4)
    assert p3.score == 5
    assert p4.score == 0

    p5 = AlwaysDefectPlayer()
    p6 = AlwaysDefectPlayer()
    gm.play_game(p5, p6)
    assert p5.score == 1
    assert p6.score == 1


def test_zero_noise_guarantee():
    gm = GameMachine(error_rate=0.0)
    p1 = AlwaysCooperatePlayer()
    p2 = AlwaysDefectPlayer()

    # Play 100 rounds: should NEVER flip
    for _ in range(100):
        gm.play_game(p1, p2)

    assert len(p1.game_history) == 100
    for my_move, opp_move in p1.game_history:
        assert my_move == Move.COOPERATE
        assert opp_move == Move.DEFECT

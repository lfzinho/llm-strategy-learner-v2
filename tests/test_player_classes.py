from src.constants import Move
from src.player_classes import (
    AlwaysCooperatePlayer,
    AlwaysDefectPlayer,
    TitForTatPlayer,
    SuspiciousTitForTatPlayer,
    TitForTwoTatsPlayer,
    TatForTitPlayer,
    DetectivePlayer,
    GrimTriggerPlayer,
    PavlovPlayer,
    get_strategy_class,
)
from src.game_machine import GameMachine


def test_tit_for_tat():
    gm = GameMachine(error_rate=0.0)
    tft = TitForTatPlayer()
    ad = AlwaysDefectPlayer()

    # Round 1: TFT cooperates, AD defects
    gm.play_game(tft, ad)
    assert tft.game_history[-1] == (Move.COOPERATE, Move.DEFECT)

    # Round 2: TFT defects in response
    gm.play_game(tft, ad)
    assert tft.game_history[-1] == (Move.DEFECT, Move.DEFECT)


def test_suspicious_tit_for_tat():
    gm = GameMachine(error_rate=0.0)
    stft = SuspiciousTitForTatPlayer()
    ac = AlwaysCooperatePlayer()

    # Round 1: STFT starts with DEFECT
    gm.play_game(stft, ac)
    assert stft.game_history[-1] == (Move.DEFECT, Move.COOPERATE)

    # Round 2: STFT copies Cooperate
    gm.play_game(stft, ac)
    assert stft.game_history[-1] == (Move.COOPERATE, Move.COOPERATE)


def test_tit_for_two_tats():
    gm = GameMachine(error_rate=0.0)
    t2t = TitForTwoTatsPlayer()
    ad = AlwaysDefectPlayer()

    # Round 1: T2T cooperates
    gm.play_game(t2t, ad)
    assert t2t.game_history[-1] == (Move.COOPERATE, Move.DEFECT)

    # Round 2: T2T cooperates (only 1 defection so far)
    gm.play_game(t2t, ad)
    assert t2t.game_history[-1] == (Move.COOPERATE, Move.DEFECT)

    # Round 3: T2T defects (2 consecutive defections)
    gm.play_game(t2t, ad)
    assert t2t.game_history[-1] == (Move.DEFECT, Move.DEFECT)


def test_detective_exploit_and_retaliate():
    gm = GameMachine(error_rate=0.0)
    
    # Case 1: Opponent never retaliates -> Detective exploits
    det1 = DetectivePlayer()
    ac = AlwaysCooperatePlayer()
    for _ in range(5):
        gm.play_game(det1, ac)
    # Round 5 should be DEFECT
    assert det1.game_history[-1][0] == Move.DEFECT

    # Case 2: Opponent retaliates against round 2 defect -> Detective switches to TFT
    det2 = DetectivePlayer()
    tft = TitForTatPlayer()
    for _ in range(5):
        gm.play_game(det2, tft)
    # Detective switches to TFT after noticing defection
    assert len(det2.game_history) == 5


def test_strategy_factory():
    cls = get_strategy_class("tit_for_tat")
    assert cls == TitForTatPlayer
    cls = get_strategy_class("always defect")
    assert cls == AlwaysDefectPlayer

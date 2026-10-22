from src.memory import MemoryStream, similarity


def test_near_duplicate_reflections_are_merged_and_gain_importance():
    m = MemoryStream()
    first = m.add("reflection", "Always defect against a player that never retaliates.", importance=4)
    again = m.add("reflection", "always defect against a player that never retaliates", importance=8)
    assert first is not None and again is None
    assert len(m.entries) == 1 and m.entries[0].importance == 8


def test_forty_repeated_lessons_do_not_grow_the_memory():
    m = MemoryStream()
    for i in range(40):
        m.add("reflection", f"Defect immediately to enforce the equilibrium (variation {i % 2})")
    assert len(m.entries) <= 2


def test_same_text_different_kind_is_kept():
    m = MemoryStream()
    m.add("observation", "They copied my last move.")
    assert m.add("reflection", "They copied my last move.") is not None


def test_empty_text_is_ignored_and_importance_is_clamped():
    m = MemoryStream()
    assert m.add("reflection", "   ") is None
    assert m.add("reflection", "x one", importance=99).importance == 10
    assert m.add("reflection", "y two", importance=-5).importance == 1


def test_retrieval_prefers_important_recent_and_relevant():
    m = MemoryStream()
    m.add("reflection", "minor thing about cats", importance=1)
    m.advance()
    m.add("reflection", "important rule about retaliation after beta", importance=9)
    for _ in range(3):
        m.advance()
    m.add("reflection", "recent plain note", importance=3)

    texts = [e.text for e in m.retrieve("opponent retaliation after beta", k=2)]
    assert texts[0].startswith("important rule")
    assert "minor thing about cats" not in texts


def test_retrieval_is_capped_and_recency_decays():
    m = MemoryStream(recency_decay=0.5)
    for i in range(20):
        m.add("reflection", f"unique lesson number{i} word{i}", importance=5)
        m.advance()
    top = m.retrieve("", k=5)
    assert len(top) == 5
    assert top[0].text.startswith("unique lesson number19")


def test_opponent_tag_only_matters_when_passed():
    m = MemoryStream()
    a = m.add("reflection", "alpha lesson", importance=5, opponent="TitForTat")
    b = m.add("reflection", "beta lesson", importance=5, opponent="AlwaysDefect")
    assert m.retrieve("", k=1, opponent="TitForTat")[0].id == a.id
    assert m.retrieve("", k=1, opponent="AlwaysDefect")[0].id == b.id


def test_serialisation_roundtrip():
    m = MemoryStream()
    m.add("observation", "obs", opponent="Random")
    m.add("reflection", "refl", importance=7, evidence=[1])
    m.advance()
    clone = MemoryStream()
    clone.load_list(m.to_list(), m.step)
    assert clone.to_list() == m.to_list() and clone.step == 1


def test_similarity_bounds():
    assert similarity("a b c", "a b c") == 1.0
    assert similarity("alpha", "beta") == 0.0
    assert similarity("", "x") == 0.0

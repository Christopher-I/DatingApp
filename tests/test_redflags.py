from copilot.brain.embeddings import HashEmbedder
from copilot.brain.engine import Engine
from copilot.brain.matcher import CentroidClassifier, Matcher
from copilot.brain.redflags import RedFlagFilter
from copilot.config import Config, RedFlagConfig
from copilot.drivers.base import Profile


def _profile(**kw):
    base = dict(external_ref="p", photos=[b"x"], age=29, bio_text="", fields={})
    base.update(kw)
    return Profile(**base)


def test_keyword_phrase_is_a_hard_pass():
    emb = HashEmbedder(dim=32)
    cfg = RedFlagConfig(keyword_phrases=("no hookups",), image_prompts=())
    rf = RedFlagFilter(emb, cfg)
    result = rf.check(_profile(bio_text="Looking for something serious, no hookups."))
    assert result.hit is True
    assert any("no hookups" in r for r in result.reasons)


def test_clean_bio_no_flag():
    emb = HashEmbedder(dim=32)
    cfg = RedFlagConfig(keyword_phrases=("no hookups",), image_prompts=())
    rf = RedFlagFilter(emb, cfg)
    result = rf.check(_profile(bio_text="Love hiking and dogs."))
    assert result.hit is False


def test_bio_similarity_flags_matching_theme():
    emb = HashEmbedder(dim=64)
    phrase = "anxious attachment in therapy"
    # A very low threshold guarantees the hash-similarity check trips for the
    # identical sentence; this exercises the embedding-similarity path.
    cfg = RedFlagConfig(
        example_phrases=(phrase,),
        bio_similarity_threshold=0.99,
        image_prompts=(),
    )
    rf = RedFlagFilter(emb, cfg)
    result = rf.check(_profile(bio_text=phrase + "."))
    assert result.hit is True


def test_engine_structured_prefilter_passes_out_of_range_age():
    emb = HashEmbedder(dim=16)
    cfg = Config(redflags=RedFlagConfig(image_prompts=()))
    matcher = Matcher(emb, CentroidClassifier(), cfg.match)
    rf = RedFlagFilter(emb, cfg.redflags)
    engine = Engine(emb, matcher, rf, cfg)
    decision = engine.evaluate(_profile(age=45))
    assert decision.action.value == "pass"
    assert "age" in decision.reason


def test_engine_redflag_overrides_score():
    emb = HashEmbedder(dim=16)
    cfg = Config(redflags=RedFlagConfig(keyword_phrases=("red flag",), image_prompts=()))
    matcher = Matcher(emb, CentroidClassifier(), cfg.match)
    rf = RedFlagFilter(emb, cfg.redflags)
    engine = Engine(emb, matcher, rf, cfg)
    decision = engine.evaluate(_profile(age=29, bio_text="this is a red flag bio"))
    assert decision.action.value == "pass"
    assert decision.red_flags

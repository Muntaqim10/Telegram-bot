"""The bot must not display confidence it does not have.

Two independent ways the conviction score becomes meaningless: the strategy never
computed the model's features (so it is scoring hardcoded defaults), or the model itself
cannot separate inputs. Either must read UNSCORED rather than a tier and a percentage.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from src.ai.xgb_micro_v2 import MIN_USEFUL_SPREAD, PROBE_GRID, XGBMicroSentinelV2
from src.execution.signal_pipeline import SignalPipeline
from src.models.signal import TradeSignal
from src.strategy.donchian_daily import DonchianSwingStrategy
from src.utils.telegram_formatter import format_telegram_alert


@pytest.fixture(scope="module")
def model():
    m = XGBMicroSentinelV2()
    if not m.is_active:
        pytest.skip("no trained model on disk")
    return m


def synthetic_bars(seed, trend, n=80):
    rng = np.random.default_rng(seed)
    close = np.maximum(100 + np.cumsum(rng.normal(trend, 1.5, n)), 1.0)
    return pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99,
                         "close": close, "volume": rng.integers(10**6, 5 * 10**6, n)})


class TestDiscriminationProbe:
    def test_the_probe_sweeps_a_real_grid(self):
        points = 1
        for values in PROBE_GRID.values():
            points *= len(values)
        assert points >= 100, "a one-point probe would prove nothing"

    def test_it_records_a_spread_and_agrees_with_it(self, model):
        assert model.prediction_spread > 0.0
        assert model.is_discriminating == (model.prediction_spread >= MIN_USEFUL_SPREAD)

    def test_a_collapsed_model_says_so(self, model):
        real_flag, real_spread = model.is_discriminating, model.prediction_spread
        try:
            model.is_discriminating, model.prediction_spread = False, 0.0001
            r = model.validate_setup({"sma_spread": 0.08, "sma20_ratio": 1.3,
                                      "rsi_14": 77.0, "direction_code": 1})
            assert r["verdict"] == "NO_DISCRIMINATION"
            assert "prediction_spread" in r
        finally:
            model.is_discriminating, model.prediction_spread = real_flag, real_spread

    def test_a_working_model_returns_a_real_verdict(self, model):
        r = model.validate_setup({"sma_spread": 0.08, "sma20_ratio": 1.3,
                                  "rsi_14": 77.0, "direction_code": 1})
        assert r["verdict"] in ("CONCORDANT", "AMBIGUOUS", "HALLUCINATION")


class TestAlertText:
    BASE = dict(
        ticker="PLTR", price=150.0, signal_direction="Long", strategy_type="ORB Breakout",
        timestamp="2026-09-03 10:00:00", win_probability=0.3654, xgb_win_prob=0.3654,
        sentinel_verdict="UNSCORED_NO_FEATURES", context_score="Intraday ORB",
        catalyst="Test", historical_edge="n/a", stop_loss=145.0, take_profit=160.0,
    )

    def test_unscored_shows_no_percentage(self):
        msg = format_telegram_alert(TradeSignal(conviction="⚪ UNSCORED", **self.BASE))
        assert "UNSCORED" in msg
        assert "not discriminating" in msg or "no information" in msg
        assert "36%" not in msg and "37%" not in msg, \
            "a percentage leaked into an unscored alert"

    def test_a_scored_alert_still_shows_its_number(self):
        msg = format_telegram_alert(
            TradeSignal(conviction="🟢 HIGH", **{**self.BASE, "win_probability": 0.42}))
        assert "HIGH" in msg
        assert "42%" in msg


class TestPipelineGating:
    @pytest.fixture(scope="class")
    def source(self):
        return inspect.getsource(SignalPipeline.process_signal)

    def test_it_checks_for_missing_features(self, source):
        assert "missing = [" in source

    def test_it_no_longer_scores_hardcoded_defaults(self, source):
        assert 'signal.get("sma_spread", 0.02)' not in source, \
            "the 0.02/55.0 fallbacks gave every alert the same two scores"

    def test_conviction_requires_supplied_features(self, source):
        assert "features_supplied and" in source

    def test_the_win_prob_gate_is_skipped_when_unscored(self, source):
        assert "model_scores and (win_prob" in source

    def test_it_borrows_cached_daily_features_before_giving_up(self, source):
        assert "feature_cache" in source
        assert source.index("feature_cache") < source.index("UNSCORED_NO_FEATURES")


class TestDailyFeatures:
    def test_the_extractor_returns_the_three_model_features(self):
        feats = DonchianSwingStrategy.extract_model_features(
            DonchianSwingStrategy.compute_indicators(synthetic_bars(1, 0.9)))
        assert set(feats) == {"sma20_ratio", "sma_spread", "rsi_14"}

    def test_it_refuses_an_unusable_frame(self):
        assert DonchianSwingStrategy.extract_model_features(
            pd.DataFrame({"close": [1.0]})) is None

    def test_measured_features_move_the_score(self, model):
        if not model.is_discriminating:
            pytest.skip("model is not discriminating")
        probs = []
        for seed, trend in ((1, 0.9), (2, 0.3), (3, 0.0), (4, -0.3), (5, -0.9)):
            f = DonchianSwingStrategy.extract_model_features(
                DonchianSwingStrategy.compute_indicators(synthetic_bars(seed, trend)))
            if f:
                probs.append(model.validate_setup({**f, "direction_code": 1})["win_prob"])

        assert max(probs) - min(probs) >= MIN_USEFUL_SPREAD
        assert probs[-1] < probs[0], "a hard downtrend must score below a strong uptrend"

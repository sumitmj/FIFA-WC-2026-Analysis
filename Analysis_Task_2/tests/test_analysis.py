import math
import unittest
from pathlib import Path

import pandas as pd

from src.acquire_fbref import (
    PLAYER_FIELDS,
    parse_tables,
    validate_frozen_extract,
)
from src.analyse_midfield import (
    STAGES,
    build_data_quality_summary,
    draw_stratified_sample,
    load_and_prepare,
    welch_test,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/fbref_2026_midfield_population.csv"


class AcquisitionTest(unittest.TestCase):
    def test_bundled_extract_passes_acquisition_validation(self):
        population = validate_frozen_extract(DATA)
        self.assertEqual(len(population), 240)
        self.assertEqual(population["team"].nunique(), 48)

    def test_commented_tables_and_opponent_table_are_parsed(self):
        player = pd.DataFrame(
            [["Test Player", "MF", "au Australia", "25-100", 2.0, 3, 4, 5, 6]],
            columns=PLAYER_FIELDS,
        )
        squad = pd.DataFrame({"Squad": ["au Australia"], "90s": [4.0]})
        opponent = pd.DataFrame({"Squad": ["vs Australia"], "90s": [4.0]})
        html = (
            "<html><!--"
            + opponent.to_html(index=False)
            + squad.to_html(index=False)
            + player.to_html(index=False)
            + "--></html>"
        )

        parsed_players, parsed_squads = parse_tables(html)

        self.assertEqual(parsed_players.loc[0, "Player"], "Test Player")
        self.assertEqual(parsed_squads.loc[0, "Squad"], "au Australia")


class DatasetDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.population = load_and_prepare(DATA)

    def test_population_rules_and_expected_coverage(self):
        self.assertEqual(len(self.population), 240)
        self.assertEqual(self.population["team"].nunique(), 48)
        self.assertEqual(
            self.population["stage"].value_counts().to_dict(),
            {"Knockout stage": 180, "Group-stage exit": 60},
        )
        self.assertTrue(self.population["position"].str.contains("MF", regex=False).all())
        self.assertGreaterEqual(self.population["minutes"].min(), 180)
        self.assertTrue(self.population["tackles_won_per90"].map(math.isfinite).all())

    def test_all_data_quality_checks_pass(self):
        checks = build_data_quality_summary(self.population)
        self.assertTrue(checks["passed"].all(), checks.to_string(index=False))

    def test_seeded_sample_is_reproducible_and_balanced(self):
        first = draw_stratified_sample(self.population, seed=2026, sample_per_stratum=50)
        second = draw_stratified_sample(self.population, seed=2026, sample_per_stratum=50)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first["stage"].value_counts().to_dict(), dict.fromkeys(STAGES, 50))
        expected_weights = {"Group-stage exit": 1.2, "Knockout stage": 3.6}
        for stage, expected in expected_weights.items():
            actual = first.loc[first["stage"] == stage, "sample_weight"].unique()
            self.assertEqual(len(actual), 1)
            self.assertAlmostEqual(float(actual[0]), expected, places=12)


class StatisticalResultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        population = load_and_prepare(DATA)
        cls.sample = draw_stratified_sample(population, seed=2026, sample_per_stratum=50)

    def test_primary_welch_result_is_stable(self):
        knockout = self.sample.loc[
            self.sample["stage"] == "Knockout stage", "tackles_won_per90"
        ]
        exits = self.sample.loc[
            self.sample["stage"] == "Group-stage exit", "tackles_won_per90"
        ]
        result = welch_test(knockout, exits)
        self.assertAlmostEqual(result["difference"], 0.1369821157195823, places=12)
        self.assertAlmostEqual(result["ci_low"], -0.1385057102955733, places=12)
        self.assertAlmostEqual(result["ci_high"], 0.4124699417347379, places=12)
        self.assertAlmostEqual(result["p_value_two_sided"], 0.32617486645384614, places=12)
        self.assertGreater(result["p_value_two_sided"], 0.05)
        self.assertLess(result["ci_low"], 0)
        self.assertGreater(result["ci_high"], 0)


if __name__ == "__main__":
    unittest.main()

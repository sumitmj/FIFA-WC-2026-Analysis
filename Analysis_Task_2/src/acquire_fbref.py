#!/usr/bin/env python3
"""Semi-automated FBref acquisition for the Objective 1 midfield population.

The preferred input is a locally saved copy of the FBref 2026 World Cup
miscellaneous-statistics page because FBref may reject automated requests.
The parser also supports a live URL and fails without overwriting the frozen
project extract when the page structure or expected coverage changes.
"""

from __future__ import annotations

import argparse
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests


DEFAULT_URL = "https://fbref.com/en/comps/1/misc/World-Cup-Stats"
PLAYER_FIELDS = ["Player", "Pos", "Squad", "Age", "90s", "Fls", "Fld", "Int", "TklW"]
OUTPUT_FIELDS = [
    "player", "position", "team", "age", "minutes_90s", "fouls", "fouled",
    "interceptions", "tackles_won", "stage", "team_90s",
]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_PATH = PROJECT_ROOT / "data/raw/fbref_2026_midfield_population.csv"
EXPECTED_STAGE_COUNTS = {"Knockout stage": 180, "Group-stage exit": 60}

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
    "Referer": "https://fbref.com/en/comps/1/World-Cup-Stats",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
    "Connection": "keep-alive",
}
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

BLOCKED_MESSAGE = """FBref did not serve the page to this script ({detail}).

A live refresh is best-effort: FBref refuses most automated requests with
HTTP 403, and some networks block the host outright. Either way this is an
access limitation rather than a defect in the project code, and the frozen
extract in data/raw/ has not been touched.

Reliable refresh route (documented in data/raw/SOURCE.md):
  1. Open {url} in a browser.
  2. File > Save Page As... and choose "Page Source" / "Web Page, HTML Only".
  3. Re-run against the saved file:
       python src/acquire_fbref.py --html /path/to/saved_fbref_page.html

If a live attempt is required, wait a few minutes between tries: FBref
throttles repeated automated requests from the same address."""


class AcquisitionError(RuntimeError):
    """The FBref page could not be retrieved automatically."""


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the informative leaf label from FBref's grouped table headings."""
    result = frame.copy()
    labels = []
    for column in result.columns:
        parts = column if isinstance(column, tuple) else (column,)
        useful = [str(part).strip() for part in parts if not str(part).startswith("Unnamed")]
        labels.append(useful[-1] if useful else str(parts[-1]).strip())
    result.columns = labels
    return result.loc[:, ~result.columns.duplicated()].copy()


def download_html(url: str, attempts: int = 3, pause: float = 5.0) -> str:
    """Fetch the live page, retrying throttled responses before giving up.

    Raises AcquisitionError with the saved-page instructions rather than a
    bare HTTPError, because a refusal here is the expected provider behaviour.
    """
    detail = "no response received"
    with requests.Session() as session:
        session.headers.update(BROWSER_HEADERS)
        for attempt in range(1, attempts + 1):
            try:
                response = session.get(url, timeout=30)
            except requests.RequestException as error:
                detail = f"{type(error).__name__}: {error}"
            else:
                if response.ok:
                    return response.text
                detail = f"HTTP {response.status_code} {response.reason}"
                if response.status_code not in RETRYABLE_STATUS:
                    break
            if attempt < attempts:
                time.sleep(pause * attempt)
    raise AcquisitionError(BLOCKED_MESSAGE.format(detail=detail, url=url))


def parse_tables(html: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # FBref sometimes wraps tables in HTML comments for browser-side display.
    visible_html = html.replace("<!--", "").replace("-->", "")
    tables = [flatten_columns(table) for table in pd.read_html(StringIO(visible_html))]
    player_candidates = [
        table for table in tables if set(PLAYER_FIELDS).issubset(set(table.columns))
    ]
    squad_candidates = [
        table
        for table in tables
        if "Squad" in table.columns
        and "90s" in table.columns
        and "Player" not in table.columns
        and not is_opponent_table(table)
    ]
    if not player_candidates or not squad_candidates:
        available = [sorted(set(table.columns))[:12] for table in tables]
        raise ValueError(f"Required FBref tables were not found. Parsed headings: {available}")
    player_table = max(player_candidates, key=len)
    squad_table = max(squad_candidates, key=lambda table: clean_team(table["Squad"]).nunique())
    return player_table, squad_table


def clean_team(series: pd.Series) -> pd.Series:
    """Drop FBref's lowercase country-code prefix (2 or 3 letters, e.g. "eng")."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"^[a-z]{2,3}\s+", "", regex=True)
    )


def is_opponent_table(table: pd.DataFrame) -> bool:
    """FBref pairs each squad table with an identically shaped opponent table.

    The opponent rows are labelled "vs <team>" and must never be used as the
    squad playing-time source, otherwise every join silently fails.
    """
    labels = table["Squad"].astype(str).str.strip()
    return bool(labels.str.startswith("vs ").mean() > 0.5)


def build_eligible_population(
    player_table: pd.DataFrame, squad_table: pd.DataFrame
) -> pd.DataFrame:
    players = player_table[PLAYER_FIELDS].copy()
    players = players.loc[players["Player"].astype(str) != "Player"].copy()
    players["Squad"] = clean_team(players["Squad"])
    players["Age"] = players["Age"].astype(str).str.extract(r"(\d+)", expand=False)
    for field in ["Age", "90s", "Fls", "Fld", "Int", "TklW"]:
        players[field] = pd.to_numeric(players[field], errors="coerce")

    squads = squad_table[["Squad", "90s"]].copy()
    squads = squads.loc[squads["Squad"].astype(str) != "Squad"].copy()
    squads["Squad"] = clean_team(squads["Squad"])
    squads["90s"] = pd.to_numeric(squads["90s"], errors="coerce")
    squads = squads.dropna().drop_duplicates("Squad").rename(columns={"90s": "team_90s"})

    eligible = players.loc[
        players["Pos"].astype(str).str.contains("MF", regex=False)
        & players["90s"].ge(2.0)
    ].copy()
    eligible = eligible.merge(squads, on="Squad", how="left", validate="many_to_one")
    if eligible["team_90s"].isna().any():
        missing = sorted(eligible.loc[eligible["team_90s"].isna(), "Squad"].unique())
        raise ValueError(f"Squad playing time could not be joined for: {missing}")
    eligible["stage"] = np.where(
        eligible["team_90s"] > 3.0, "Knockout stage", "Group-stage exit"
    )
    eligible = eligible.rename(
        columns={
            "Player": "player",
            "Pos": "position",
            "Squad": "team",
            "Age": "age",
            "90s": "minutes_90s",
            "Fls": "fouls",
            "Fld": "fouled",
            "Int": "interceptions",
            "TklW": "tackles_won",
        }
    )
    eligible = eligible[OUTPUT_FIELDS].sort_values(
        ["stage", "player", "team"]
    ).reset_index(drop=True)
    validate_population(eligible, "Refreshed extract")
    return eligible


def validate_population(population: pd.DataFrame, label: str) -> None:
    missing = sorted(set(OUTPUT_FIELDS) - set(population.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")

    stage_counts = population["stage"].value_counts().to_dict()
    problems = []
    if len(population) != 240:
        problems.append(f"rows={len(population)} (expected 240)")
    if population["team"].nunique() != 48:
        problems.append(f"teams={population['team'].nunique()} (expected 48)")
    if stage_counts != EXPECTED_STAGE_COUNTS:
        problems.append(f"stages={stage_counts} (expected {EXPECTED_STAGE_COUNTS})")
    if not population["position"].astype(str).str.contains("MF", regex=False).all():
        problems.append("non-midfield position present")
    if pd.to_numeric(population["minutes_90s"], errors="coerce").lt(2.0).any():
        problems.append("player below the 2.0 90s eligibility threshold")
    if population.duplicated(["player", "team"]).any():
        problems.append("duplicate player-team rows present")
    if population[OUTPUT_FIELDS].isna().any().any():
        problems.append("missing values present")
    if problems:
        raise ValueError(f"{label} failed validation: {'; '.join(problems)}")


def validate_frozen_extract(path: Path = FROZEN_PATH) -> pd.DataFrame:
    if not path.is_file():
        raise AcquisitionError(f"Bundled FBref extract not found: {path}")
    population = pd.read_csv(path)
    validate_population(population, "Bundled FBref extract")
    return population


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--html", type=Path, help="Saved FBref page (recommended)")
    source.add_argument(
        "--url",
        nargs="?",
        const=DEFAULT_URL,
        metavar="URL",
        help="Attempt a live refresh; optionally provide a different FBref URL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/fbref_2026_midfield_population_refresh.csv"),
        help="New output path; intentionally does not overwrite the frozen extract",
    )
    return parser.parse_args()


def read_source_html(args: argparse.Namespace) -> str:
    if args.html is None:
        return download_html(args.url)
    if not args.html.is_file():
        raise AcquisitionError(f"Saved FBref page not found: {args.html}")
    return args.html.read_text(encoding="utf-8", errors="replace")


def main() -> None:
    args = parse_args()
    try:
        if args.html is None and args.url is None:
            population = validate_frozen_extract()
            print(
                f"Validated bundled FBref extract: {len(population)} eligible "
                f"midfielders across {population['team'].nunique()} teams"
            )
            print("To refresh it, use --html SAVED_PAGE.html (recommended) or --url.")
            return
        html = read_source_html(args)
        players, squads = parse_tables(html)
        eligible = build_eligible_population(players, squads)
    except (AcquisitionError, ValueError) as error:
        raise SystemExit(f"Refresh aborted. {error}") from None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(args.output, index=False)
    print(f"Saved {len(eligible)} eligible midfielders across {eligible['team'].nunique()} teams")
    print(f"Review before replacing the frozen extract: {args.output.resolve()}")


if __name__ == "__main__":
    main()

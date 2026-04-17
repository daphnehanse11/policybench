"""Tests for scenario generation."""

import json

import pandas as pd
import pytest

from policybench.scenarios import (
    SUPPORTED_FILING_STATUSES,
    generate_scenarios,
    load_excluded_household_ids,
    load_scenarios_from_manifest,
    scenario_manifest,
    scenarios_from_cps_frame,
    scenarios_from_uk_frames,
)


@pytest.fixture
def sample_person_frame():
    return pd.DataFrame(
        [
            {
                "person_id": 1,
                "household_id": 101,
                "tax_unit_id": 201,
                "spm_unit_id": 301,
                "family_id": 401,
                "marital_unit_id": 501,
                "household_weight": 2.0,
                "state_code": "CA",
                "filing_status": "HEAD_OF_HOUSEHOLD",
                "age": 35,
                "employment_income": 30_000.0,
                "real_estate_taxes": 4_000.0,
                "home_mortgage_interest": 9_000.0,
                "health_savings_account_ald": 800.0,
                "spm_unit_pre_subsidy_childcare_expenses": 2_400.0,
                "auto_loan_interest": 300.0,
                "auto_loan_balance": 5_000.0,
                "is_tax_unit_head": True,
            },
            {
                "person_id": 2,
                "household_id": 101,
                "tax_unit_id": 201,
                "spm_unit_id": 301,
                "family_id": 401,
                "marital_unit_id": 501,
                "household_weight": 2.0,
                "state_code": "CA",
                "filing_status": "HEAD_OF_HOUSEHOLD",
                "age": 8,
                "employment_income": 0.0,
            },
            {
                "person_id": 3,
                "household_id": 102,
                "tax_unit_id": 202,
                "spm_unit_id": 302,
                "family_id": 402,
                "marital_unit_id": 502,
                "household_weight": 3.0,
                "state_code": "TX",
                "filing_status": "JOINT",
                "age": 40,
                "employment_income": 70_000.0,
                "is_tax_unit_head": True,
            },
            {
                "person_id": 4,
                "household_id": 102,
                "tax_unit_id": 202,
                "spm_unit_id": 302,
                "family_id": 402,
                "marital_unit_id": 502,
                "household_weight": 3.0,
                "state_code": "TX",
                "filing_status": "JOINT",
                "age": 38,
                "employment_income": 30_000.0,
                "self_employment_income": 5_000.0,
                "is_tax_unit_spouse": True,
            },
            {
                "person_id": 5,
                "household_id": 102,
                "tax_unit_id": 202,
                "spm_unit_id": 302,
                "family_id": 402,
                "marital_unit_id": 502,
                "household_weight": 3.0,
                "state_code": "TX",
                "filing_status": "JOINT",
                "age": 10,
                "employment_income": 0.0,
            },
            {
                "person_id": 6,
                "household_id": 102,
                "tax_unit_id": 202,
                "spm_unit_id": 302,
                "family_id": 402,
                "marital_unit_id": 502,
                "household_weight": 3.0,
                "state_code": "TX",
                "filing_status": "JOINT",
                "age": 5,
                "employment_income": 0.0,
            },
            {
                "person_id": 7,
                "household_id": 103,
                "tax_unit_id": 203,
                "spm_unit_id": 303,
                "family_id": 403,
                "marital_unit_id": 503,
                "household_weight": 1.0,
                "state_code": "NY",
                "filing_status": "SINGLE",
                "age": 67,
                "employment_income": 0.0,
                "social_security_retirement": 22_000.0,
                "is_tax_unit_head": True,
            },
            {
                "person_id": 8,
                "household_id": 103,
                "tax_unit_id": 203,
                "spm_unit_id": 303,
                "family_id": 403,
                "marital_unit_id": 504,
                "household_weight": 1.0,
                "state_code": "NY",
                "filing_status": "SINGLE",
                "age": 19,
                "employment_income": 0.0,
                "is_full_time_college_student": True,
            },
            {
                "person_id": 9,
                "household_id": 104,
                "tax_unit_id": 204,
                "spm_unit_id": 304,
                "family_id": 404,
                "marital_unit_id": 505,
                "household_weight": 4.0,
                "state_code": "FL",
                "filing_status": "SINGLE",
                "age": 50,
                "employment_income": 45_000.0,
                "is_tax_unit_head": True,
            },
            {
                "person_id": 10,
                "household_id": 104,
                "tax_unit_id": 205,
                "spm_unit_id": 304,
                "family_id": 404,
                "marital_unit_id": 505,
                "household_weight": 4.0,
                "state_code": "FL",
                "filing_status": "SINGLE",
                "age": 22,
                "employment_income": 12_000.0,
            },
            {
                "person_id": 11,
                "household_id": 105,
                "tax_unit_id": 206,
                "spm_unit_id": 306,
                "family_id": 406,
                "marital_unit_id": 506,
                "household_weight": 5.0,
                "state_code": "CO",
                "filing_status": "SINGLE",
                "age": 52,
                "employment_income": 0.0,
                "disability_benefits": 18_000.0,
                "is_disabled": True,
                "is_tax_unit_head": True,
            },
            {
                "person_id": 12,
                "household_id": 106,
                "tax_unit_id": 207,
                "spm_unit_id": 307,
                "family_id": 407,
                "marital_unit_id": 507,
                "household_weight": 6.0,
                "state_code": "AZ",
                "filing_status": "HEAD_OF_HOUSEHOLD",
                "age": 45,
                "employment_income": 38_000.0,
                "is_tax_unit_head": True,
            },
            {
                "person_id": 13,
                "household_id": 106,
                "tax_unit_id": 207,
                "spm_unit_id": 307,
                "family_id": 407,
                "marital_unit_id": 507,
                "household_weight": 6.0,
                "state_code": "AZ",
                "filing_status": "HEAD_OF_HOUSEHOLD",
                "age": 16,
                "employment_income": 0.0,
            },
            {
                "person_id": 14,
                "household_id": 106,
                "tax_unit_id": 207,
                "spm_unit_id": 307,
                "family_id": 407,
                "marital_unit_id": 507,
                "household_weight": 6.0,
                "state_code": "AZ",
                "filing_status": "HEAD_OF_HOUSEHOLD",
                "age": 14,
                "employment_income": 0.0,
            },
            {
                "person_id": 15,
                "household_id": 106,
                "tax_unit_id": 207,
                "spm_unit_id": 307,
                "family_id": 407,
                "marital_unit_id": 507,
                "household_weight": 6.0,
                "state_code": "AZ",
                "filing_status": "HEAD_OF_HOUSEHOLD",
                "age": 12,
                "employment_income": 0.0,
            },
            {
                "person_id": 16,
                "household_id": 106,
                "tax_unit_id": 207,
                "spm_unit_id": 307,
                "family_id": 407,
                "marital_unit_id": 507,
                "household_weight": 6.0,
                "state_code": "AZ",
                "filing_status": "HEAD_OF_HOUSEHOLD",
                "age": 10,
                "employment_income": 0.0,
            },
            {
                "person_id": 17,
                "household_id": 106,
                "tax_unit_id": 207,
                "spm_unit_id": 307,
                "family_id": 407,
                "marital_unit_id": 507,
                "household_weight": 6.0,
                "state_code": "AZ",
                "filing_status": "HEAD_OF_HOUSEHOLD",
                "age": 8,
                "employment_income": 0.0,
            },
            {
                "person_id": 18,
                "household_id": 106,
                "tax_unit_id": 207,
                "spm_unit_id": 307,
                "family_id": 407,
                "marital_unit_id": 507,
                "household_weight": 6.0,
                "state_code": "AZ",
                "filing_status": "HEAD_OF_HOUSEHOLD",
                "age": 6,
                "employment_income": 0.0,
            },
        ]
    )


@pytest.fixture
def sample_uk_frames():
    person_df = pd.DataFrame(
        [
            {
                "person_id": 1,
                "person_household_id": 1001,
                "person_benunit_id": 2001,
                "age": 42,
                "gender": "FEMALE",
                "marital_status": "MARRIED",
                "employment_income": 42_000.0,
                "self_employment_income": 0.0,
                "savings_interest_income": 120.0,
                "dividend_income": 50.0,
                "private_pension_income": 0.0,
                "state_pension_reported": 0.0,
                "capital_gains_before_response": 0.0,
                "property_income": 0.0,
                "miscellaneous_income": 0.0,
                "employment_expenses": 150.0,
                "private_pension_contributions": 600.0,
                "gift_aid": 75.0,
                "blind_persons_allowance": 0.0,
                "is_disabled_for_benefits": False,
                "pip_dl_category": "NONE",
                "pip_m_category": "NONE",
                "hours_worked": 37.5,
                "is_disabled": False,
                "is_student": False,
            },
            {
                "person_id": 2,
                "person_household_id": 1001,
                "person_benunit_id": 2001,
                "age": 12,
                "gender": "MALE",
                "marital_status": "SINGLE",
                "employment_income": 0.0,
                "self_employment_income": 0.0,
                "savings_interest_income": 0.0,
                "dividend_income": 0.0,
                "private_pension_income": 0.0,
                "state_pension_reported": 0.0,
                "capital_gains_before_response": 0.0,
                "property_income": 0.0,
                "miscellaneous_income": 0.0,
                "employment_expenses": 0.0,
                "private_pension_contributions": 0.0,
                "gift_aid": 0.0,
                "blind_persons_allowance": 0.0,
                "is_disabled_for_benefits": False,
                "pip_dl_category": "NONE",
                "pip_m_category": "NONE",
                "hours_worked": 0.0,
                "is_disabled": False,
                "is_student": True,
            },
            {
                "person_id": 3,
                "person_household_id": 1002,
                "person_benunit_id": 2002,
                "age": 72,
                "gender": "MALE",
                "marital_status": "SINGLE",
                "employment_income": 0.0,
                "self_employment_income": 0.0,
                "savings_interest_income": 40.0,
                "dividend_income": 0.0,
                "private_pension_income": 8_500.0,
                "state_pension_reported": 11_000.0,
                "capital_gains_before_response": 0.0,
                "property_income": 0.0,
                "miscellaneous_income": 0.0,
                "employment_expenses": 0.0,
                "private_pension_contributions": 0.0,
                "gift_aid": 0.0,
                "blind_persons_allowance": 0.0,
                "is_disabled_for_benefits": True,
                "pip_dl_category": "STANDARD",
                "pip_m_category": "NONE",
                "hours_worked": 0.0,
                "is_disabled": True,
                "is_student": False,
            },
        ]
    )
    household_df = pd.DataFrame(
        [
            {
                "household_id": 1001,
                "household_weight": 2.5,
                "region": "LONDON",
                "tenure_type": "RENT_PRIVATELY",
                "council_tax": 1_800.0,
                "rent": 14_400.0,
                "mortgage_interest_repayment": 0.0,
                "mortgage_capital_repayment": 0.0,
                "savings": 2_500.0,
                "household_wealth": 22_000.0,
                "num_vehicles": 1.0,
                "council_tax_band": "C",
            },
            {
                "household_id": 1002,
                "household_weight": 1.5,
                "region": "WALES",
                "tenure_type": "OWNED_OUTRIGHT",
                "council_tax": 1_200.0,
                "rent": 0.0,
                "mortgage_interest_repayment": 0.0,
                "mortgage_capital_repayment": 0.0,
                "savings": 6_000.0,
                "household_wealth": 95_000.0,
                "num_vehicles": 0.0,
                "council_tax_band": "B",
            },
        ]
    )
    return person_df, household_df


def test_generate_scenarios_count(sample_person_frame):
    """Generates the requested number of scenarios."""
    scenarios = scenarios_from_cps_frame(sample_person_frame, n=3, seed=42)
    assert len(scenarios) == 3


def test_generate_scenarios_deterministic(sample_person_frame):
    """Same seed produces identical scenarios."""
    s1 = scenarios_from_cps_frame(sample_person_frame, n=3, seed=123)
    s2 = scenarios_from_cps_frame(sample_person_frame, n=3, seed=123)
    for a, b in zip(s1, s2):
        assert a.id == b.id
        assert a.state == b.state
        assert a.filing_status == b.filing_status
        assert a.total_income == b.total_income
        assert a.num_children == b.num_children


def test_generate_scenarios_different_seeds(sample_person_frame):
    """Different seeds produce different samples."""
    s1 = scenarios_from_cps_frame(sample_person_frame, n=2, seed=1)
    s2 = scenarios_from_cps_frame(sample_person_frame, n=2, seed=2)
    different = sum(
        1
        for a, b in zip(s1, s2)
        if a.state != b.state or a.total_income != b.total_income
    )
    assert different > 0


def test_scenario_structure(sample_person_frame):
    """Each scenario has required fields."""
    scenarios = scenarios_from_cps_frame(sample_person_frame, n=3, seed=0)
    for scenario in scenarios:
        assert scenario.id.startswith("scenario_")
        assert len(scenario.state) == 2
        assert scenario.filing_status in SUPPORTED_FILING_STATUSES.values()
        assert len(scenario.adults) >= 1
        assert scenario.num_children >= 0
        assert scenario.year == 2025
        assert scenario.source_dataset == "enhanced_cps"


def test_invalid_households_are_filtered(sample_person_frame):
    """Ambiguous or multi-tax-unit households should not be benchmark scenarios."""
    scenarios = scenarios_from_cps_frame(sample_person_frame, n=4, seed=0)
    household_ids = {scenario.metadata["household_id"] for scenario in scenarios}
    assert 104 not in household_ids
    assert 103 not in household_ids


def test_large_households_are_allowed_when_structure_is_clean(sample_person_frame):
    """Large households with one tax/SPM/family/marital unit should remain eligible."""
    scenarios = scenarios_from_cps_frame(sample_person_frame, n=4, seed=0)
    scenario_106 = next(
        scenario for scenario in scenarios if scenario.metadata["household_id"] == 106
    )
    assert len(scenario_106.adults) == 1
    assert scenario_106.num_children == 6


def test_joint_filers_have_exactly_two_adults(sample_person_frame):
    """Joint filers should map to exactly two adults in promptable scenarios."""
    scenarios = scenarios_from_cps_frame(sample_person_frame, n=4, seed=0)
    for scenario in scenarios:
        if scenario.filing_status == "joint":
            assert len(scenario.adults) == 2


def test_single_and_hoh_filers_have_exactly_one_adult(sample_person_frame):
    """Single and HoH scenarios should exclude extra unlabeled adults."""
    scenarios = scenarios_from_cps_frame(sample_person_frame, n=4, seed=0)
    for scenario in scenarios:
        if scenario.filing_status in {"single", "head_of_household"}:
            assert len(scenario.adults) == 1


def test_richer_cps_inputs_are_preserved(sample_person_frame):
    """Raw nonzero inputs across entities should be carried into the scenario."""
    scenarios = scenarios_from_cps_frame(sample_person_frame, n=4, seed=0)
    joint = next(s for s in scenarios if s.filing_status == "joint")
    assert joint.adults[1].inputs["self_employment_income"] == 5_000.0

    disabled_single = next(s for s in scenarios if s.state == "CO")
    assert disabled_single.adults[0].inputs["is_disabled"] is True
    assert disabled_single.adults[0].inputs["disability_benefits"] == 18_000.0

    hoh = next(s for s in scenarios if s.state == "CA")
    assert hoh.adults[0].inputs["real_estate_taxes"] == 4_000.0
    assert hoh.adults[0].inputs["home_mortgage_interest"] == 9_000.0
    assert hoh.tax_unit_inputs["health_savings_account_ald"] == 800.0
    assert hoh.spm_unit_inputs["spm_unit_pre_subsidy_childcare_expenses"] == 2_400.0
    assert hoh.household_inputs["auto_loan_interest"] == 300.0
    assert hoh.household_inputs["auto_loan_balance"] == 5_000.0


def test_children_and_adults_split_by_age(sample_person_frame):
    """Adults and children should be split at age 18."""
    scenarios = scenarios_from_cps_frame(sample_person_frame, n=4, seed=0)
    for scenario in scenarios:
        for adult in scenario.adults:
            assert adult.age >= 18
        for child in scenario.children:
            assert child.age < 18


def test_pe_household_format(simple_single_scenario):
    """PE household JSON has required structure, including filing status."""
    household = simple_single_scenario.to_pe_household()

    assert "people" in household
    assert "tax_units" in household
    assert "spm_units" in household
    assert "families" in household
    assert "households" in household

    assert "adult1" in household["people"]
    person = household["people"]["adult1"]
    assert "age" in person
    assert "employment_income" in person

    tax_unit = household["tax_units"]["tax_unit"]
    assert tax_unit["filing_status"]["2025"] == "SINGLE"
    assert tax_unit["takes_up_eitc"]["2025"] is True
    assert tax_unit["would_file_if_eligible_for_refundable_credit"]["2025"] is True
    assert tax_unit["would_file_taxes_voluntarily"]["2025"] is True

    housing = household["households"]["household"]
    assert "state_code" in housing
    assert (
        household["people"]["adult1"]["takes_up_medicaid_if_eligible"]["2025"] is True
    )
    assert household["people"]["adult1"]["takes_up_ssi_if_eligible"]["2025"] is True


def test_pe_household_includes_cross_entity_inputs():
    """Scenario-level tax-unit, SPM, and household inputs should be emitted."""
    scenario = scenarios_from_cps_frame(
        pd.DataFrame(
            [
                {
                    "person_id": 1,
                    "household_id": 1,
                    "tax_unit_id": 1,
                    "spm_unit_id": 1,
                    "family_id": 1,
                    "marital_unit_id": 1,
                    "household_weight": 1.0,
                    "state_code": "CA",
                    "filing_status": "SINGLE",
                    "age": 40,
                    "employment_income": 50_000.0,
                    "real_estate_taxes": 4_200.0,
                    "health_savings_account_ald": 900.0,
                    "spm_unit_pre_subsidy_childcare_expenses": 1_200.0,
                    "auto_loan_interest": 250.0,
                    "is_tax_unit_head": True,
                }
            ]
        ),
        n=1,
        seed=0,
    )[0]

    household = scenario.to_pe_household()
    assert household["people"]["adult1"]["real_estate_taxes"]["2025"] == 4_200.0
    assert (
        household["tax_units"]["tax_unit"]["health_savings_account_ald"]["2025"]
        == 900.0
    )
    assert (
        household["spm_units"]["spm_unit"]["spm_unit_pre_subsidy_childcare_expenses"][
            "2025"
        ]
        == 1_200.0
    )
    assert (
        household["spm_units"]["spm_unit"]["takes_up_snap_if_eligible"]["2025"] is True
    )
    assert household["households"]["household"]["auto_loan_interest"]["2025"] == 250.0


def test_total_income_ignores_deductions_and_hours(sample_person_frame):
    """Display income should not add deductions or weekly hours as income."""
    scenarios = scenarios_from_cps_frame(sample_person_frame, n=4, seed=0)
    hoh = next(s for s in scenarios if s.state == "CA")
    assert hoh.total_income == 30_000.0


def test_pe_household_with_children(family_scenario):
    """PE household includes children in all groups."""
    household = family_scenario.to_pe_household()

    all_members = household["tax_units"]["tax_unit"]["members"]
    assert "adult1" in all_members
    assert "adult2" in all_members
    assert "child1" in all_members
    assert "child2" in all_members
    assert household["tax_units"]["tax_unit"]["filing_status"]["2025"] == "JOINT"
    assert len(household["people"]) == 4


def test_generate_scenarios_uses_loader(monkeypatch, sample_person_frame):
    """Top-level generation should delegate to the Enhanced CPS loader."""

    def fake_loader():
        return sample_person_frame, 2024

    monkeypatch.setattr(
        "policybench.scenarios.load_enhanced_cps_person_frame", fake_loader
    )
    scenarios = generate_scenarios(n=3, seed=0)

    assert len(scenarios) == 3
    assert all(scenario.metadata["dataset_year"] == 2024 for scenario in scenarios)
    assert all(scenario.source_dataset == "enhanced_cps_2024" for scenario in scenarios)


def test_generate_uk_scenarios_uses_loader(monkeypatch, sample_uk_frames):
    person_df, household_df = sample_uk_frames

    def fake_loader():
        return person_df, household_df, 2025

    monkeypatch.setattr(
        "policybench.scenarios.load_uk_enhanced_cps_frames", fake_loader
    )
    scenarios = generate_scenarios(n=2, seed=0, country="uk")

    assert len(scenarios) == 2
    assert all(scenario.country == "uk" for scenario in scenarios)
    assert all(scenario.filing_status is None for scenario in scenarios)
    assert all(scenario.metadata["dataset_year"] == 2025 for scenario in scenarios)
    assert all(
        scenario.source_dataset == "enhanced_cps_uk_2025" for scenario in scenarios
    )


def test_scenarios_from_cps_frame_can_exclude_households(sample_person_frame):
    """Sampling can exclude already-used benchmark households."""
    scenarios = scenarios_from_cps_frame(
        sample_person_frame,
        n=2,
        seed=0,
        excluded_household_ids={102},
    )

    assert len(scenarios) == 2
    household_ids = {scenario.metadata["household_id"] for scenario in scenarios}
    assert 102 not in household_ids
    assert len(household_ids) == 2


def test_load_excluded_household_ids_from_manifest_json(tmp_path):
    """Manifest helper should recover household ids from serialized scenarios."""
    manifest_path = tmp_path / "scenarios.csv"
    pd.DataFrame(
        [
            {
                "scenario_id": "scenario_000",
                "scenario_json": json.dumps({"metadata": {"household_id": 101}}),
            },
            {
                "scenario_id": "scenario_001",
                "scenario_json": json.dumps({"metadata": {"household_id": 205}}),
            },
        ]
    ).to_csv(manifest_path, index=False)

    assert load_excluded_household_ids(manifest_path) == {101, 205}


def test_scenario_manifest_exports_summary_fields(sample_person_frame):
    """Scenario manifest should expose the dashboard summary fields."""
    scenarios = scenarios_from_cps_frame(sample_person_frame, n=2, seed=0)
    manifest = scenario_manifest(scenarios)

    assert set(manifest.columns) == {
        "scenario_id",
        "country",
        "state",
        "filing_status",
        "num_adults",
        "num_children",
        "total_income",
        "source_dataset",
        "scenario_json",
    }
    assert len(manifest) == 2
    assert manifest["scenario_id"].str.startswith("scenario_").all()


def test_load_scenarios_from_manifest_round_trips(sample_person_frame, tmp_path):
    """Serialized manifests should reconstruct the exact scenarios."""
    scenarios = scenarios_from_cps_frame(sample_person_frame, n=2, seed=0)
    manifest_path = tmp_path / "scenarios.csv"
    scenario_manifest(scenarios).to_csv(manifest_path, index=False)

    loaded = load_scenarios_from_manifest(manifest_path)

    assert [scenario.id for scenario in loaded] == [scenario.id for scenario in scenarios]
    assert [scenario.state for scenario in loaded] == [scenario.state for scenario in scenarios]
    assert [scenario.filing_status for scenario in loaded] == [
        scenario.filing_status for scenario in scenarios
    ]
    assert [scenario.total_income for scenario in loaded] == [
        scenario.total_income for scenario in scenarios
    ]


def test_scenarios_from_uk_frames_include_region_and_household_inputs(sample_uk_frames):
    person_df, household_df = sample_uk_frames
    scenarios = scenarios_from_uk_frames(person_df, household_df, n=2, seed=0)

    assert len(scenarios) == 2
    london = next(s for s in scenarios if s.state == "LONDON")
    assert london.country == "uk"
    assert london.filing_status is None
    assert london.household_inputs["rent"] == 14_400.0
    assert london.household_inputs["tenure_type"] == "RENT_PRIVATELY"
    assert "council_tax_band" not in london.household_inputs
    assert "council_tax" not in london.household_inputs
    assert "benunit_id" not in london.household_inputs
    assert london.children[0].inputs["is_student"] is True

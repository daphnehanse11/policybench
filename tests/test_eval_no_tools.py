"""Tests for AI-alone evaluation (mocked LiteLLM calls)."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import litellm
import pytest

from policybench.config import COUNTRY_PROGRAMS
from policybench.eval_no_tools import (
    _build_resume_metadata,
    _write_resume_metadata,
    _build_answer_tool,
    extract_explanations,
    extract_number,
    extract_prediction,
    extract_predictions,
    load_repeated_predictions,
    run_no_tools_eval,
    run_no_tools_single_output_eval,
    run_repeated_no_tools_eval,
    run_single_no_tools,
)
from policybench.prompts import make_no_tools_batch_prompt, make_no_tools_prompt
from policybench.scenarios import Person, Scenario


def _write_resume_sidecar(
    output_path,
    scenarios,
    *,
    models=None,
    programs=None,
    task="eval_no_tools_batch",
    run_id=None,
    include_explanations=False,
):
    models = {"gpt-5.4": "gpt-5.4"} if models is None else models
    programs = ["income_tax"] if programs is None else programs
    metadata = _build_resume_metadata(
        task=task,
        scenarios=scenarios,
        models=models,
        programs=programs,
        run_id=run_id,
        include_explanations=include_explanations,
    )
    _write_resume_metadata(str(output_path), metadata)


@pytest.fixture
def mini_scenario():
    return Scenario(
        id="mini",
        state="CA",
        filing_status="single",
        adults=[Person(name="adult1", age=35, employment_income=50_000.0)],
        year=2025,
    )


@pytest.fixture
def rich_scenario():
    return Scenario(
        id="rich",
        state="CA",
        filing_status="head_of_household",
        adults=[
            Person(
                name="adult1",
                age=35,
                employment_income=50_000.0,
                inputs={
                    "real_estate_taxes": 4_000.0,
                    "home_mortgage_interest": 9_000.0,
                    "weekly_hours_worked": 40.0,
                },
            )
        ],
        children=[Person(name="child1", age=8, employment_income=0.0)],
        tax_unit_inputs={"health_savings_account_ald": 800.0},
        spm_unit_inputs={"spm_unit_pre_subsidy_childcare_expenses": 2_400.0},
        household_inputs={"auto_loan_interest": 300.0},
        year=2025,
    )


@pytest.fixture
def uk_scenario():
    return Scenario(
        id="uk-mini",
        country="uk",
        state="LONDON",
        filing_status=None,
        adults=[
            Person(
                name="adult1",
                age=42,
                employment_income=42_000.0,
                inputs={
                    "gender": "FEMALE",
                    "marital_status": "MARRIED",
                    "savings_interest_income": 120.0,
                    "private_pension_contributions": 600.0,
                    "hours_worked": 38.0,
                },
            )
        ],
        children=[
            Person(
                name="child1",
                age=12,
                employment_income=0.0,
                inputs={"is_student": True},
            )
        ],
        household_inputs={
            "tenure_type": "RENT_PRIVATELY",
            "rent": 14_400.0,
            "savings": 2_500.0,
            "household_wealth": 22_000.0,
        },
        year=2025,
    )


class TestExtractNumber:
    def test_plain_number(self):
        assert extract_number("5000") == 5000.0

    def test_number_with_commas(self):
        assert extract_number("5,000") == 5000.0

    def test_number_with_dollar_sign(self):
        assert extract_number("$5,000") == 5000.0

    def test_decimal(self):
        assert extract_number("0.25") == 0.25

    def test_negative(self):
        assert extract_number("-1500") == -1500.0

    def test_json_answer_object(self):
        assert extract_number('{"answer": 3500}') == 3500.0

    def test_none_for_json_answer_with_extra_text(self):
        assert (
            extract_number('{"answer": 3500}\nThe household has no refundable credits.')
            is None
        )

    def test_none_for_truncated_json_answer_prefix(self):
        assert extract_number('{"answer": 4923') is None

    def test_none_for_multiline_response_with_numeric_final_line(self):
        assert (
            extract_number("I estimate the federal income tax is modest.\n\n3500")
            is None
        )

    def test_none_for_explanatory_sentence(self):
        assert extract_number("The income tax is approximately 3500.") is None

    def test_none_for_empty(self):
        assert extract_number("") is None

    def test_none_for_no_number(self):
        assert extract_number("I cannot determine this.") is None

    def test_none_for_multiple_numbers_in_explanation(self):
        assert extract_number("Between 3000 and 5000, I estimate 4200.") is None

    def test_none_for_verbose_response_without_standalone_final_answer(self):
        assert (
            extract_number(
                "Adult 2 has $24,002 of wage income, so the household owes "
                "about $11,271 in income tax."
            )
            is None
        )

    def test_none_for_age_in_verbose_boolean_response(self):
        assert (
            extract_number(
                "Adult 2 is 42 years old and therefore the household should "
                "qualify for Medicaid."
            )
            is None
        )


class TestExtractPrediction:
    def test_tool_call_arguments_take_priority(self):
        tool_calls = [
            SimpleNamespace(
                function=SimpleNamespace(
                    name="submit_answers",
                    arguments='{"answer": 3500}',
                )
            )
        ]

        assert (
            extract_prediction(
                content="Adult 2 has $24,002 of wage income, so the answer is 11271.",
                tool_calls=tool_calls,
            )
            == 3500.0
        )

    def test_legacy_function_call_is_supported(self):
        function_call = {"name": "submit_answers", "arguments": '{"answer": 0.25}'}

        assert extract_prediction(content=None, function_call=function_call) == 0.25

    def test_falls_back_to_text_when_no_tool_call_exists(self):
        assert extract_prediction(content='{"answer": 4923}', tool_calls=None) == 4923.0

    def test_extract_predictions_returns_mapping(self):
        predictions = extract_predictions(
            content='{"income_tax": 4923, "eitc": 0}',
            variables=["income_tax", "eitc"],
            tool_calls=None,
        )
        assert predictions == {"income_tax": 4923.0, "eitc": 0.0}

    def test_extract_predictions_rejects_truncated_payload(
        self,
    ):
        predictions = extract_predictions(
            content=(
                '{"answers":{"income_tax":4923,"eitc":0},'
                '"explanations":{"income_tax":"Moderate taxable income'
            ),
            variables=["income_tax", "eitc"],
            tool_calls=None,
        )
        assert predictions == {"income_tax": None, "eitc": None}

    def test_extract_explanations_returns_mapping(self):
        explanations = extract_explanations(
            content=(
                '{"answers": {"income_tax": 4923, '
                '"eitc": 0}, "explanations": '
                '{"income_tax": '
                '"Taxable income is moderate."}}'
            ),
            variables=["income_tax", "eitc"],
            tool_calls=None,
        )
        assert explanations == {
            "income_tax": "Taxable income is moderate.",
            "eitc": None,
        }

    def test_extract_explanations_rejects_truncated_payload(self):
        explanations = extract_explanations(
            content=(
                '{"answers":{"income_tax":4923,"eitc":0},"explanations":{'
                '"income_tax":"Moderate taxable income.",'
                '"eitc":"Income too high'
            ),
            variables=["income_tax", "eitc"],
            tool_calls=None,
        )
        assert explanations == {
            "income_tax": None,
            "eitc": None,
        }

    def test_extract_predictions_does_not_scrape_stray_numbers_from_prose(self):
        predictions = extract_predictions(
            content=(
                "For free school meals, household income must be at or below "
                "130% of the federal poverty level."
            ),
            variables=["free_school_meals"],
            tool_calls=None,
        )
        assert predictions == {"free_school_meals": None}


def test_no_tools_prompt_contains_household_info(mini_scenario):
    """Prompt should describe the household."""
    prompt = make_no_tools_prompt(mini_scenario, "income_tax")
    prompt_lower = prompt.lower()
    assert "household:" in prompt_lower
    assert "filing status: single" in prompt_lower
    assert "state: ca" in prompt_lower
    assert "employment income: $50,000" in prompt
    assert "tax year: 2025" in prompt_lower


def test_no_tools_prompt_supports_uk_households(uk_scenario):
    prompt = make_no_tools_prompt(uk_scenario, "income_tax")
    prompt_lower = prompt.lower()
    assert "region: london" in prompt_lower
    assert "filing status" not in prompt_lower
    assert "annual uk income tax liability" in prompt_lower
    assert "tenure: rent privately" in prompt_lower
    assert "£42,000" in prompt
    assert "£14,400" in prompt
    assert "currency amount" in prompt_lower


def test_no_tools_prompt_asks_for_numeric(mini_scenario):
    """Prompt should request numeric-only response."""
    prompt = make_no_tools_prompt(mini_scenario, "income_tax")
    assert "numeric" in prompt.lower()
    assert "submit_answer" in prompt
    assert "plain text" in prompt.lower()


def test_no_tools_prompt_supports_json_contract(mini_scenario):
    """Gemini path should be able to request JSON output instead of a tool call."""
    prompt = make_no_tools_prompt(mini_scenario, "income_tax", answer_contract="json")
    assert '{"answer": 1234.5}' in prompt
    assert "submit_answer" not in prompt


def test_no_tools_batch_prompt_requests_all_variables(mini_scenario):
    """Batch benchmark prompt should ask for every output in one response."""
    prompt = make_no_tools_batch_prompt(
        mini_scenario,
        ["income_tax_before_refundable_credits", "eitc"],
    )
    prompt_lower = prompt.lower()
    assert "provide the following policy quantities" in prompt_lower
    assert "- income_tax_before_refundable_credits:" in prompt
    assert "- eitc:" in prompt
    assert "submit_answers" in prompt
    assert "include every requested key exactly once" in prompt.lower()


def test_no_tools_batch_prompt_requires_explanations_in_diagnostic_mode(mini_scenario):
    prompt = make_no_tools_batch_prompt(
        mini_scenario,
        ["income_tax_before_refundable_credits", "eitc"],
        include_explanations=True,
    )
    assert "`answers` object" in prompt
    assert "`explanations` object" in prompt
    assert "required" in prompt.lower()
    assert (
        "do not leave any explanation blank" in prompt.lower()
        or "non-empty" in prompt.lower()
    )


def test_answer_tool_requires_explanations_in_diagnostic_mode():
    tool = _build_answer_tool(
        ["income_tax_before_refundable_credits", "eitc"],
        country="us",
        include_explanations=True,
    )

    parameters = tool["function"]["parameters"]
    assert parameters["required"] == ["answers", "explanations"]
    assert parameters["properties"]["explanations"]["required"] == [
        "income_tax_before_refundable_credits",
        "eitc",
    ]


def test_no_tools_prompt_includes_nonzero_raw_inputs_across_entities(rich_scenario):
    """Prompt should expose the same raw nonzero inputs passed to PE."""
    prompt = make_no_tools_prompt(rich_scenario, "ctc")
    prompt_lower = prompt.lower()
    assert "real estate taxes: $4,000" in prompt_lower
    assert "home mortgage interest: $9,000" in prompt_lower
    assert "hours worked per week: 40" in prompt_lower
    assert "tax unit:" in prompt_lower
    assert "health savings account deduction: $800" in prompt_lower
    assert "spm unit:" in prompt_lower
    assert "pre-subsidy childcare expenses: $2,400" in prompt_lower
    assert "household inputs:" in prompt_lower
    assert "auto loan interest: $300" in prompt_lower


def test_income_tax_prompt_clarifies_negative_after_refundable_credits(mini_scenario):
    """Income tax prompt should make the target semantics explicit."""
    prompt = make_no_tools_prompt(mini_scenario, "income_tax")
    prompt_lower = prompt.lower()
    assert "after refundable credits" in prompt_lower
    assert "can be negative" in prompt_lower


def test_precredit_tax_prompt_distinguishes_it_from_final_tax(mini_scenario):
    """Pre-credit tax prompt should explicitly exclude refundable credits."""
    prompt = make_no_tools_prompt(mini_scenario, "income_tax_before_refundable_credits")
    prompt_lower = prompt.lower()
    assert "federal income tax" in prompt_lower
    assert "before refundable credits" in prompt_lower
    assert "do not subtract refundable credits" in prompt_lower


def test_refundable_credits_prompt_distinguishes_it_from_tax_liability(mini_scenario):
    """Refundable credits prompt should not be confusable with tax liability."""
    prompt = make_no_tools_prompt(mini_scenario, "income_tax_refundable_credits")
    prompt_lower = prompt.lower()
    assert "refundable federal tax credits only" in prompt_lower
    assert "not pre-credit tax liability" in prompt_lower


def test_agi_prompt_is_explicitly_federal(mini_scenario):
    """AGI prompt should clarify that the quantity is federal unless marked state."""
    prompt = make_no_tools_prompt(mini_scenario, "adjusted_gross_income")
    prompt_lower = prompt.lower()
    assert "federal adjusted gross income" in prompt_lower


def test_free_school_meals_prompt_clarifies_household_boolean(mini_scenario):
    """School meals prompt should ask for household
    free-meal eligibility, not dollars."""
    prompt = make_no_tools_prompt(mini_scenario, "free_school_meals")
    prompt_lower = prompt.lower()
    assert "benchmark household free-school-meal eligibility label" in prompt_lower
    assert "derived from policyengine school-meal support" in prompt_lower
    assert "reduced-price meals do not count as 1" in prompt_lower


def test_medicaid_prompt_clarifies_anyone_in_household(mini_scenario):
    """Medicaid prompt should ask whether any household member is eligible."""
    prompt = make_no_tools_prompt(mini_scenario, "is_medicaid_eligible")
    prompt_lower = prompt.lower()
    assert "anyone in the household" in prompt_lower
    assert "(1 if yes, 0 if no)" in prompt_lower


def test_state_income_tax_before_refundable_prompt_is_explicit(mini_scenario):
    """State pre-credit tax prompt should distinguish it from final state tax."""
    prompt = make_no_tools_prompt(
        mini_scenario, "state_income_tax_before_refundable_credits"
    )
    prompt_lower = prompt.lower()
    assert "state income tax before refundable credits" in prompt_lower
    assert "do not subtract state refundable credits" in prompt_lower


def test_state_refundable_credits_prompt_is_explicit(mini_scenario):
    """State refundable credits prompt should not be conflated with final state tax."""
    prompt = make_no_tools_prompt(mini_scenario, "state_refundable_credits")
    assert "total refundable state tax credits only" in prompt.lower()


@patch("policybench.eval_no_tools.responses")
def test_run_single_no_tools(mock_responses, mini_scenario):
    """GPT-5 models should use Responses API and extract tool-call answers."""
    response = SimpleNamespace(
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="submit_answers",
                arguments='{"answer": 3500}',
            )
        ],
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=3,
            total_tokens=15,
            cost=0.00123,
            input_tokens_details=SimpleNamespace(cached_tokens=4),
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
    )
    mock_responses.return_value = response

    result = run_single_no_tools(mini_scenario, "income_tax", "gpt-5.4")

    assert result["prediction"] == 3500.0
    assert result["predictions"]["income_tax"] == 3500.0
    assert "submit_answers" in result["raw_response"]
    assert "3500" in result["raw_response"]
    assert result["error"] is None
    assert result["prompt_tokens"] == 12
    assert result["completion_tokens"] == 3
    assert result["total_tokens"] == 15
    assert result["reasoning_tokens"] == 2
    assert result["cached_prompt_tokens"] == 4
    assert result["provider_reported_cost_usd"] == 0.00123
    assert result["total_cost_usd"] == 0.00123
    assert result["cost_is_estimated"] is False
    assert result["estimated_cost_usd"] == 0.00123
    assert result["elapsed_seconds"] is not None
    assert result["elapsed_seconds"] >= 0
    mock_responses.assert_called_once()
    assert mock_responses.call_args.kwargs["timeout"] == 20
    assert mock_responses.call_args.kwargs["max_output_tokens"] == 256
    assert mock_responses.call_args.kwargs["tools"][0]["name"] == "submit_answers"
    assert mock_responses.call_args.kwargs["tool_choice"]["name"] == "submit_answers"
    assert "temperature" not in mock_responses.call_args.kwargs
    assert "reasoning_effort" not in mock_responses.call_args.kwargs


@patch("policybench.eval_no_tools.completion")
def test_run_single_no_tools_does_not_retry_auth_error(mock_completion, mini_scenario):
    """Authentication failures should fail fast instead of backing off."""
    mock_completion.side_effect = litellm.AuthenticationError(
        message="missing key",
        llm_provider="anthropic",
        model="claude-opus-4-6",
    )

    with pytest.raises(litellm.AuthenticationError):
        run_single_no_tools(mini_scenario, "income_tax", "claude-opus-4-6")

    mock_completion.assert_called_once()


@patch("policybench.eval_no_tools.completion")
def test_run_single_no_tools_uses_default_completion_budget_for_claude(
    mock_completion, mini_scenario
):
    """Anthropic models should get the extended cap without GPT reasoning knobs."""
    message = MagicMock()
    message.content = None
    message.tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="submit_answers",
                arguments='{"answer": 3500}',
            )
        )
    ]
    message.function_call = None

    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    response.usage = litellm.Usage(
        prompt_tokens=12, completion_tokens=3, total_tokens=15
    )
    mock_completion.return_value = response

    run_single_no_tools(mini_scenario, "income_tax", "claude-opus-4-6")

    assert mock_completion.call_args.kwargs["max_completion_tokens"] == 256
    assert "reasoning_effort" not in mock_completion.call_args.kwargs
    assert "response_format" not in mock_completion.call_args.kwargs


@patch("policybench.eval_no_tools.completion")
def test_run_single_no_tools_repairs_partial_batch_response(
    mock_completion, mini_scenario
):
    """A follow-up repair request should fill in missing Anthropic keys."""
    first_message = MagicMock()
    first_message.content = None
    first_message.tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="submit_answers",
                arguments='{"income_tax_before_refundable_credits": 3500}',
            )
        )
    ]
    first_message.function_call = None

    second_message = MagicMock()
    second_message.content = None
    second_message.tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="submit_answers",
                arguments='{"eitc": 1200}',
            )
        )
    ]
    second_message.function_call = None

    first_response = MagicMock()
    first_response.choices = [MagicMock(message=first_message)]
    first_response.usage = litellm.Usage(
        prompt_tokens=12,
        completion_tokens=3,
        total_tokens=15,
    )

    second_response = MagicMock()
    second_response.choices = [MagicMock(message=second_message)]
    second_response.usage = litellm.Usage(
        prompt_tokens=4,
        completion_tokens=2,
        total_tokens=6,
    )

    mock_completion.side_effect = [first_response, second_response]

    result = run_single_no_tools(
        mini_scenario,
        ["income_tax_before_refundable_credits", "eitc"],
        "claude-opus-4-6",
    )

    assert result["predictions"] == {
        "income_tax_before_refundable_credits": 3500.0,
        "eitc": 1200.0,
    }
    assert result["error"] is None
    assert result["prompt_tokens"] == 16
    assert result["completion_tokens"] == 5
    assert result["total_tokens"] == 21
    assert mock_completion.call_count == 2
    repair_prompt = mock_completion.call_args_list[1].kwargs["messages"][0]["content"]
    assert (
        "provide only the following listed policy quantities" in repair_prompt.lower()
    )
    assert "- eitc:" in repair_prompt
    assert "- income_tax_before_refundable_credits:" not in repair_prompt
    assert '"responses"' in result["raw_response"]


@patch("policybench.eval_no_tools.completion")
def test_run_single_no_tools_repairs_missing_explanations(
    mock_completion, mini_scenario
):
    first_message = MagicMock()
    first_message.content = None
    first_message.tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="submit_answers",
                arguments='{"answers":{"income_tax_before_refundable_credits":3500,"eitc":1200}}',
            )
        )
    ]
    first_message.function_call = None

    second_message = MagicMock()
    second_message.content = None
    second_message.tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="submit_answers",
                arguments=(
                    '{"answers":'
                    '{"income_tax_before_refundable_credits"'
                    ':3500,"eitc":1200},'
                    '"explanations":'
                    '{"income_tax_before_refundable_credits"'
                    ':"Tax after allowances.",'
                    '"eitc":'
                    '"Credit from low earnings."}}'
                ),
            )
        )
    ]
    second_message.function_call = None

    first_response = MagicMock()
    first_response.choices = [MagicMock(message=first_message)]
    first_response.usage = litellm.Usage(
        prompt_tokens=12,
        completion_tokens=3,
        total_tokens=15,
    )

    second_response = MagicMock()
    second_response.choices = [MagicMock(message=second_message)]
    second_response.usage = litellm.Usage(
        prompt_tokens=4,
        completion_tokens=2,
        total_tokens=6,
    )

    mock_completion.side_effect = [first_response, second_response]

    result = run_single_no_tools(
        mini_scenario,
        ["income_tax_before_refundable_credits", "eitc"],
        "claude-opus-4-6",
        include_explanations=True,
    )

    assert result["error"] is None
    assert result["explanations"] == {
        "income_tax_before_refundable_credits": "Tax after allowances.",
        "eitc": "Credit from low earnings.",
    }
    repair_prompt = mock_completion.call_args_list[1].kwargs["messages"][0]["content"]
    assert "required answers and/or explanations" in repair_prompt.lower()


@patch("policybench.eval_no_tools.completion")
def test_run_single_no_tools_chunks_claude_explanation_batches(
    mock_completion, mini_scenario
):
    variables = COUNTRY_PROGRAMS["us"]
    chunk_payloads = [
        {
            "answers": {
                variable: float(index + 1) for index, variable in enumerate(chunk)
            },
            "explanations": {
                variable: f"Explanation for {variable}." for variable in chunk
            },
        }
        for chunk in [
            variables[:3],
            variables[3:6],
            variables[6:9],
            variables[9:12],
            variables[12:13],
        ]
    ]

    responses = []
    for payload in chunk_payloads:
        message = MagicMock()
        message.content = None
        message.tool_calls = [
            SimpleNamespace(
                function=SimpleNamespace(
                    name="submit_answers",
                    arguments=json.dumps(payload),
                )
            )
        ]
        message.function_call = None

        response = MagicMock()
        response.choices = [MagicMock(message=message)]
        response.usage = litellm.Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )
        responses.append(response)

    mock_completion.side_effect = responses

    result = run_single_no_tools(
        mini_scenario,
        variables,
        "claude-sonnet-4-6",
        include_explanations=True,
    )

    assert result["error"] is None
    assert len(result["predictions"]) == len(variables)
    assert len(result["explanations"]) == len(variables)
    assert all(result["explanations"][variable] for variable in variables)
    assert result["prompt_tokens"] == 50
    assert result["completion_tokens"] == 25
    assert mock_completion.call_count == 5
    assert '"chunked_responses"' in result["raw_response"]


@patch("policybench.eval_no_tools.completion")
def test_run_single_no_tools_marks_missing_predictions_after_repair(
    mock_completion, mini_scenario
):
    """Rows should remain retryable when keys are still missing after repair."""
    message = MagicMock()
    message.content = None
    message.tool_calls = [
        SimpleNamespace(
            function=SimpleNamespace(
                name="submit_answers",
                arguments='{"income_tax_before_refundable_credits": 3500}',
            )
        )
    ]
    message.function_call = None

    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    response.usage = litellm.Usage(
        prompt_tokens=12, completion_tokens=3, total_tokens=15
    )
    mock_completion.side_effect = [response, response, response]

    result = run_single_no_tools(
        mini_scenario,
        ["income_tax_before_refundable_credits", "eitc"],
        "claude-opus-4-6",
    )

    assert result["predictions"]["income_tax_before_refundable_credits"] == 3500.0
    assert result["predictions"]["eitc"] is None
    assert "Missing predictions after repair: eitc" == result["error"]
    assert mock_completion.call_count == 3


@patch("policybench.eval_no_tools.responses")
def test_run_single_no_tools_falls_back_to_text_content(mock_responses, mini_scenario):
    """Responses API text output should still parse when no function call is present."""
    response = SimpleNamespace(
        output_text='{"income_tax": 777}',
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text='{"income_tax": 777}')
                ],
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=3, total_tokens=15),
    )
    mock_responses.return_value = response

    result = run_single_no_tools(mini_scenario, "income_tax", "gpt-5.4")

    assert result["prediction"] == 777.0
    assert result["predictions"]["income_tax"] == 777.0
    assert result["raw_response"] == '{"income_tax": 777}'


@patch("policybench.eval_no_tools.responses")
def test_run_single_no_tools_supports_multiple_variables(mock_responses, mini_scenario):
    """Batch path should parse a full mapping from one Responses API tool call."""
    response = SimpleNamespace(
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                name="submit_answers",
                arguments='{"income_tax_before_refundable_credits": 3500, "eitc": 0}',
            )
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=3, total_tokens=15),
    )
    mock_responses.return_value = response

    result = run_single_no_tools(
        mini_scenario,
        ["income_tax_before_refundable_credits", "eitc"],
        "gpt-5.4",
    )

    assert result["predictions"] == {
        "income_tax_before_refundable_credits": 3500.0,
        "eitc": 0.0,
    }
    assert "submit_answers" in result["raw_response"]


@patch("policybench.eval_no_tools.completion")
def test_run_single_no_tools_uses_json_contract_for_gemini(
    mock_completion, mini_scenario
):
    """Gemini should use JSON structured output instead of the tool-call transport."""
    message = MagicMock()
    message.content = '{"income_tax": 3500}'
    message.tool_calls = None
    message.function_call = None

    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    response.usage = litellm.Usage(
        prompt_tokens=12, completion_tokens=3, total_tokens=15
    )
    mock_completion.return_value = response

    result = run_single_no_tools(
        mini_scenario,
        "income_tax",
        "gemini/gemini-3.1-pro-preview",
    )

    assert result["prediction"] == 3500.0
    assert result["predictions"]["income_tax"] == 3500.0
    assert mock_completion.call_args.kwargs["response_format"] == {
        "type": "json_object"
    }
    assert mock_completion.call_args.kwargs["max_completion_tokens"] == 2048
    assert mock_completion.call_args.kwargs["timeout"] == 60
    assert "tools" not in mock_completion.call_args.kwargs
    assert "tool_choice" not in mock_completion.call_args.kwargs


@patch("policybench.eval_no_tools.completion")
def test_run_single_no_tools_uses_xai_max_tokens(mock_completion, mini_scenario):
    """xAI models should use chat-completions function calling with max_tokens."""
    message = MagicMock()
    message.content = None
    message.tool_calls = [
        {
            "function": {
                "name": "submit_answers",
                "arguments": '{"answers": {"income_tax": 1234}}',
            }
        }
    ]
    message.function_call = None

    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    response.usage = litellm.Usage(
        prompt_tokens=12, completion_tokens=3, total_tokens=15
    )
    mock_completion.return_value = response

    result = run_single_no_tools(
        mini_scenario, "income_tax", "xai/grok-4-1-fast-non-reasoning"
    )

    assert result["prediction"] == 1234.0
    assert result["predictions"]["income_tax"] == 1234.0
    assert mock_completion.call_args.kwargs["max_tokens"] == 256
    assert "max_completion_tokens" not in mock_completion.call_args.kwargs
    assert (
        mock_completion.call_args.kwargs["tools"][0]["function"]["name"]
        == "submit_answers"
    )


@patch("policybench.eval_no_tools.completion_cost", return_value=0.00456)
@patch("policybench.eval_no_tools.responses")
def test_run_single_no_tools_falls_back_to_completion_cost(
    mock_responses,
    mock_completion_cost,
    mini_scenario,
):
    """Responses cost should be reconstructed when the provider omits billed cost."""
    response = SimpleNamespace(
        output_text='{"eitc": 123}',
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text='{"eitc": 123}')],
            )
        ],
        usage=SimpleNamespace(
            input_tokens=9,
            output_tokens=2,
            total_tokens=11,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens_details=SimpleNamespace(reasoning_tokens=1),
            cost=None,
        ),
    )
    mock_responses.return_value = response

    result = run_single_no_tools(mini_scenario, "eitc", "gpt-5.4")

    assert result["provider_reported_cost_usd"] is None
    assert result["reconstructed_cost_usd"] == 0.00456
    assert result["total_cost_usd"] == 0.00456
    assert result["cost_is_estimated"] is True
    assert result["estimated_cost_usd"] == 0.00456
    assert result["reasoning_tokens"] == 1
    mock_completion_cost.assert_called_once()


@patch("policybench.eval_no_tools.run_single_no_tools")
def test_run_no_tools_eval_skips_remaining_model_after_fatal_error(
    mock_run_single_no_tools,
    mini_scenario,
):
    """A fatal model error should not abort the whole benchmark."""
    auth_error = litellm.AuthenticationError(
        message="missing key",
        llm_provider="anthropic",
        model="claude-opus-4-6",
    )
    mock_run_single_no_tools.side_effect = auth_error

    df = run_no_tools_eval(
        [mini_scenario],
        models={"claude-opus-4.6": "claude-opus-4-6"},
        programs=["income_tax", "eitc"],
    )

    assert df.empty
    mock_run_single_no_tools.assert_called_once()


@patch("policybench.eval_no_tools.run_single_no_tools")
def test_run_no_tools_eval_resumes_from_existing_output(
    mock_run_single_no_tools,
    mini_scenario,
    tmp_path,
):
    """Existing rows should be loaded and skipped on resume."""
    output_path = tmp_path / "predictions.csv"
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        [
            {
                "model": "gpt-5.4",
                "scenario_id": "mini",
                "variable": "income_tax",
                "prediction": 123.0,
                "raw_response": "123",
                "error": None,
            }
        ]
    ).to_csv(output_path, index=False)
    _write_resume_sidecar(
        output_path,
        [mini_scenario],
        models={"gpt-5.4": "gpt-5.4"},
        programs=["income_tax", "eitc"],
    )

    mock_run_single_no_tools.return_value = {
        "predictions": {"income_tax": 123.0, "eitc": 456.0},
        "prediction": 123.0,
        "raw_response": "456",
        "error": None,
    }

    df = run_no_tools_eval(
        [mini_scenario],
        models={"gpt-5.4": "gpt-5.4"},
        programs=["income_tax", "eitc"],
        output_path=str(output_path),
    )

    assert len(df) == 2
    assert set(df["variable"]) == {"income_tax", "eitc"}
    assert df.loc[df["variable"] == "income_tax", "prediction"].iloc[0] == 123.0
    assert mock_run_single_no_tools.call_count == 1


@patch("policybench.eval_no_tools.run_single_no_tools")
def test_run_no_tools_eval_retries_rows_with_existing_errors(
    mock_run_single_no_tools,
    mini_scenario,
    tmp_path,
):
    """Error rows should not count as completed when resuming."""
    output_path = tmp_path / "predictions.csv"
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        [
            {
                "model": "gpt-5.4",
                "scenario_id": "mini",
                "variable": "income_tax",
                "prediction": None,
                "raw_response": None,
                "error": "AuthenticationError: missing key",
            },
            {
                "model": "gpt-5.4",
                "scenario_id": "mini",
                "variable": "eitc",
                "prediction": 456.0,
                "raw_response": "456",
                "error": None,
            },
        ]
    ).to_csv(output_path, index=False)
    _write_resume_sidecar(
        output_path,
        [mini_scenario],
        models={"gpt-5.4": "gpt-5.4"},
        programs=["income_tax", "eitc"],
    )

    mock_run_single_no_tools.return_value = {
        "predictions": {"income_tax": 123.0, "eitc": 456.0},
        "prediction": 123.0,
        "raw_response": "123",
        "error": None,
    }

    df = run_no_tools_eval(
        [mini_scenario],
        models={"gpt-5.4": "gpt-5.4"},
        programs=["income_tax", "eitc"],
        output_path=str(output_path),
    )

    assert len(df) == 2
    assert set(df["variable"]) == {"income_tax", "eitc"}
    assert df.loc[df["variable"] == "income_tax", "prediction"].iloc[0] == 123.0
    assert mock_run_single_no_tools.call_count == 1


@patch("policybench.eval_no_tools.run_single_no_tools")
def test_run_no_tools_eval_retries_rows_with_missing_predictions(
    mock_run_single_no_tools,
    mini_scenario,
    tmp_path,
):
    """Rows without a parsed prediction should be retried on resume."""
    output_path = tmp_path / "predictions.csv"
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        [
            {
                "model": "gpt-5.4",
                "scenario_id": "mini",
                "variable": "income_tax",
                "prediction": None,
                "raw_response": '{"answer":',
                "error": None,
            },
            {
                "model": "gpt-5.4",
                "scenario_id": "mini",
                "variable": "eitc",
                "prediction": 456.0,
                "raw_response": "456",
                "error": None,
            },
        ]
    ).to_csv(output_path, index=False)
    _write_resume_sidecar(
        output_path,
        [mini_scenario],
        models={"gpt-5.4": "gpt-5.4"},
        programs=["income_tax", "eitc"],
    )

    mock_run_single_no_tools.return_value = {
        "predictions": {"income_tax": 123.0, "eitc": 456.0},
        "prediction": 123.0,
        "raw_response": "123",
        "error": None,
    }

    df = run_no_tools_eval(
        [mini_scenario],
        models={"gpt-5.4": "gpt-5.4"},
        programs=["income_tax", "eitc"],
        output_path=str(output_path),
    )

    assert len(df) == 2
    assert df.loc[df["variable"] == "income_tax", "prediction"].iloc[0] == 123.0
    assert mock_run_single_no_tools.call_count == 1


@patch("policybench.eval_no_tools.run_single_no_tools")
def test_run_no_tools_eval_writes_resume_metadata(
    mock_run_single_no_tools,
    mini_scenario,
    tmp_path,
):
    """Each checkpointed output should include a metadata sidecar for safe resume."""
    output_path = tmp_path / "predictions.csv"
    mock_run_single_no_tools.return_value = {
        "predictions": {"income_tax": 123.0},
        "prediction": 123.0,
        "raw_response": "123",
        "error": None,
    }

    run_no_tools_eval(
        [mini_scenario],
        models={"gpt-5.4": "gpt-5.4"},
        programs=["income_tax"],
        output_path=str(output_path),
    )

    metadata_path = tmp_path / "predictions.csv.meta.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["task"] == "eval_no_tools_batch"
    assert metadata["scenario_count"] == 1
    assert metadata["programs"] == ["income_tax"]
    assert metadata["models"] == {"gpt-5.4": "gpt-5.4"}


@patch("policybench.eval_no_tools.run_single_no_tools")
def test_run_no_tools_eval_rejects_existing_output_without_metadata(
    mock_run_single_no_tools,
    mini_scenario,
    tmp_path,
):
    """Existing CSVs without a sidecar should not be resumed silently."""
    output_path = tmp_path / "predictions.csv"
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        [
            {
                "model": "gpt-5.4",
                "scenario_id": "mini",
                "variable": "income_tax",
                "prediction": 123.0,
                "raw_response": "123",
                "error": None,
            }
        ]
    ).to_csv(output_path, index=False)

    with pytest.raises(ValueError, match="missing its resume metadata sidecar"):
        run_no_tools_eval(
            [mini_scenario],
            models={"gpt-5.4": "gpt-5.4"},
            programs=["income_tax"],
            output_path=str(output_path),
        )

    mock_run_single_no_tools.assert_not_called()


@patch("policybench.eval_no_tools.run_single_no_tools")
def test_run_no_tools_eval_rejects_mismatched_resume_metadata(
    mock_run_single_no_tools,
    mini_scenario,
    tmp_path,
):
    """A sidecar from a different benchmark configuration should fail fast."""
    output_path = tmp_path / "predictions.csv"
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        [
            {
                "model": "gpt-5.4",
                "scenario_id": "mini",
                "variable": "income_tax",
                "prediction": 123.0,
                "raw_response": "123",
                "error": None,
            }
        ]
    ).to_csv(output_path, index=False)
    metadata_path = tmp_path / "predictions.csv.meta.json"
    metadata_path.write_text(
        json.dumps(
            {
                "metadata_version": 1,
                "task": "eval_no_tools_batch",
                "run_id": None,
                "include_explanations": False,
                "scenario_count": 1,
                "scenario_hash": "different",
                "programs": ["income_tax"],
                "models": {"gpt-5.4": "gpt-5.4"},
            }
        )
    )

    with pytest.raises(ValueError, match="does not match the requested benchmark settings"):
        run_no_tools_eval(
            [mini_scenario],
            models={"gpt-5.4": "gpt-5.4"},
            programs=["income_tax"],
            output_path=str(output_path),
        )

    mock_run_single_no_tools.assert_not_called()


@patch("policybench.eval_no_tools.run_single_no_tools")
def test_run_no_tools_eval_includes_run_id(
    mock_run_single_no_tools,
    mini_scenario,
):
    """Repeated runs should stamp each output row with a run_id."""
    mock_run_single_no_tools.return_value = {
        "predictions": {"income_tax": 123.0},
        "prediction": 123.0,
        "raw_response": "123",
        "error": None,
    }

    df = run_no_tools_eval(
        [mini_scenario],
        models={"gpt-5.4": "gpt-5.4"},
        programs=["income_tax"],
        run_id="run_000",
    )

    assert df["run_id"].tolist() == ["run_000"]


@patch("policybench.eval_no_tools.run_single_no_tools")
def test_run_no_tools_eval_stores_optional_explanations(
    mock_run_single_no_tools,
    mini_scenario,
):
    mock_run_single_no_tools.return_value = {
        "predictions": {"income_tax": 123.0},
        "explanations": {"income_tax": "Used wage income and filing status."},
        "prediction": 123.0,
        "raw_response": "123",
        "error": None,
    }

    df = run_no_tools_eval(
        [mini_scenario],
        models={"gpt-5.4": "gpt-5.4"},
        programs=["income_tax"],
        include_explanations=True,
    )

    assert df["explanation"].tolist() == ["Used wage income and filing status."]


@patch("policybench.eval_no_tools.run_single_no_tools")
def test_run_no_tools_single_output_eval_writes_one_row_per_call(
    mock_run_single_no_tools,
    mini_scenario,
    tmp_path,
):
    output_path = tmp_path / "predictions.csv"
    mock_run_single_no_tools.side_effect = [
        {
            "predictions": {"income_tax": 123.0},
            "explanations": {"income_tax": "Taxable wage income only."},
            "prediction": 123.0,
            "raw_response": "first",
            "error": None,
            "elapsed_seconds": 1.5,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "reasoning_tokens": 2,
            "cached_prompt_tokens": 0,
            "provider_reported_cost_usd": 0.01,
            "reconstructed_cost_usd": 0.01,
            "total_cost_usd": 0.01,
            "cost_is_estimated": False,
            "estimated_cost_usd": 0.01,
        },
        {
            "predictions": {"eitc": 456.0},
            "explanations": {"eitc": "Low earnings with one child."},
            "prediction": 456.0,
            "raw_response": "second",
            "error": None,
            "elapsed_seconds": 2.0,
            "prompt_tokens": 11,
            "completion_tokens": 6,
            "total_tokens": 17,
            "reasoning_tokens": 3,
            "cached_prompt_tokens": 0,
            "provider_reported_cost_usd": 0.02,
            "reconstructed_cost_usd": 0.02,
            "total_cost_usd": 0.02,
            "cost_is_estimated": False,
            "estimated_cost_usd": 0.02,
        },
    ]

    df = run_no_tools_single_output_eval(
        [mini_scenario],
        models={"gpt-5.4-mini": "gpt-5.4-mini"},
        programs=["income_tax", "eitc"],
        output_path=str(output_path),
        include_explanations=True,
    )

    assert len(df) == 2
    assert set(df["variable"]) == {"income_tax", "eitc"}
    assert set(df["explanation"]) == {
        "Taxable wage income only.",
        "Low earnings with one child.",
    }
    assert mock_run_single_no_tools.call_count == 2


@patch("policybench.eval_no_tools.run_no_tools_eval")
def test_run_repeated_no_tools_eval_writes_one_file_per_run(
    mock_run_no_tools_eval,
    mini_scenario,
    tmp_path,
):
    """Repeated evaluation should create separate artifacts and preserve run ids."""

    def fake_run(*args, **kwargs):
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame(
            [
                {
                    "run_id": kwargs["run_id"],
                    "model": "gpt-5.4",
                    "scenario_id": "mini",
                    "variable": "income_tax",
                    "prediction": 123.0,
                    "raw_response": "123",
                    "error": None,
                }
            ]
        )
        frame.to_csv(kwargs["output_path"], index=False)
        return frame

    mock_run_no_tools_eval.side_effect = fake_run

    df = run_repeated_no_tools_eval(
        [mini_scenario],
        repeats=2,
        output_dir=str(tmp_path),
        models={"gpt-5.4": "gpt-5.4"},
        programs=["income_tax"],
    )

    assert set(df["run_id"]) == {"run_000", "run_001"}
    assert (tmp_path / "run_000.csv").exists()
    assert (tmp_path / "run_001.csv").exists()


def test_load_repeated_predictions_adds_run_id_from_filename(tmp_path):
    """Repeated-run loader should infer run ids when the column is missing."""
    pd = pytest.importorskip("pandas")
    pd.DataFrame(
        [
            {
                "model": "gpt-5.4",
                "scenario_id": "mini",
                "variable": "income_tax",
                "prediction": 123.0,
            }
        ]
    ).to_csv(tmp_path / "run_000.csv", index=False)
    pd.DataFrame(
        [
            {
                "model": "gpt-5.4",
                "scenario_id": "mini",
                "variable": "income_tax",
                "prediction": 456.0,
                "run_id": "already_set",
            }
        ]
    ).to_csv(tmp_path / "run_001.csv", index=False)

    df = load_repeated_predictions(str(tmp_path))

    assert set(df["run_id"]) == {"run_000", "already_set"}

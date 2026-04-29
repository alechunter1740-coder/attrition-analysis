import pandas as pd
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


def _sample_df():
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "department": ["Sales", "Sales", "HR", "HR", "R&D", "R&D"],
            "overtime": ["Yes", "Yes", "No", "No", "Yes", "No"],
            "job_satisfaction": [1, 1, 2, 2, 3, 3],
            "monthly_income": [4000, 5000, 6000, 7000, 8000, 9000],
            "attrition": ["Yes", "No", "Yes", "Yes", "No", "No"],
        }
    )


def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    assert attrition_rate(df) == 50.0


def test_attrition_by_department_returns_expected_columns():
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4],
            "department": ["Sales", "Sales", "HR", "HR"],
            "attrition": ["Yes", "No", "No", "Yes"],
        }
    )
    result = attrition_by_department(df)
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_computes_within_group_rate():
    df = _sample_df()
    result = attrition_by_department(df).set_index("department")
    assert result.loc["HR", "attrition_rate"] == 100.0
    assert result.loc["Sales", "attrition_rate"] == 50.0
    assert result.loc["R&D", "attrition_rate"] == 0.0
    assert int(result.loc["HR", "leavers"]) == 2
    assert int(result.loc["HR", "employees"]) == 2


def test_attrition_by_department_sorted_descending():
    df = _sample_df()
    rates = attrition_by_department(df)["attrition_rate"].tolist()
    assert rates == sorted(rates, reverse=True)


def test_attrition_by_overtime_computes_within_group_rate():
    df = _sample_df()
    result = attrition_by_overtime(df).set_index("overtime")
    # Overtime=Yes: 3 employees, 1 left → 33.33%
    # Overtime=No:  3 employees, 2 left → 66.67%
    assert result.loc["Yes", "attrition_rate"] == 33.33
    assert result.loc["No", "attrition_rate"] == 66.67


def test_average_income_by_attrition_computes_group_means():
    df = _sample_df()
    result = average_income_by_attrition(df).set_index("attrition")
    # Leavers (Yes): 4000, 6000, 7000 → 5666.67
    # Stayers (No): 5000, 8000, 9000 → 7333.33
    assert result.loc["Yes", "avg_monthly_income"] == 5666.67
    assert result.loc["No", "avg_monthly_income"] == 7333.33


def test_satisfaction_summary_uses_within_group_denominator():
    """Guards against regression of the bug where attrition_rate divided by
    total company leavers instead of total employees in the satisfaction group."""
    df = _sample_df()
    result = satisfaction_summary(df).set_index("job_satisfaction")
    # Level 1: 2 employees, 1 left → 50%
    # Level 2: 2 employees, 2 left → 100%
    # Level 3: 2 employees, 0 left → 0%
    assert result.loc[1, "attrition_rate"] == 50.0
    assert result.loc[2, "attrition_rate"] == 100.0
    assert result.loc[3, "attrition_rate"] == 0.0


def test_satisfaction_summary_columns_and_sorted_by_satisfaction():
    df = _sample_df()
    result = satisfaction_summary(df)
    assert list(result.columns) == [
        "job_satisfaction",
        "total_employees",
        "leavers",
        "attrition_rate",
    ]
    assert result["job_satisfaction"].tolist() == sorted(result["job_satisfaction"].tolist())


def test_satisfaction_summary_rates_do_not_falsely_sum_to_100():
    """The old buggy version produced rates summing to 100 across groups.
    A correct within-group rate should not."""
    df = _sample_df()
    total = satisfaction_summary(df)["attrition_rate"].sum()
    assert total != 100.0

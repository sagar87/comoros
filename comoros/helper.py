import json
import os

import genomicsurveillance as gs
import numpy as np
import pandas as pd

from .config import Spim
from .utils import time_to_str


def get_file_path(out, prefix: str = "specimen", suffix: str = "", ending: str = "csv"):
    """
    Returns
    """
    if out.is_dir():
        file_path = os.path.join(out, f"{prefix}-{suffix}.{ending}")
    else:
        file_path = out

    return file_path


def get_lineage_tensor(genomes, england):
    dates = genomes.WeekEndDate.unique().tolist()
    ordered_lineages, other_lineages = gs.sort_lineages(
        genomes.Lineage.unique().tolist()
    )

    all_lineages = ordered_lineages + other_lineages
    all_tensor = np.stack(
        [
            (
                genomes[genomes.WeekEndDate == d]
                .pivot_table(index="LTLA", columns="Lineage", values="Count")
                .merge(england, left_index=True, right_on="lad19cd", how="right")
                .reindex(columns=all_lineages)
                .fillna(0)
                .values
            )
            for d in dates
        ],
        1,
    )
    return all_lineages, all_tensor


def rebase_lineage_tensor(merged_tensor, merged_names, baseline_lineage):
    lin_tensor = np.concatenate(
        [
            merged_tensor[
                ...,
                [
                    i
                    for i in range(merged_tensor.shape[-1])
                    if merged_names[i] != baseline_lineage
                ],
            ],
            merged_tensor[..., [merged_names.index(baseline_lineage)]],
        ],
        -1,
    )

    lin_names = [name for name in merged_names if name != baseline_lineage] + [
        baseline_lineage
    ]

    return lin_names, lin_tensor


def create_json_output(model, start_date, end_date, analysis_date, last_days=7):
    england = gs.get_england()
    R_england = model.aggregate_log_R(england.ctry19id.values)
    lambda_england = model.aggregate_lambda(england.ctry19id.values)

    R_df = (
        pd.DataFrame(
            np.quantile(
                np.exp(R_england).squeeze()[:, -last_days:].mean(1),
                [0.05, 0.25, 0.5, 0.75, 0.95],
            ).reshape(1, -1),
            columns=[5, 25, 50, 75, 95],
        )
        .melt(var_name="quantile")
        .assign(region="England")
    )

    incidence_df = (
        pd.DataFrame(
            np.quantile(
                (lambda_england.squeeze()[:, -last_days:].mean(1) / england.pop18.sum())
                * 1e5,
                [0.05, 0.25, 0.5, 0.75, 0.95],
            ).reshape(1, -1),
            columns=[5, 25, 50, 75, 95],
        )
        .melt(var_name="quantile")
        .assign(region=lambda df: "England")
    )

    model_info = {}
    model_info["model_name"] = "comoros"
    model_info["model_version"] = "0.4.1"

    meta_data = {}
    meta_data["estimate_start_date"] = time_to_str(
        pd.to_datetime(end_date) - pd.Timedelta(f"{last_days} days")
    )
    meta_data["estimate_end_date"] = end_date

    gov_uk_cases = {}
    gov_uk_cases["date_data_read"] = analysis_date
    gov_uk_cases["date_data_truncated"] = end_date
    gov_uk_cases["first_date"] = start_date

    meta_data["gov_uk_cases"] = gov_uk_cases

    R = {}
    R["estimates"] = json.loads(R_df.to_json(orient="records"))
    R["meta_data"] = meta_data

    incidence = {}
    incidence["estimates"] = json.loads(incidence_df.to_json(orient="records"))
    incidence["meta_data"] = meta_data

    json_object = {}
    json_object["model_identifier"] = model_info
    json_object["R"] = R
    json_object["incidence"] = incidence

    return json_object


def create_metaframe(date_range, analysis_date, value_type="R", geography="England"):
    df = pd.DataFrame(
        {
            "Group": ["JBC"] * len(date_range),
            "Model": ["comoros"] * len(date_range),
            "Scenario": ["Nowcast"] * len(date_range),
            "ModelType": ["Multiple"] * len(date_range),
            "Version": [0.4] * len(date_range),
            "Creation Day": [analysis_date.day] * len(date_range),
            "Creation Month": [analysis_date.month] * len(date_range),
            "Creation Year": [analysis_date.year] * len(date_range),
            "Day of Value": date_range.day,
            "Month of Value": date_range.month,
            "Year of Value": date_range.year,
            "AgeBand": ["All"] * len(date_range),
            "Geography": [geography] * len(date_range),
            "ValueType": [value_type] * len(date_range),
        }
    )
    return df


def create_dataframe(array):
    q = np.append(0.5, np.arange(0.05, 1, 0.05))
    return pd.DataFrame(
        np.quantile(array.squeeze(), q=q, axis=0).T, columns=Spim.SPIM_DATACOLS
    )


def create_spim_table(model, start_date, end_date, analysis_date):
    england = gs.get_england()
    date_range = pd.date_range(start_date, end_date)
    R_england = model.aggregate_log_R(england.ctry19id.values)
    lambda_england = model.aggregate_lambda(england.ctry19id.values)

    spim_output = pd.concat(
        [
            pd.concat(
                [
                    create_metaframe(
                        date_range, pd.to_datetime(analysis_date), value_type="R"
                    ),
                    create_dataframe(np.exp(R_england)),
                ],
                1,
            ),
            pd.concat(
                [
                    create_metaframe(
                        date_range,
                        pd.to_datetime(analysis_date),
                        value_type="incidence",
                    ),
                    create_dataframe(lambda_england),
                ],
                1,
            ),
        ],
        0,
    )

    return spim_output

from pathlib import Path
import pickle
import json
from typing import Optional
import genomicsurveillance as gs

import numpy as np
import pandas as pd
import typer

from .utils import time_to_str
from .helper import (
    get_file_path,
    get_lineage_tensor,
    rebase_lineage_tensor,
    create_json_output,
    create_spim_table,
)
from .config import Urls, Meta

app = typer.Typer()


@app.callback()
def callback():
    """
    Awesome Portal Gun
    """


@app.command()
def spim(
    analysis: Path = typer.Argument(...),
    out: Optional[Path] = typer.Argument(...),
):
    typer.secho("Loading analysis data ...", fg=typer.colors.YELLOW)
    data = pickle.load(open(analysis, "rb"))

    model = gs.MultiLineage(
        data["cases"],
        data["lin_tensor"],
        data["lin_dates"],
        data["population"],
        tau=5.1,
        alpha1=(data["cases"][..., data["lin_dates"]] / 2).reshape(317, -1, 1),
        model_kwargs=dict(handler="SVI", num_epochs=100, num_samples=500, lr=0.001),
        posterior=data["posterior"],
    )

    spim_table = create_spim_table(
        model, data["start_date"], data["end_date"], data["analysis_date"]
    )

    file_path = get_file_path(
        out, prefix="comoros", suffix=data["analysis_date"], ending="csv"
    )
    spim_table.to_csv(out, index=None)


@app.command()
def emrg(
    analysis: Path = typer.Argument(...),
    out: Optional[Path] = typer.Argument(...),
):
    typer.secho("Loading analysis data ...", fg=typer.colors.YELLOW)
    data = pickle.load(open(analysis, "rb"))

    model = gs.MultiLineage(
        data["cases"],
        data["lin_tensor"],
        data["lin_dates"],
        data["population"],
        tau=5.1,
        alpha1=(data["cases"][..., data["lin_dates"]] / 2).reshape(317, -1, 1),
        model_kwargs=dict(handler="SVI", num_epochs=100, num_samples=500, lr=0.001),
        posterior=data["posterior"],
    )

    json_object = create_json_output(
        model,
        data["start_date"],
        data["end_date"],
        data["analysis_date"],
    )

    file_path = get_file_path(
        out, prefix="comoros", suffix=data["analysis_date"], ending="json"
    )

    with open(file_path, "w") as f:
        json.dump(json_object, f)


@app.command()
def run(
    cases: Path = typer.Argument(...),
    genomes: Path = typer.Argument(...),
    out: Optional[Path] = typer.Argument(...),
):
    """
    Shoot the portal gun
    """
    typer.secho("Loading cases data ...", fg=typer.colors.YELLOW)
    analysis_date = pd.to_datetime("today")
    england = gs.get_england()
    specimen = pd.read_csv(cases, index_col=0)
    cases = specimen.values
    start_date, end_date = specimen.columns[0], specimen.columns[-1]

    genomes = pd.read_csv(genomes, index_col=0)
    all_lineages, all_tensor = get_lineage_tensor(genomes, england)

    merged_names, merged_tensor, merged_cluster = gs.preprocess_lineage_tensor(
        all_lineages, all_tensor, vocs=Meta.VOCS
    )

    baseline_lineage = "B.1.177"
    lin_names, lin_tensor = rebase_lineage_tensor(
        merged_tensor, merged_names, baseline_lineage
    )

    lin_dates = np.array(
        [
            gs.create_date_list(cases.shape[-1], time_to_str(start_date)).index(d)
            for d in genomes.WeekEndDate.unique().tolist()
        ]
    )

    model = gs.MultiLineage(
        cases,
        lin_tensor,
        lin_dates,
        england.pop18.values,
        tau=5.1,
        alpha1=(cases[..., lin_dates] / 2).reshape(317, -1, 1),
        model_kwargs=dict(handler="SVI", num_epochs=100, num_samples=500, lr=0.001),
    )
    model.fit()

    data = dict(
        start_date=start_date,
        end_date=end_date,
        analysis_date=time_to_str(analysis_date),
        cases=cases,
        lin_tensor=lin_tensor,
        lin_dates=lin_dates,
        population=england.pop18.values,
        alpha1=(cases[..., lin_dates] / 2).reshape(317, -1, 1),
        posterior=model.posterior.data,
    )
    file_path = get_file_path(
        out, prefix="analysis", suffix=time_to_str(analysis_date), ending="pkl"
    )

    pickle.dump(data, open(file_path, "wb"))
    typer.secho("All done!", fg=typer.colors.GREEN)


@app.command()
def genomes(
    out: Optional[Path] = typer.Argument(...),
    url: Optional[str] = typer.Option(
        Urls.SANGER,
        "-s",
        "--start-date",
        help="start time of the time series",
        show_default=True,
    ),
):
    """
    Laod and preprocess genomes.
    """
    typer.secho(
        f"Trying to download genome table from {url} ...", fg=typer.colors.YELLOW
    )
    genomes = pd.read_csv(url, sep="\t")

    typer.secho("Download successful ... Formatting ...", fg=typer.colors.GREEN)
    genomes_flat = (
        genomes.loc[lambda df: df["Lineage"] != "Lineage data suppressed"]
        .assign(
            Lineage=lambda df: df["Lineage"].apply(lambda x: Meta.CONVENTIONS.get(x, x))
        )
        .assign(
            Lineage=lambda df: df["Lineage"].apply(
                lambda x: Meta.ALIASES[x]
                if ((x in Meta.ALIASES.keys()) and (x not in Meta.VOCS))
                else x
            )
        )
    )

    analysis_date = pd.to_datetime("today")
    file_path = get_file_path(out, prefix="genomes", suffix=time_to_str(analysis_date))
    typer.secho(f"Saving data to {file_path} ...", fg=typer.colors.YELLOW)
    genomes_flat.to_csv(file_path)
    typer.secho("All done!", fg=typer.colors.GREEN)


@app.command()
def specimen(
    out: Optional[Path] = typer.Argument(...),
    start_date: Optional[str] = typer.Option(
        "2020-09-01",
        "-s",
        "--start-date",
        help="start time of the time series",
        show_default=True,
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "-e",
        "--end_date",
        help="end time of the time series (if no date is passed the last 5 days are truncated)",
        show_default=True,
    ),
):
    """
    Downloads specimen data from GOV UK API.
    """
    analysis_date = pd.to_datetime("today")
    file_path = get_file_path(out, prefix="specimen", suffix=time_to_str(analysis_date))
    if end_date is None:
        end_date = time_to_str(analysis_date - pd.Timedelta("5 days"))

    typer.secho("Downloading data ...", fg=typer.colors.YELLOW)
    england = gs.get_england()
    specimen = gs.get_specimen()
    typer.secho(f"Formatting data saving file {file_path} ...", fg=typer.colors.YELLOW)
    cases = england.merge(
        specimen.T, left_on="lad19cd", right_index=True, how="left"
    ).loc[:, time_to_str(start_date) : time_to_str(end_date)]
    cases.to_csv(file_path)
    typer.secho("All done!", fg=typer.colors.GREEN)

#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import typing as t

import click
import requests


@click.command()
@click.option("--environment", default="production", type=str, show_default=True)
@click.option(
    "--ci_api_v4_url",
    envvar="CI_API_V4_URL",
    type=str,
    help="The GitLab API v4 root URL.",
    show_default=True,
)
@click.option(
    "--ci_project_id",
    envvar="CI_PROJECT_ID",
    type=int,
    help="The project's ID",
    required=True,
)
@click.option(
    "--private_token",
    envvar="PRIVATE_TOKEN",
    required=False,
    help="A token to authenticate",
)
# @click.option(
#     "--ci_job_token",
#     envvar="CI_JOB_TOKEN",
#     required=False,
#     help="A token to authenticate",
# )
def main(
    environment: str,
    ci_api_v4_url: str,
    ci_project_id: int,
    private_token: t.Optional[str],
    # ci_job_token: t.Optional[str],
):
    # Create token
    headers = {}

    # if ci_job_token:
    #     headers["TOKEN"] = ci_job_token
    if private_token:
        headers["PRIVATE-TOKEN"] = private_token
    else:
        raise NotImplementedError

    # Get main http address
    addr = f"{ci_api_v4_url}/projects/{ci_project_id}"

    # Get list of environments
    # GET /projects/:id/environments
    url_env_id = f"{addr}/environments?name={environment}"

    r = requests.get(url_env_id, headers=headers)
    r.raise_for_status()

    rsp = r.json()
    environment_id = rsp[0]["id"]  # type: int

    # Get a specific environment
    # GET /projects/:id/environments/:environment_id
    url_env = f"{addr}/environments/{environment_id}"

    r = requests.get(url_env, headers=headers)
    r.raise_for_status()

    rsp = r.json()
    deployable_dct = rsp["last_deployment"]["deployable"]  # type: t.Mapping
    job_id = deployable_dct["id"]  # type: int
    arctifact_filename = deployable_dct["artifacts_file"]["filename"]  # type: str

    # Download artifact
    url_download = f"{addr}/jobs/{job_id}/artifacts"

    r = requests.get(url_download, headers=headers)
    r.raise_for_status()

    # Save arctifact
    with open(arctifact_filename, "wb") as fh:
        fh.write(r.content)


if __name__ == "__main__":
    main()

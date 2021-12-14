#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020, 2021.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.

import sys
import typing as t

import click
import requests


@click.command()
@click.option("--environment", default="production", type=str, show_default=True)
@click.option(
    "--ci_server_host",
    envvar="CI_SERVER_HOST",
    type=str,
    help="Server host",
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
@click.option(
    "--ci_job_token",
    envvar="CI_JOB_TOKEN",
    required=False,
    help="Job token",
)
def main(
    environment: str,
    ci_server_host: str,
    ci_project_id: int,
    private_token: t.Optional[str],
    ci_job_token: t.Optional[str],
):
    # https://gitlab-ci-token:[MASKED]@gitlab.esa.int/sci-fv/pyxel-mirror.git
    headers = {}

    if private_token:
        headers["PRIVATE-TOKEN"] = private_token
        server_host = f"https://{ci_server_host}"  # type: str
    elif ci_job_token:
        server_host = f"https://gitlab-ci-token:{ci_job_token}@{ci_server_host}"

    else:
        raise NotImplementedError

    # Get main http address
    addr = f"{server_host}/api/v4/projects/{ci_project_id}"

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
    artifact_filename = deployable_dct["artifacts_file"]["filename"]  # type: str

    # Download artifact
    url_download = f"{addr}/jobs/{job_id}/artifacts"

    r = requests.get(url_download, headers=headers)
    r.raise_for_status()

    # Save artifact
    with open(artifact_filename, "wb") as fh:
        fh.write(r.content)

    print(f"File {artifact_filename!r} is successfully downloaded !")
    sys.exit()


if __name__ == "__main__":
    main()

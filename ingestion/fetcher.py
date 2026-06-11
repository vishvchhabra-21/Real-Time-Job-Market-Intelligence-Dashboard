from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from requests import Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import streamlit as st
except Exception:  # pragma: no cover - Streamlit is optional for CLI ingestion
    st = None

from config.loader import load_config


LOGGER = logging.getLogger(__name__)
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ENDPOINT = "https://jsearch.p.rapidapi.com/search"
RAPIDAPI_HOST = "jsearch.p.rapidapi.com"
DEFAULT_TIMEOUT_SECONDS = 30
SECRETS_CONFIG = load_config().get("secrets", {})
SEARCH_API_KEY_NAME = str(SECRETS_CONFIG.get("search_api_key_name", ""))


class JobSourceError(RuntimeError):
    """Raised when an upstream job source fails in a non-recoverable way."""


@dataclass(frozen=True)
class JobQuery:
    search_term: str
    location: str
    results_wanted: int = 25
    hours_old: int | None = 168
    country_indeed: str = "india"

    def as_search_phrase(self) -> str:
        if self.location.strip().lower() == "remote":
            return f"{self.search_term} remote"
        return f"{self.search_term} in {self.location}"


def _default_api_file_candidates() -> list[Path]:
    return [
        WORKSPACE_ROOT / "api.txt",
        WORKSPACE_ROOT / "API.txt",
    ]


def load_workspace_credentials(api_file_path: str | Path | None = None) -> dict[str, str]:
    """
    Load local credentials directly from the workspace credential file.

    The project spec asks for credentials to be sourced from `api.txt`; this
    helper also checks `API.txt` to tolerate the current workspace naming.
    """

    candidates: list[Path] = []
    if api_file_path is not None:
        candidates.append(Path(api_file_path))
    candidates.extend(_default_api_file_candidates())

    credentials: dict[str, str] = {}
    for candidate in candidates:
        if not candidate.exists():
            continue

        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            credentials[key.strip()] = value.strip()

        if credentials:
            return credentials

    return credentials


def get_credential(name: str, api_file_path: str | Path | None = None) -> str | None:
    if st is not None:
        try:
            if name in st.secrets:
                return str(st.secrets[name])
        except Exception as exc:
            LOGGER.debug("Unable to read Streamlit secrets for %s: %s", name, exc)

    credentials = load_workspace_credentials(api_file_path=api_file_path)
    return credentials.get(name) or os.getenv(name)


def get_jsearch_key(api_file_path: str | Path | None = None) -> str:
    """
    Resolve the RapidAPI key from Streamlit secrets, the environment, or a local
    api.txt/API.txt file.
    """

    secret_names = [SEARCH_API_KEY_NAME, "JSEARCH_KEY", "RAPIDAPI_KEY"]
    if st is not None:
        try:
            for secret_name in secret_names:
                if secret_name and secret_name in st.secrets:
                    key = str(st.secrets[secret_name]).strip()
                    if key:
                        return key
        except Exception as exc:
            LOGGER.debug("Unable to read Streamlit secrets: %s", exc)

    for env_name in secret_names:
        if env_name:
            key = os.environ.get(env_name, "").strip()
            if key:
                return key

    credentials = load_workspace_credentials(api_file_path=api_file_path)
    for credential_name in secret_names:
        if credential_name and credentials.get(credential_name):
            return str(credentials[credential_name]).strip()

    return ""


def build_retrying_session() -> Session:
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class JSearchClient:
    """Thin RapidAPI client for JSearch with conservative retry behavior."""

    def __init__(
        self,
        credential_value: str | None = None,
        *,
        api_file_path: str | Path | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        session: Session | None = None,
    ) -> None:
        self.credential_value = credential_value or get_jsearch_key(api_file_path=api_file_path)
        self.timeout_seconds = timeout_seconds
        self.session = session or build_retrying_session()

    def is_configured(self) -> bool:
        return bool(self.credential_value)

    def fetch_jobs(self, query: JobQuery) -> list[dict[str, Any]]:
        if not self.credential_value:
            raise JobSourceError("Search credential is missing; cannot call JSearch.")

        params = {
            "query": query.as_search_phrase(),
            "page": "1",
            "num_pages": "1",
            "country": "in",
            "date_posted": "all",
        }
        headers = {
            "X-RapidAPI-Key": self.credential_value,
            "X-RapidAPI-Host": RAPIDAPI_HOST,
        }

        response = self.session.get(
            SEARCH_ENDPOINT,
            params=params,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        if response.status_code >= 400:
            raise JobSourceError(
                f"JSearch request failed with status {response.status_code}: {response.text[:200]}"
            )

        payload = response.json()
        if not isinstance(payload, dict):
            raise JobSourceError("JSearch returned a non-object JSON payload.")
        records = payload.get("data", [])
        if not isinstance(records, list):
            raise JobSourceError("JSearch payload did not contain a list under 'data'.")

        limited_records = records[: query.results_wanted]
        for record in limited_records:
            record["_source"] = "jsearch"
        return limited_records


def fetch_jobs(
    search_term: str,
    location: str,
    *,
    results_wanted: int = 25,
    api_file_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch raw JSearch jobs for a single search term/location pair.
    """

    client = JSearchClient(api_file_path=api_file_path)
    query = JobQuery(search_term=search_term, location=location, results_wanted=results_wanted)
    return client.fetch_jobs(query)


class IndeedJobSpyClient:
    """
    Optional adapter around `python-jobspy`.

    The underlying scraper can break when upstream site markup changes, so this
    client is intentionally isolated behind a soft dependency and clear errors.
    """

    def __init__(self, *, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> None:
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def is_available() -> bool:
        try:
            from jobspy import scrape_jobs  # noqa: F401
        except Exception:
            return False
        return True

    def fetch_jobs(self, query: JobQuery) -> list[dict[str, Any]]:
        try:
            from jobspy import scrape_jobs
        except Exception as exc:  # pragma: no cover - exercised by runtime environment
            raise JobSourceError(
                "python-jobspy is not installed or failed to import; Indeed scraping is unavailable."
            ) from exc

        try:
            jobs_frame = scrape_jobs(
                site_name="indeed",
                search_term=query.search_term,
                location=query.location,
                results_wanted=query.results_wanted,
                hours_old=query.hours_old,
                country_indeed=query.country_indeed,
            )
        except Exception as exc:
            raise JobSourceError(f"Indeed scraper failed for {query.as_search_phrase()}.") from exc

        if jobs_frame is None:
            return []

        records = jobs_frame.to_dict(orient="records")
        for record in records:
            record["_source"] = "indeed"
        return records


def fetch_all_sources(
    query: JobQuery,
    *,
    jsearch_client: JSearchClient | None = None,
    indeed_client: IndeedJobSpyClient | None = None,
    include_jsearch: bool = True,
    include_indeed: bool = True,
) -> list[dict[str, Any]]:
    """
    Convenience helper for callers that want a merged raw payload.

    Errors are logged per source and do not prevent other sources from running.
    """

    records: list[dict[str, Any]] = []

    if include_jsearch:
        client = jsearch_client or JSearchClient()
        try:
            records.extend(client.fetch_jobs(query))
        except JobSourceError as exc:
            LOGGER.warning("JSearch fetch failed for '%s': %s", query.as_search_phrase(), exc)

    if include_indeed:
        client = indeed_client or IndeedJobSpyClient()
        try:
            records.extend(client.fetch_jobs(query))
        except JobSourceError as exc:
            LOGGER.warning("Indeed fetch failed for '%s': %s", query.as_search_phrase(), exc)

    return records

from __future__ import annotations

import hashlib
import html
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from config.loader import load_config


CONFIG = load_config()
MAX_DESCRIPTION_LENGTH = int(CONFIG["ingestion"]["description_max_chars"])
UNKNOWN_COMPANY = "Unknown Company"
UNKNOWN_LOCATION = "Unknown"

LOCATION_RULES: list[tuple[tuple[str, ...], str]] = [
    (("remote", "work from home", "wfh"), "Remote"),
    (("new delhi", "delhi ncr", "delhi/ncr", "ncr", "delhi", "gurugram", "gurgaon", "noida"), "Delhi"),
    (("bangalore", "bengaluru"), "Bangalore"),
    (("mumbai", "bombay"), "Mumbai"),
    (("hyderabad",), "Hyderabad"),
    (("pune",), "Pune"),
    (("chennai",), "Chennai"),
    (("kolkata", "calcutta"), "Kolkata"),
]

AMOUNT_PATTERN = re.compile(
    r"(?P<currency>₹|rs\.?|inr|\$|usd)?\s*"
    r"(?P<value>\d+(?:[\d,]*\d)?(?:\.\d+)?)\s*"
    r"(?P<suffix>k|l|lac|lakh|m|mn|million|cr|crore)?",
    re.IGNORECASE,
)


def collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", value)
    unescaped = html.unescape(without_tags)
    return collapse_whitespace(unescaped)


def truncate_text(value: str, max_chars: int = MAX_DESCRIPTION_LENGTH) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip()


def clean_description(value: str | None, max_chars: int = MAX_DESCRIPTION_LENGTH) -> str:
    return truncate_text(strip_html(value), max_chars=max_chars)


def clean_job(
    raw_job: dict[str, Any],
    *,
    source: str = "jsearch",
    fetched_at: str | None = None,
) -> dict[str, str | None]:
    return build_canonical_job(raw_job, source=source, fetched_at=fetched_at)


def clean_jobs(
    raw_jobs: list[dict[str, Any]] | None,
    *,
    source: str = "jsearch",
    fetched_at: str | None = None,
) -> list[dict[str, str | None]]:
    if not raw_jobs:
        return []

    batch_fetched_at = fetched_at or datetime.now(timezone.utc).isoformat()
    cleaned_jobs: list[dict[str, str | None]] = []
    for raw_job in raw_jobs:
        raw_source = str(raw_job.get("_source") or source)
        cleaned_jobs.append(
            build_canonical_job(
                raw_job,
                source=raw_source,
                fetched_at=batch_fetched_at,
            )
        )
    return cleaned_jobs


def normalize_location(value: str | None) -> str:
    if not value:
        return UNKNOWN_LOCATION

    lowered = value.strip().lower()
    simplified = re.sub(r"[^a-z0-9/\s-]+", " ", lowered)
    simplified = collapse_whitespace(simplified)

    for aliases, canonical in LOCATION_RULES:
        if any(alias in simplified for alias in aliases):
            return canonical

    return collapse_whitespace(value).title()


def _currency_code(raw_value: str) -> str:
    lowered = raw_value.lower()
    if "$" in raw_value or "usd" in lowered:
        return "USD"
    return "INR"


def _salary_interval(raw_value: str | None) -> str | None:
    if not raw_value:
        return None

    lowered = raw_value.lower()
    if "hour" in lowered or "/hr" in lowered:
        return "hourly"
    if "month" in lowered or "/mo" in lowered:
        return "monthly"
    if "week" in lowered:
        return "weekly"
    if "day" in lowered:
        return "daily"
    if any(token in lowered for token in ("year", "annum", "lpa", "ctc", "/yr")):
        return "yearly"
    return None


def _amount_multiplier(suffix: str | None) -> float:
    normalized = (suffix or "").lower()
    if normalized == "k":
        return 1_000
    if normalized in {"l", "lac", "lakh"}:
        return 100_000
    if normalized in {"m", "mn", "million"}:
        return 1_000_000
    if normalized in {"cr", "crore"}:
        return 10_000_000
    return 1


def _extract_amounts(raw_value: str) -> list[int]:
    amounts: list[int] = []
    for match in AMOUNT_PATTERN.finditer(raw_value):
        value_text = match.group("value")
        if not value_text:
            continue
        numeric_value = float(value_text.replace(",", ""))
        scaled_value = int(numeric_value * _amount_multiplier(match.group("suffix")))
        amounts.append(scaled_value)
    return amounts


def format_salary(min_amount: int | None, max_amount: int | None, currency: str, interval: str | None) -> str | None:
    if min_amount is None and max_amount is None:
        return None

    if min_amount is not None and max_amount is not None and min_amount > max_amount:
        min_amount, max_amount = max_amount, min_amount

    if min_amount is not None and max_amount is not None:
        body = f"{currency} {min_amount} - {max_amount}"
    else:
        single_amount = min_amount if min_amount is not None else max_amount
        body = f"{currency} {single_amount}"

    if interval:
        return f"{body} {interval}"
    return body


def parse_salary_string(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, dict):
        currency = str(
            value.get("currency")
            or value.get("job_salary_currency")
            or value.get("salary_currency")
            or "INR"
        ).upper()
        interval = value.get("interval") or value.get("salary_period") or value.get("period")
        min_amount = value.get("min") or value.get("min_amount") or value.get("job_min_salary")
        max_amount = value.get("max") or value.get("max_amount") or value.get("job_max_salary")
        try:
            min_cast = int(float(min_amount)) if min_amount is not None else None
            max_cast = int(float(max_amount)) if max_amount is not None else None
        except (TypeError, ValueError):
            min_cast, max_cast = None, None
        return format_salary(min_cast, max_cast, currency=currency, interval=str(interval).lower() if interval else None)

    if isinstance(value, (list, tuple)):
        numeric_values = [int(float(item)) for item in value if item is not None]
        if not numeric_values:
            return None
        min_amount = min(numeric_values)
        max_amount = max(numeric_values) if len(numeric_values) > 1 else None
        return format_salary(min_amount, max_amount, currency="INR", interval=None)

    if isinstance(value, (int, float)):
        return format_salary(int(value), None, currency="INR", interval=None)

    raw_string = collapse_whitespace(str(value))
    if not raw_string:
        return None

    amounts = _extract_amounts(raw_string)
    currency = _currency_code(raw_string)
    interval = _salary_interval(raw_string)
    if amounts:
        first_amount = amounts[0]
        second_amount = amounts[1] if len(amounts) > 1 else None
        return format_salary(first_amount, second_amount, currency=currency, interval=interval)

    return raw_string


def canonicalize_skills(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        cleaned = collapse_whitespace(value)
        return cleaned or None

    if isinstance(value, (list, tuple, set)):
        unique_values: list[str] = []
        seen: set[str] = set()
        for item in value:
            candidate = collapse_whitespace(strip_html(str(item)))
            if not candidate:
                continue
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique_values.append(candidate)
        return ", ".join(unique_values) if unique_values else None

    if isinstance(value, dict):
        collected: list[str] = []
        for maybe_values in value.values():
            if isinstance(maybe_values, (list, tuple, set)):
                collected.extend(str(item) for item in maybe_values)
        return canonicalize_skills(collected)

    return collapse_whitespace(str(value)) or None


def ensure_iso8601(value: Any, *, fallback: str | None = None) -> str:
    if isinstance(value, datetime):
        candidate = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return candidate.astimezone(timezone.utc).isoformat()

    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).isoformat()

    if isinstance(value, str) and value.strip():
        normalized = value.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return fallback or datetime.now(timezone.utc).isoformat()
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()

    return fallback or datetime.now(timezone.utc).isoformat()


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}, ()):
            return value
    return None


def _compose_location(raw_job: dict[str, Any]) -> str:
    if raw_job.get("job_is_remote") or str(raw_job.get("is_remote", "")).lower() == "true":
        return "Remote"

    direct_location = _coalesce(
        raw_job.get("location"),
        raw_job.get("job_location"),
        raw_job.get("search_location"),
    )
    if direct_location:
        return str(direct_location)

    city = _coalesce(raw_job.get("job_city"), raw_job.get("city"))
    state = _coalesce(raw_job.get("job_state"), raw_job.get("state"))
    country = _coalesce(raw_job.get("job_country"), raw_job.get("country"))
    parts = [str(part) for part in (city, state, country) if part]
    return ", ".join(parts)


def derive_job_id(source: str, raw_job: dict[str, Any]) -> str:
    explicit_id = _coalesce(raw_job.get("job_id"), raw_job.get("id"))
    if explicit_id:
        return f"{source}:{explicit_id}"

    url_identifier = _coalesce(raw_job.get("job_url"), raw_job.get("url"), raw_job.get("job_url_direct"))
    if url_identifier:
        return f"{source}:{url_identifier}"

    fingerprint_seed = "||".join(
        [
            source,
            str(_coalesce(raw_job.get("job_title"), raw_job.get("title"), "")),
            str(_coalesce(raw_job.get("employer_name"), raw_job.get("company"), "")),
            str(_compose_location(raw_job)),
        ]
    )
    digest = hashlib.sha1(fingerprint_seed.encode("utf-8")).hexdigest()
    return f"{source}:{digest}"


def build_canonical_job(
    raw_job: dict[str, Any],
    *,
    source: str,
    fetched_at: str | None = None,
) -> dict[str, str | None]:
    title = strip_html(str(_coalesce(raw_job.get("job_title"), raw_job.get("title"), "Untitled Role")))
    company = strip_html(str(_coalesce(raw_job.get("employer_name"), raw_job.get("company"), UNKNOWN_COMPANY)))
    location = normalize_location(_compose_location(raw_job))

    salary_payload = _coalesce(
        raw_job.get("salary"),
        {
            "job_min_salary": raw_job.get("job_min_salary"),
            "job_max_salary": raw_job.get("job_max_salary"),
            "job_salary_currency": raw_job.get("job_salary_currency"),
            "salary_period": raw_job.get("job_salary_period"),
        },
        raw_job.get("compensation"),
    )
    description = clean_description(
        _coalesce(raw_job.get("job_description"), raw_job.get("description"), raw_job.get("snippet"), title)
    )
    skills = canonicalize_skills(
        _coalesce(raw_job.get("skills"), raw_job.get("job_required_skills"), raw_job.get("job_highlights"))
    )
    posted_at = ensure_iso8601(
        _coalesce(
            raw_job.get("job_posted_at_datetime_utc"),
            raw_job.get("date_posted"),
            raw_job.get("posted_at"),
        )
    )
    fetched_iso = ensure_iso8601(fetched_at)

    return {
        "id": derive_job_id(source, raw_job),
        "title": title or "Untitled Role",
        "company": company or UNKNOWN_COMPANY,
        "location": location or UNKNOWN_LOCATION,
        "salary": parse_salary_string(salary_payload),
        "description": description or title or "No description provided.",
        "skills": skills,
        "posted_at": posted_at,
        "fetched_at": fetched_iso,
        "source": source,
    }

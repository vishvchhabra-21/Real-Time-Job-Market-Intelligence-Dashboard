# Contributing Guide

Thanks for helping improve the Real-Time Job Market Intelligence Dashboard.

## Workflow

1. Fork the repository.
2. Create a feature branch from `main`.
3. Open a pull request when your change is ready.

## Branch and PR Naming

- Branches: `feature/<short-name>`, `fix/<short-name>`, or `docs/<short-name>`
- Pull requests: `feat: <short description>`, `fix: <short description>`, or `docs: <short description>`

## Adding New Skills

1. Open `config/settings.yaml`.
2. Add the new keyword under `ml.skill_keywords`.
3. Keep entries lowercase and concise.
4. The skill extractor will pick it up automatically on the next run.

## Adding a New Job Source

1. Add the source client in `ingestion/fetcher.py`.
2. Normalize the payload in `ingestion/cleaner.py` if the source uses different field names.
3. Register it in `ingestion/scheduler.py`.
4. Test the source against the SQLite pipeline before opening a PR.

## Good PRs Include

- A clear summary of the change.
- Screenshots for UI updates.
- Test evidence for backend or ML changes.
- Notes about any new config keys or environment variables.

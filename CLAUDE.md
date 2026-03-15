# Claude Code Instructions

## Git workflow
- Always work directly on `main`. Do not create worktree branches.
- Commit and push to `main` after every change.

## Running the app
```bash
.venv\Scripts\python.exe app.py   # Windows
```
Runs on port 8080. Admin panel at `/admin`.

## Database
- Uses `mysql.connector` (not pymysql).
- Connection env vars: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`.

## GCS
- Images are uploaded to GCS via `upload_bytes_to_gcs()`.
- Env var: `GCS_BUCKET_NAME`. Optional prefix: `GCS_PATH_PREFIX`.
- Local static files are used as fallback when GCS is not configured.

## Deployment
- Hosted on Google Cloud Run (europe-west1).
- CI/CD via GitHub Actions (`.github/workflows/main.yml`).
- Env vars are passed from GitHub Secrets to Cloud Run at deploy time.
- FB token: System User token (never expires) via Meta Business portfolio.

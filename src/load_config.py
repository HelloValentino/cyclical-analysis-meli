# src/load_config.py
from __future__ import annotations

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def load_dotenv(dotenv_path: Path) -> bool:
    """
    Minimal .env loader (no external deps).
    Loads KEY=VALUE lines into os.environ if not already set.
    """
    if not dotenv_path.exists():
        return False

    loaded_any = False
    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            loaded_any = True
    return loaded_any


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    dotenv_path = project_root / ".env"

    if load_dotenv(dotenv_path):
        logger.info("✓ Loaded .env from: %s", dotenv_path)
    else:
        logger.warning("No .env loaded. Expected at: %s", dotenv_path)

    # Surface SEC identity readiness (for compliance)
    email = os.getenv("SEC_EDGAR_EMAIL", "").strip()
    name = os.getenv("SEC_EDGAR_COMPANY_NAME", "").strip()
    if email and name:
        logger.info("✓ SEC EDGAR identity configured: %s | %s", name, email)
    else:
        logger.warning(
            "SEC EDGAR identity not configured. Set SEC_EDGAR_EMAIL and SEC_EDGAR_COMPANY_NAME in .env for SEC compliance."
        )


if __name__ == "__main__":
    main()
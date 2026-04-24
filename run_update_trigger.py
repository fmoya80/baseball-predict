import os
import sys
import requests


def main() -> int:
    base_url = os.getenv("INTERNAL_UPDATE_BASE_URL")
    token = os.getenv("INTERNAL_UPDATE_TOKEN")

    if not base_url:
        print("ERROR: INTERNAL_UPDATE_BASE_URL no está configurada")
        return 1

    if not token:
        print("ERROR: INTERNAL_UPDATE_TOKEN no está configurada")
        return 1

    url = f"{base_url.rstrip('/')}/internal/run-update"

    print(f"Disparando update hacia: {url}")

    response = requests.post(
        url,
        headers={"x-update-token": token},
        timeout=(30, 3600),
    )

    print(f"Status code: {response.status_code}")
    print(f"Response body: {response.text}")

    response.raise_for_status()
    return 0


if __name__ == "__main__":
    sys.exit(main())
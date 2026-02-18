import json
import requests
import time

from ocr_workflow.settings import (
    USERNAME_TRANSKRIBUS_RA,
    PASSWORD_TRANSKRIBUS_RA,
    USERNAME_TRANSKRIBUS_GL,
    PASSWORD_TRANSKRIBUS_GL,
)

# Timeout in seconds for all outgoing Transkribus HTTP requests.
# connect timeout = 10 s, read timeout = 60 s (image uploads can be large).
REQUEST_TIMEOUT = (10, 60)


def transkribus_login(username: str, password: str) -> str:
    """
    Authenticates a user with the Transkribus API and returns an access token.

    Sends a password-grant OAuth request to the ReadCoop authentication server.
    The returned token must be included as a Bearer token in subsequent API calls.

    Args:
        username: The Transkribus account username.
        password: The Transkribus account password.

    Returns:
        A string containing the OAuth access token.

    Raises:
        requests.HTTPError: If the authentication request fails (e.g. wrong credentials).
        requests.Timeout: If the request exceeds ``REQUEST_TIMEOUT``.
    """
    token_url = "https://account.readcoop.eu/auth/realms/readcoop/protocol/openid-connect/token"
    credentials = {
        "grant_type": "password",
        "client_id": "processing-api-client",
        "username": username,
        "password": password,
    }
    response = requests.post(token_url, data=credentials, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    access_token = response.json().get("access_token")

    return access_token


def start_recognition_process(access_token: str, image_data: str, model_id: int) -> tuple[str, dict]:
    """
    Submits an image to the Transkribus processing API and starts an HTR recognition job.

    The image is sent as a Base64-encoded string along with the desired HTR model ID.
    The API assigns a process ID that can be used to poll for the job's completion status.

    Args:
        access_token: A valid Transkribus OAuth access token.
        image_data: The image to recognise, encoded as a Base64 string.
        model_id: The numeric ID of the HTR model to use for text recognition.

    Returns:
        A tuple of (process_id, headers) where process_id is the string identifier
        assigned by Transkribus and headers is the auth header dict to reuse for
        subsequent status checks.

    Raises:
        requests.HTTPError: If the API request fails.
        requests.Timeout: If the request exceeds ``REQUEST_TIMEOUT``.
    """
    process_url = "https://transkribus.eu/processing/v1/processes"
    headers = {"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"}
    process_data = {
        "image": {
            "base64": image_data,
        },
        "config": {"textRecognition": {"htrId": model_id}},
    }

    response = requests.post(process_url, headers=headers, data=json.dumps(process_data), timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    process_response = response.json()
    process_id = process_response["processId"]

    return process_id, headers


def check_process_status(process_id: str, headers: dict) -> dict:
    """
    Retrieves the current status of a Transkribus processing job.

    Polls the Transkribus API for the given process ID. The returned dict contains
    at minimum a ``status`` field (e.g. ``"RUNNING"``, ``"FINISHED"``, ``"ERROR"``).
    When the status is ``"FINISHED"``, the dict also contains a ``content`` key with
    the recognised text.

    Args:
        process_id: The process ID returned by :func:`start_recognition_process`.
        headers: The authorisation headers (Bearer token) for the request.

    Returns:
        A dict with the full status response from the Transkribus API.

    Raises:
        requests.HTTPError: If the status request fails.
        requests.Timeout: If the request exceeds ``REQUEST_TIMEOUT``.
    """
    status_url = f"https://transkribus.eu/processing/v1/processes/{process_id}"
    response = requests.get(status_url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    status_response = response.json()
    return status_response


def check_process_status_with_retry(process_id: str, headers: dict, username: str, password: str) -> tuple[dict, dict]:
    """
    Retrieves the status of a Transkribus job, refreshing the access token on 401 errors.

    Behaves like :func:`check_process_status` but transparently re-authenticates when
    the server returns HTTP 401 Unauthorized (e.g. because the token has expired) and
    retries the request once with the new token. Any other HTTP error is re-raised.

    Args:
        process_id: The process ID returned by :func:`start_recognition_process`.
        headers: The current authorisation headers (Bearer token) for the request.
        username: The Transkribus account username used to refresh the token.
        password: The Transkribus account password used to refresh the token.

    Returns:
        A tuple of (status_response, headers) where status_response is the full API
        response dict and headers is the (potentially refreshed) authorisation header
        dict to use for the next call.

    Raises:
        requests.HTTPError: If the status request fails with a non-401 error, or if
            the retry after token refresh also fails.
        requests.Timeout: If any request exceeds ``REQUEST_TIMEOUT``.
    """
    status_url = f"https://transkribus.eu/processing/v1/processes/{process_id}"

    try:
        response = requests.get(status_url, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        status_response = response.json()
        return status_response, headers
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            new_access_token = transkribus_login(username, password)
            new_headers = {"Authorization": f"Bearer {new_access_token}", "Content-Type": "application/json"}

            response = requests.get(status_url, headers=new_headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            status_response = response.json()
            return status_response, new_headers
        else:
            raise


def get_process_result(status_response: dict) -> str:
    """
    Extracts the recognised text from a finished Transkribus status response.

    Validates that the process has reached ``"FINISHED"`` status before extracting
    the result text. Raises if the process is in an error state or has not yet
    completed.

    Args:
        status_response: The status dict returned by :func:`check_process_status`
            or :func:`check_process_status_with_retry`.

    Returns:
        The plain text recognised from the image.

    Raises:
        RuntimeError: If the process status is ``"ERROR"`` or any status other than
            ``"FINISHED"``.
    """
    status = status_response.get("status")

    if status == "ERROR":
        raise RuntimeError("Transkribus-Verarbeitung fehlgeschlagen (Status: ERROR).")
    elif status != "FINISHED":
        raise RuntimeError(f"Transkribus-Prozess noch nicht abgeschlossen. Aktueller Status: {status}")

    recognized_text = status_response["content"]["text"]
    return recognized_text


def get_recognition_result(access_token: str, image_data: str, model_id: int) -> str:
    """
    Runs a full synchronous HTR recognition cycle for a single image.

    Submits the image to Transkribus, then polls the status endpoint with
    exponential back-off (capped at 30 s) until the job finishes, and finally
    returns the recognised text. Intended for use in scripts or contexts where
    async polling is not required.

    Args:
        access_token: A valid Transkribus OAuth access token.
        image_data: The image to recognise, encoded as a Base64 string.
        model_id: The numeric ID of the HTR model to use.

    Returns:
        The plain text recognised from the image.

    Raises:
        RuntimeError: If the Transkribus job fails (status ``"ERROR"``).
        requests.HTTPError: If any API request fails.
    """
    process_id, headers = start_recognition_process(access_token, image_data, model_id)
    poll_interval = 5

    while True:
        status_response = check_process_status(process_id, headers)
        status = status_response.get("status")

        if status == "FINISHED":
            return get_process_result(status_response)
        elif status == "ERROR":
            raise RuntimeError(f"Transkribus-Verarbeitung fehlgeschlagen (Prozess-ID: {process_id}).")

        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, 30)


def get_result_from_transkribus(image_data: str, model_id: int, mode: str) -> str:
    """
    Authenticates with Transkribus and returns the HTR result for a single image.

    Selects the Transkribus credentials for the given mode (``"RA"`` or ``"GL"``),
    logs in, and delegates to :func:`get_recognition_result` for the full
    recognition cycle.

    Args:
        image_data: The image to recognise, encoded as a Base64 string.
        model_id: The numeric ID of the HTR model to use.
        mode: Account selector — ``"RA"`` uses the RA credentials,
            ``"GL"`` uses the GL credentials.

    Returns:
        The plain text recognised from the image.

    Raises:
        RuntimeError: If the Transkribus job fails.
        requests.HTTPError: If any API request fails.
    """
    username = ""
    password = ""

    if mode == "RA":
        username = USERNAME_TRANSKRIBUS_RA
        password = PASSWORD_TRANSKRIBUS_RA
    elif mode == "GL":
        username = USERNAME_TRANSKRIBUS_GL
        password = PASSWORD_TRANSKRIBUS_GL

    access_token = transkribus_login(username, password)
    recognized_text = get_recognition_result(access_token, image_data, model_id)
    return recognized_text

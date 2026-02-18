import concurrent.futures
import threading
import time
import uuid

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render

from ocr_workflow.components.LLMs import (
    get_mllm_only_atr_result,
    get_merged_atr_mllm_result,
    get_tei_from_mllm,
)
from ocr_workflow.components.processing import (
    encode_image,
    remove_accent,
    remove_empty_lines,
    remove_first_empty_line,
    remove_specific_string,
    remove_xml_tags,
    strip_lines,
)
from ocr_workflow.components.transkribus_api import (
    check_process_status_with_retry,
    get_process_result,
    start_recognition_process,
    transkribus_login,
)
from ocr_workflow.settings import (
    USERNAME_TRANSKRIBUS_RA,
    PASSWORD_TRANSKRIBUS_RA,
    USERNAME_TRANSKRIBUS_GL,
    PASSWORD_TRANSKRIBUS_GL,
)

# Global dict to store process status
process_status_store = {}
process_status_lock = threading.Lock()
PROCESS_ENTRY_TTL = 3600  # Seconds until a completed entry is purged


def _cleanup_old_entries() -> None:
    """
    Removes process store entries that have exceeded the TTL.

    Iterates over ``process_status_store`` under the lock and deletes any entry
    whose ``created_at`` timestamp is older than ``PROCESS_ENTRY_TTL`` seconds.
    Should be called before adding a new entry to prevent unbounded memory growth.
    """
    cutoff = time.time() - PROCESS_ENTRY_TTL
    with process_status_lock:
        to_delete = [
            pid for pid, data in process_status_store.items()
            if data.get("created_at", 0) < cutoff
        ]
        for pid in to_delete:
            del process_status_store[pid]

VALID_MODES = {"RA", "GL"}
VALID_MODELS = {
    "gpt-5.2",
    "claude-sonnet-4-5-20250929",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
}
VALID_PROMPTS = {"prompt-tei-gl", "prompt-tei-ra", "prompt-tei-custom"}
MAX_PAGES = 10
MAX_MLLM_WORKERS = 10
TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0


def _start_transkribus_jobs(
    images_base64: list[str],
    transkribus_model: int,
    mode: str,
) -> tuple[list[str], dict, str, str]:
    """
    Logs in to Transkribus and submits all images as parallel HTR jobs.

    Args:
        images_base64: Base64-encoded images to submit.
        transkribus_model: Numeric HTR model ID.
        mode: Account selector (``"RA"`` or ``"GL"``).

    Returns:
        A tuple of (process_ids, headers, username, password) where process_ids is
        the list of Transkribus process IDs (one per image), headers is the auth
        header dict for subsequent status calls, and username/password are kept for
        token refresh in :func:`_poll_transkribus_jobs`.
    """
    username = USERNAME_TRANSKRIBUS_RA if mode == "RA" else USERNAME_TRANSKRIBUS_GL
    password = PASSWORD_TRANSKRIBUS_RA if mode == "RA" else PASSWORD_TRANSKRIBUS_GL
    access_token = transkribus_login(username, password)

    process_ids = []
    headers: dict = {}
    for image_data in images_base64:
        pid, headers = start_recognition_process(access_token, image_data, transkribus_model)
        process_ids.append(pid)

    return process_ids, headers, username, password


def _poll_transkribus_jobs(
    process_ids: list[str],
    headers: dict,
    username: str,
    password: str,
    process_id: str,
    total_pages: int,
) -> list[str]:
    """
    Polls all Transkribus jobs until every page has finished and returns the results.

    Uses exponential back-off (starting at 5 s, capped at 30 s) between polling
    rounds. Writes progress updates to ``process_status_store``.

    Args:
        process_ids: Transkribus process IDs to poll (one per page).
        headers: Auth headers for the Transkribus API.
        username: Transkribus username (used to refresh a stale token).
        password: Transkribus password (used to refresh a stale token).
        process_id: The job UUID stored in ``process_status_store``.
        total_pages: Total number of pages being processed.

    Returns:
        A list of recognised text strings, one per page, in submission order.

    Raises:
        RuntimeError: If any individual Transkribus job reports an error status.
    """
    results: list[str | None] = [None] * total_pages
    finished_count = 0
    poll_interval = 5

    while finished_count < total_pages:
        time.sleep(poll_interval)
        poll_interval = min(poll_interval * 1.5, 30)

        for page_index, pid in enumerate(process_ids):
            if results[page_index] is not None:
                continue  # Already finished

            status_response, headers = check_process_status_with_retry(
                pid, headers, username, password
            )
            transkribus_status = status_response.get("status")

            if transkribus_status == "FINISHED":
                results[page_index] = get_process_result(status_response)
                finished_count += 1
                process_status_store[process_id]["status"] = (
                    f"Texerkennung bei {finished_count} von {total_pages} Seite(n) abgeschlossen."
                )
            elif transkribus_status == "ERROR":
                process_status_store[process_id]["status"] = (
                    f"Fehler bei der Texterkennung bei Seite {page_index + 1}"
                )
                process_status_store[process_id]["transkribus_status"] = "ERROR"
                raise RuntimeError(f"Transkribus processing failed for page {page_index + 1}")

    return results  # type: ignore[return-value]


def _merge_page_results(
    images_base64: list[str],
    transkribus_results: list[str],
    mllm_results: list[str],
    model: str,
    temperature: float,
    mode: str,
    process_id: str,
) -> list[dict]:
    """
    Merges Transkribus and MLLM transcriptions for every page.

    Calls :func:`get_merged_atr_mllm_result` for each page and returns a list of
    per-page result dicts ready to be stored in ``process_status_store``.

    Args:
        images_base64: Base64-encoded source images (one per page).
        transkribus_results: HTR results from Transkribus (one per page).
        mllm_results: Standalone MLLM transcriptions (one per page).
        model: Multimodal model name used for merging.
        temperature: Sampling temperature for the merge call.
        mode: Account selector (``"RA"`` or ``"GL"``).
        process_id: The job UUID stored in ``process_status_store``.

    Returns:
        A list of dicts, each with keys ``page``, ``mllm_merged_result``,
        ``ocr_engine_result``, and ``mllm_only_result``.
    """
    process_status_store[process_id]["status"] = "Ergebnisse für alle Seiten werden zusammengeführt..."
    page_results = []

    for page_index in range(len(images_base64)):
        merged = get_merged_atr_mllm_result(
            model,
            transkribus_results[page_index],
            mllm_results[page_index],
            images_base64[page_index],
            temperature,
            mode,
        )
        page_results.append({
            "page": page_index,
            "mllm_merged_result": merged,
            "ocr_engine_result": transkribus_results[page_index],
            "mllm_only_result": mllm_results[page_index],
        })

    return page_results


def atr_workflow(request: HttpRequest) -> HttpResponse:
    """
    Renders the main ATR Workflow page.

    This view serves the entry point of the application. It renders the HTML
    template that contains the upload form, model selectors, and result panels.

    Args:
        request: The incoming HTTP request.

    Returns:
        An HttpResponse containing the rendered ``ATR_Workflow.html`` template.
    """

    return render(request, "main/ATR_Workflow.html")


def upload_image(request: HttpRequest) -> JsonResponse:
    """
    Accepts uploaded images and starts a background OCR processing job.

    Validates all POST parameters (model, Transkribus model ID, temperature, mode,
    page count), encodes the uploaded images as Base64, then spawns a background
    thread that runs Transkribus HTR and the multimodal LLM in parallel for each
    page and finally merges the results.

    The response is returned immediately with a ``process_id`` that the client can
    use to poll :func:`check_status` for progress and results.

    Expected POST parameters:
        - ``images``: One or more image files (max ``MAX_PAGES``).
        - ``multimodal-llm-ocr``: Model name from ``VALID_MODELS``.
        - ``transkribus-model``: Integer HTR model ID.
        - ``temperature-ocr``: Float in [``TEMPERATURE_MIN``, ``TEMPERATURE_MAX``].
        - ``mode``: ``"RA"`` or ``"GL"``.

    Args:
        request: The incoming HTTP request.

    Returns:
        A JsonResponse with ``{"process_id": "<uuid>"}`` on success, or
        ``{"error": "<message>"}`` with an appropriate HTTP status code on failure.
    """
    # Only accept POST requests that include at least one image
    if request.method == "POST" and request.FILES.getlist("images"):
        # Multimodal model to use for text recognition
        multimodal_llm_ocr = request.POST.get("multimodal-llm-ocr")
        if multimodal_llm_ocr not in VALID_MODELS:
            return JsonResponse({"error": f"Ungültiges Modell: {multimodal_llm_ocr}"}, status=400)

        # Transkribus HTR model ID – must be cast to int
        try:
            transkribus_model = int(request.POST.get("transkribus-model"))
        except (TypeError, ValueError):
            return JsonResponse({"error": "Ungültiges Transkribus-Modell"}, status=400)

        # Temperature for OCR
        try:
            temperature_ocr = float(request.POST.get("temperature-ocr", "0.0"))
        except ValueError:
            return JsonResponse({"error": "Ungültiger Temperature-Wert"}, status=400)
        if not (TEMPERATURE_MIN <= temperature_ocr <= TEMPERATURE_MAX):
            return JsonResponse({"error": f"Temperature muss zwischen {TEMPERATURE_MIN} und {TEMPERATURE_MAX} liegen"}, status=400)

        # Retrieve uploaded images
        images = request.FILES.getlist("images")
        if len(images) > MAX_PAGES:
            return JsonResponse({"error": f"Maximal {MAX_PAGES} Seiten erlaubt"}, status=400)

        # Retrieve the current mode
        mode = request.POST.get("mode")
        if mode not in VALID_MODES:
            return JsonResponse({"error": f"Ungültiger Modus: {mode}"}, status=400)

        # Encode all images to base64 before starting background thread
        # This prevents "I/O operation on closed file" errors
        images_base64 = []
        for image in images:
            images_base64.append(encode_image(image))

        # Generate unique process ID
        process_id = str(uuid.uuid4())

        # Initialize status; cleanup old entries beforehand
        _cleanup_old_entries()
        total_pages = len(images_base64)
        with process_status_lock:
            process_status_store[process_id] = {
                "status": "Texterkennung gestartet",
                "transkribus_status": "PENDING",
                "mllm_status": "PENDING",
                "total_pages": total_pages,
                "current_page": 0,
                "results": [],
                "created_at": time.time(),
            }

        # Background processing function
        def process_ocr():
            try:
                # Submit all images to Transkribus and start MLLM jobs in parallel
                process_status_store[process_id]["status"] = f"Texterkennung für {total_pages} Seite(n) wird gestartet..."
                transkribus_pids, headers, username, password = _start_transkribus_jobs(
                    images_base64, transkribus_model, mode
                )

                process_status_store[process_id]["status"] = f"Texterkennung für alle {total_pages} Seite(n) läuft parallel..."
                process_status_store[process_id]["transkribus_status"] = "RUNNING"

                with concurrent.futures.ThreadPoolExecutor(max_workers=min(total_pages, MAX_MLLM_WORKERS)) as executor:
                    mllm_futures = [
                        executor.submit(get_mllm_only_atr_result, multimodal_llm_ocr, image_data, temperature_ocr, mode)
                        for image_data in images_base64
                    ]

                    transkribus_results = _poll_transkribus_jobs(
                        transkribus_pids, headers, username, password, process_id, total_pages
                    )

                    process_status_store[process_id]["status"] = "Auf Ergebnisse des multimodalen Modells warten..."
                    mllm_results = [future.result() for future in mllm_futures]

                process_status_store[process_id]["transkribus_status"] = "FINISHED"
                process_status_store[process_id]["mllm_status"] = "FINISHED"

                # Merge and store per-page results
                page_results = _merge_page_results(
                    images_base64, transkribus_results, mllm_results,
                    multimodal_llm_ocr, temperature_ocr, mode, process_id
                )
                process_status_store[process_id]["results"].extend(page_results)

                process_status_store[process_id]["status"] = "Verarbeitung abgeschlossen"
                process_status_store[process_id]["transkribus_status"] = "FINISHED"
                process_status_store[process_id]["mllm_status"] = "FINISHED"

            except Exception as e:
                process_status_store[process_id]["status"] = f"Fehler: {str(e)}"
                process_status_store[process_id]["error"] = str(e)

        # Start background thread
        thread = threading.Thread(target=process_ocr)
        thread.start()

        # Return process ID immediately
        return JsonResponse({"process_id": process_id})

    return JsonResponse({"error": "POST-Anfrage mit Bildern erforderlich"}, status=400)


def create_tei(request: HttpRequest) -> JsonResponse:
    """
    Accepts recognised text and images and starts a background TEI-XML generation job.

    Validates all POST parameters, encodes the uploaded image(s) as Base64, then
    spawns a background thread that calls the multimodal LLM to produce TEI-XML.
    Also derives a plain-text comparison version of the result by stripping XML tags
    and normalising whitespace.

    Supports both single-page and multi-page documents:
    - Single-page: send ``image`` (one file) without ``num_pages``.
    - Multi-page: send ``image_0``, ``image_1``, … together with ``num_pages``.

    The response is returned immediately with a "process_id" that the client can
    use to poll :func:`check_status` for progress and the final TEI-XML result.

    Expected POST parameters:
        - ``multimodal-llm-tei``: Model name from ``VALID_MODELS``.
        - ``prompt-transformation-tei``: Prompt key from ``VALID_PROMPTS``.
        - ``custom-prompt-text``: Optional free-text prompt (used when prompt key is
          ``"prompt-tei-custom"``).
        - ``merged_text``: The recognised text to transform into TEI-XML.
        - ``temperature-tei``: Float in [``TEMPERATURE_MIN``, ``TEMPERATURE_MAX``].
        - ``mode``: ``"RA"`` or ``"GL"``.
        - ``image`` or ``image_0`` … ``image_N`` + ``num_pages``: The source image(s).

    Args:
        request: The incoming HTTP request.

    Returns:
        A JsonResponse with ``{"process_id": "<uuid>"}`` on success, or
        ``{"error": "<message>"}`` with an appropriate HTTP status code on failure.
    """
    # Only accept POST requests
    if request.method == "POST":
        # Multimodal model to use for TEI-XML generation
        multimodal_llm_tei = request.POST.get("multimodal-llm-tei")
        if multimodal_llm_tei not in VALID_MODELS:
            return JsonResponse({"error": f"Ungültiges Modell: {multimodal_llm_tei}"}, status=400)

        # Prompt type to use for the TEI-XML transformation
        chosen_prompt = request.POST.get("prompt-transformation-tei")
        if chosen_prompt not in VALID_PROMPTS:
            return JsonResponse({"error": f"Ungültiger Prompt: {chosen_prompt}"}, status=400)

        # Optional custom prompt text
        custom_prompt_text = request.POST.get("custom-prompt-text", "")

        # Temperature for TEI generation
        try:
            temperature_tei = float(request.POST.get("temperature-tei", "0.0"))
        except ValueError:
            return JsonResponse({"error": "Ungültiger Temperature-Wert"}, status=400)
        if not (TEMPERATURE_MIN <= temperature_tei <= TEMPERATURE_MAX):
            return JsonResponse({"error": f"Temperature muss zwischen {TEMPERATURE_MIN} und {TEMPERATURE_MAX} liegen"}, status=400)

        # Text selected in the right comparison panel
        merged_text = request.POST.get("merged_text")
        if not merged_text:
            return JsonResponse({"error": "Kein Text übergeben"}, status=400)

        # Retrieve the current mode
        mode = request.POST.get("mode", "RA")
        if mode not in VALID_MODES:
            return JsonResponse({"error": f"Ungültiger Modus: {mode}"}, status=400)

        # Number of pages (for multi-page documents)
        num_pages = request.POST.get("num_pages")

        # Encode images before starting background thread
        images_base64 = []
        is_multipage = False

        if num_pages:
            # Multi-page document: collect all images
            try:
                num_pages = int(num_pages)
            except ValueError:
                return JsonResponse({"error": "Ungültige Seitenanzahl"}, status=400)
            if not (1 <= num_pages <= MAX_PAGES):
                return JsonResponse({"error": f"Seitenanzahl muss zwischen 1 und {MAX_PAGES} liegen"}, status=400)
            is_multipage = True

            for i in range(num_pages):
                image_key = f"image_{i}"
                if image_key in request.FILES:
                    image = request.FILES[image_key]
                    base64_part = encode_image(image)
                    images_base64.append(base64_part)
                else:
                    return JsonResponse({"error": f"Bild {i} fehlt"}, status=400)
        else:
            # Single-page document
            if "image" in request.FILES:
                image = request.FILES["image"]
                base64_part = encode_image(image)
                images_base64.append(base64_part)
            else:
                return JsonResponse({"error": "Kein Bild hochgeladen"}, status=400)

        # Generate unique process ID
        process_id = str(uuid.uuid4())

        # Initialize status; cleanup old entries beforehand
        _cleanup_old_entries()
        with process_status_lock:
            process_status_store[process_id] = {
                "status": "TEI-XML-Generierung gestartet",
                "tei_status": "RUNNING",
                "created_at": time.time(),
            }

        # Background processing function
        def process_tei():
            try:
                process_status_store[process_id]["status"] = "TEI-XML wird generiert..."

                if is_multipage:
                    # Fetch TEI-XML from the multimodal LLM (all images at once)
                    result_tei = get_tei_from_mllm(multimodal_llm_tei, images_base64, merged_text, chosen_prompt, custom_prompt_text, temperature_tei, mode)
                else:
                    # Fetch TEI-XML from the multimodal LLM
                    result_tei = get_tei_from_mllm(multimodal_llm_tei, images_base64[0], merged_text, chosen_prompt, custom_prompt_text, temperature_tei, mode)

                # Remove the leading empty line from the displayed TEI-XML
                result_tei = remove_first_empty_line(result_tei)

                # Build a plain-text comparison version of the content
                result_text_content_only = remove_first_empty_line(
                    remove_specific_string(
                        strip_lines(
                            remove_empty_lines(
                                remove_accent(
                                    remove_xml_tags(result_tei)
                                )
                            )
                        )
                    )
                )
                process_status_store[process_id]["status"] = "Verarbeitung abgeschlossen"
                process_status_store[process_id]["tei_status"] = "FINISHED"
                process_status_store[process_id]["result_tei"] = result_tei
                process_status_store[process_id]["result_text_content_only"] = result_text_content_only

            except Exception as e:
                process_status_store[process_id]["status"] = f"Fehler: {str(e)}"
                process_status_store[process_id]["error"] = str(e)
                process_status_store[process_id]["tei_status"] = "ERROR"

        # Start background thread
        thread = threading.Thread(target=process_tei)
        thread.start()

        # Return process ID immediately
        return JsonResponse({"process_id": process_id})

    return JsonResponse({"error": "POST-Anfrage erforderlich"}, status=405)


def check_status(request: HttpRequest) -> JsonResponse:
    """
    Returns the current status and results of a background processing job.

    The client polls this endpoint repeatedly after calling :func:`upload_image` or
    :func:`create_tei`. The response mirrors the entry stored in
    ``process_status_store`` for the given ``process_id``, with the internal
    ``created_at`` timestamp removed.

    Expected GET parameters:
        - ``process_id``: The UUID returned by the originating view.

    Args:
        request: The incoming HTTP request.

    Returns:
        A JsonResponse with the full status dict on success (keys include at minimum
        ``status`` and vary by job type), or ``{"error": "<message>"}`` with HTTP 404
        if the process ID is unknown, or HTTP 405 if the request method is not GET.
    """
    if request.method == "GET":
        process_id = request.GET.get("process_id")

        with process_status_lock:
            if process_id in process_status_store:
                status_data = dict(process_status_store[process_id])
            else:
                status_data = None

        if status_data is not None:
            status_data.pop("created_at", None)
            return JsonResponse(status_data)
        else:
            return JsonResponse({"error": "Prozess nicht gefunden"}, status=404)

    return JsonResponse({"error": "GET-Anfrage erforderlich"}, status=405)



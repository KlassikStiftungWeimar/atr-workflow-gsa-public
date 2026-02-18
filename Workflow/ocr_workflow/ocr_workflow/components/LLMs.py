import base64

import anthropic
from google import genai
from google.genai import types
from openai import OpenAI

from ocr_workflow.components.processing import *
from ocr_workflow.settings import (
    OPENAI_KEY_RA, OPENAI_KEY_GL,
    ANTHROPIC_KEY_RA, ANTHROPIC_KEY_GL,
    GOOGLE_KEY_RA, GOOGLE_KEY_GL,
    BASE_DIR
)

OPENAI_MODELS = {"gpt-5.2"}
ANTHROPIC_MODELS = {"claude-sonnet-4-5-20250929"}
GOOGLE_MODELS = {"gemini-2.5-pro", "gemini-3-pro-preview"}

ANTHROPIC_MAX_TOKENS = 10000
GOOGLE_MAX_OUTPUT_TOKENS = 20000

# TEI example files loaded once at module start
_tei_dir = BASE_DIR / "main" / "static" / "TEI-XMLs"
_TEI_GL_EXAMPLE_1 = (_tei_dir / "349305_97.xml").read_text(encoding="utf-8")
_TEI_GL_EXAMPLE_2 = (_tei_dir / "349305_102.xml").read_text(encoding="utf-8")
_TEI_GL_EXAMPLE_3 = (_tei_dir / "349305_219.xml").read_text(encoding="utf-8")
_TEI_RA_EXAMPLE_1 = (_tei_dir / "RA282_tei.xml").read_text(encoding="utf-8")
_TEI_RA_EXAMPLE_2 = (_tei_dir / "RA765_tei.xml").read_text(encoding="utf-8")
_TEI_RA_EXAMPLE_3 = (_tei_dir / "RA5845_tei.xml").read_text(encoding="utf-8")

# Module-level clients – created once and reused (connection pooling)
_openai_clients = {
    "RA": OpenAI(api_key=OPENAI_KEY_RA),
    "GL": OpenAI(api_key=OPENAI_KEY_GL),
}
_anthropic_clients = {
    "RA": anthropic.Anthropic(api_key=ANTHROPIC_KEY_RA),
    "GL": anthropic.Anthropic(api_key=ANTHROPIC_KEY_GL),
}
_google_clients = {
    "RA": genai.Client(api_key=GOOGLE_KEY_RA),
    "GL": genai.Client(api_key=GOOGLE_KEY_GL),
}


def _openai_image_block(image_type: str, image_data: str) -> dict:
    """
    Builds an OpenAI image content block from Base64 image data.

    Args:
        image_type (str): The image MIME subtype (e.g. "jpeg" or "png").
        image_data (str): The Base64-encoded image data.

    Returns:
        dict: An OpenAI-compatible image_url content block.
    """
    return {"type": "image_url", "image_url": {"url": f"data:image/{image_type};base64,{image_data}"}}


def _anthropic_image_block(image_type: str, image_data: str) -> dict:
    """
    Builds an Anthropic image content block from Base64 image data.

    Args:
        image_type (str): The image MIME subtype (e.g. "jpeg" or "png").
        image_data (str): The Base64-encoded image data.

    Returns:
        dict: An Anthropic-compatible image content block.
    """
    return {"type": "image", "source": {"type": "base64", "media_type": f"image/{image_type}", "data": image_data}}


def _google_image_part(image_type: str, image_data: str) -> types.Part:
    """
    Builds a Google Gemini image Part from Base64 image data.

    Args:
        image_type (str): The image MIME subtype (e.g. "jpeg" or "png").
        image_data (str): The Base64-encoded image data.

    Returns:
        types.Part: A Gemini-compatible image Part object.
    """
    return types.Part.from_bytes(data=base64.b64decode(image_data), mime_type=f"image/{image_type}")


def get_result_from_openai(messages: list, model: str, temperature: float, mode: str = "RA") -> str:
    """
    Calls the OpenAI API and returns the model's response.

    Args:
        messages (list): The list of messages to send to the model.
        model (str): The model name to use.
        temperature (float): The sampling temperature.
        mode (str): Team mode ("RA" or "GL") to select the API key.

    Returns:
        str: The content of the first response message.
    """
    completion = _openai_clients[mode].chat.completions.create(model=model, messages=messages, temperature=temperature)
    return completion.choices[0].message.content


def get_result_from_anthropic(messages: list, model: str, system_prompt: str, temperature: float, mode: str = "RA") -> str:
    """
    Calls the Anthropic Claude API and returns the model's response.

    Args:
        messages (list): The list of messages to send to the model.
        model (str): The model name to use.
        system_prompt (str): The system prompt for the model.
        temperature (float): The sampling temperature.
        mode (str): Team mode ("RA" or "GL") to select the API key.

    Returns:
        str: The content of the first response message.
    """
    completion = _anthropic_clients[mode].messages.create(
        model=model, messages=messages, system=system_prompt, max_tokens=ANTHROPIC_MAX_TOKENS, temperature=temperature
    )
    return completion.content[0].text


def get_result_from_google(messages: list, model: str, system_prompt: str, temperature: float, mode: str = "RA") -> str:
    """
    Calls the Google Gemini API and returns the model's response.

    Handles MAX_TOKENS and RECITATION finish reasons explicitly and falls back to
    candidate-level content extraction when response.text is unavailable.

    Args:
        messages (list): The list of content parts to send to the model.
        model (str): The model name to use.
        system_prompt (str): The system instruction for the model.
        temperature (float): The sampling temperature.
        mode (str): Team mode ("RA" or "GL") to select the API key.

    Returns:
        str: The text content of the model's response.

    Raises:
        ValueError: If the response hits MAX_TOKENS, is blocked due to RECITATION,
                    or contains no extractable text.
    """
    config_kwargs = dict(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=GOOGLE_MAX_OUTPUT_TOKENS,
    )
    if model == "gemini-3-pro-preview":
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="low")

    response = _google_clients[mode].models.generate_content(
        model=model,
        contents=messages,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    # Try to get text from response.text first
    if hasattr(response, 'text') and response.text:
        return response.text

    # Fallback: Try to extract from candidates if response.text is None
    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]

        # Check for MAX_TOKENS finish reason
        if hasattr(candidate, 'finish_reason'):
            from google.genai.types import FinishReason
            if candidate.finish_reason == FinishReason.MAX_TOKENS:
                # Try to extract partial content
                if candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text is not None:
                            text_parts.append(part.text)

                    if text_parts:
                        partial_text = ''.join(text_parts)
                        raise ValueError(
                            f"Gemini API reached the MAX_TOKENS limit. "
                            f"Model: {model}. "
                            f"Partial response available ({len(partial_text)} characters). "
                            f"Possible solutions: 1) reduce image size/complexity, "
                            f"2) simplify the prompt, or "
                            f"3) split into smaller sections. "
                            f"Partial content: {partial_text[:500]}..."
                        )

                # No partial content available
                raise ValueError(
                    f"Gemini API reached the MAX_TOKENS limit without returning any content. "
                    f"Model: {model}. "
                    f"The input may be too complex for this model."
                )

        # Normal extraction for other cases (including RECITATION)
        from google.genai.types import FinishReason
        if candidate.finish_reason == FinishReason.RECITATION:
            if candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
                text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
                if text_parts:
                    return ''.join(text_parts)
            raise ValueError(
                f"Gemini API returned finish reason RECITATION (no extractable content). "
                f"Model: {model}. The response was blocked due to copyright concerns."
            )

        if candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
            if text_parts:
                return ''.join(text_parts)

    # If we get here, no text was found
    raise ValueError(
        f"Gemini API returned no text content. "
        f"Model: {model}, "
        f"candidates: {len(response.candidates) if hasattr(response, 'candidates') and response.candidates else 0}."
    )


def get_result_from_mllm(messages: list, model: str, system_prompt: str | None, temperature: float, mode: str = "RA") -> str | None:
    """
    Routes a request to the appropriate LLM provider based on the model name.

    Args:
        messages (list): The list of messages to send to the model.
        model (str): The model name used to determine the provider.
        system_prompt (str | None): An optional system prompt used by Anthropic and Google.
                                    Not used for OpenAI (embedded in messages instead).
        temperature (float): The sampling temperature.
        mode (str): Team mode ("RA" or "GL") to select the API key.

    Returns:
        str | None: The model's response text, or None if no provider matched.
    """
    if model in OPENAI_MODELS:
        return get_result_from_openai(messages, model, temperature, mode)
    elif model in ANTHROPIC_MODELS:
        return get_result_from_anthropic(messages, model, system_prompt, temperature, mode)
    elif model in GOOGLE_MODELS:
        return get_result_from_google(messages, model, system_prompt, temperature, mode)


def get_mllm_only_atr_result(model: str, image_data_base64: str, temperature: float, mode: str = "RA") -> str:
    """
    Sends an image to a multimodal LLM and requests a verbatim transcription of the text.

    Args:
        model (str): The multimodal LLM model to use.
        image_data_base64 (str): Base64-encoded image data containing handwritten text.
        temperature (float): The sampling temperature.
        mode (str): Team mode ("RA" or "GL") to select the API key.

    Returns:
        str: The transcription with empty lines and leading/trailing whitespace removed.

    Raises:
        ValueError: If the model returns None.
    """
    image_type = get_image_type(image_data_base64)
    system_prompt = "You are an expert in transcribing 18th- and 19th-century handwritten texts in German and French. Your task is to produce a faithful transcription of the provided text."
    user_text = (
        "Instructions:\n"
        "1. Provide only the transcript, verbatim.\n"
        "2. Preserve all original line breaks and punctuation.\n"
        "3. Reproduce archaic spelling exactly as is.\n"
        "4. Do not make up words. Transcribe them exactly as they appear — even if they look nonsensical.\n"
        "5. Do not add commentary, Markdown, or formatting."
    )

    system_prompt_outside_message = None
    if model in OPENAI_MODELS:
        message_mllm_atr_only = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                _openai_image_block(image_type, image_data_base64),
            ]},
        ]
    elif model in ANTHROPIC_MODELS:
        system_prompt_outside_message = system_prompt
        message_mllm_atr_only = [
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                _anthropic_image_block(image_type, image_data_base64),
            ]}
        ]
    elif model in GOOGLE_MODELS:
        system_prompt_outside_message = system_prompt
        message_mllm_atr_only = [
            _google_image_part(image_type, image_data_base64),
            user_text,
        ]

    mllm_only_atr_result = get_result_from_mllm(message_mllm_atr_only, model, system_prompt_outside_message,
                                                temperature, mode)

    if mllm_only_atr_result is None:
        raise ValueError(f"Model {model} returned None.")

    improved_result = remove_empty_lines(mllm_only_atr_result)
    improved_result = strip_lines(improved_result)

    return improved_result


def get_merged_atr_mllm_result(model: str, ocr_engine_result: str, mllm_only_atr_result: str, image_data_base64: str, temperature: float,
                               mode: str = "RA") -> str:
    """
    Merges the results of an OCR engine and a multimodal LLM into an improved transcription.

    The model is shown both transcriptions alongside the original image and asked to
    select the version that best matches the image for each differing passage.

    Args:
        model (str): The multimodal LLM model to use.
        ocr_engine_result (str): The transcription produced by the OCR engine.
        mllm_only_atr_result (str): The transcription produced by the multimodal LLM.
        image_data_base64 (str): Base64-encoded image data of the handwritten text.
        temperature (float): The sampling temperature.
        mode (str): Team mode ("RA" or "GL") to select the API key.

    Returns:
        str: The merged transcription with empty lines and leading/trailing whitespace removed.

    Raises:
        ValueError: If the model returns None.
    """
    image_type = get_image_type(image_data_base64)
    system_prompt = "You are an expert in transcribing 18th- and 19th-century handwritten texts in German and French. Your task is to produce a faithful transcription of the provided text."
    merge_text = (
        "I provide you with a transcript from a Handwritten Text Recognition tool:"
        f"\n\n~~~~~~\n{ocr_engine_result}\n~~~~~~\n"
        "And with a transcript created by a multimodal LLM:"
        f"\n~~~~~~\n{mllm_only_atr_result}\n~~~~~~\n\n"
        "Identify the differences between the two text inputs and check these passages against "
        "the images. Select the text version that best conforms to the images. "
        "Provide only the transcript without any markdown or comments and "
        "make sure to keep the original line breaks."
    )

    system_prompt_outside_message = None
    if model in OPENAI_MODELS:
        message_merge_mllm = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": merge_text},
                _openai_image_block(image_type, image_data_base64),
            ]},
        ]
    elif model in ANTHROPIC_MODELS:
        system_prompt_outside_message = system_prompt
        message_merge_mllm = [
            {"role": "user", "content": [
                {"type": "text", "text": merge_text},
                _anthropic_image_block(image_type, image_data_base64),
            ]}
        ]
    elif model in GOOGLE_MODELS:
        system_prompt_outside_message = system_prompt
        message_merge_mllm = [
            _google_image_part(image_type, image_data_base64),
            merge_text,
        ]

    merged_mllm_result = get_result_from_mllm(message_merge_mllm, model, system_prompt_outside_message, temperature,
                                              mode)

    if merged_mllm_result is None:
        raise ValueError(
            f"Model {model} returned None during merge.")

    improved_result = remove_empty_lines(merged_mllm_result)
    improved_result = strip_lines(improved_result)

    return improved_result


def get_tei_from_mllm(model: str, image_data: str | list[str], merged_text: str, chosen_prompt: str, custom_prompt_text: str, temperature: float, mode: str = "RA") -> str:
    """
    Generates TEI-XML annotations for a transcribed text using a multimodal LLM.

    The model receives the transcribed text together with the original image(s) and
    annotates the text according to the selected prompt type.

    Args:
        model (str): The multimodal LLM model to use.
        image_data (str | list[str]): Base64-encoded image data. Either a single image
                                      (str) or a list of images for multi-page documents.
        merged_text (str): The transcribed text to annotate.
        chosen_prompt (str): The prompt type to use ("prompt-tei-gl", "prompt-tei-ra",
                             or "prompt-tei-custom").
        custom_prompt_text (str): Custom prompt text (only used for "prompt-tei-custom").
        temperature (float): The sampling temperature.
        mode (str): Team mode ("RA" or "GL") to select the API key.

    Returns:
        str: The TEI-XML markup with Markdown code fences and empty lines removed.
    """
    prompt_text = ""

    # Support for single and multiple images
    if isinstance(image_data, list):
        image_data_list = image_data
        image_type = get_image_type(image_data_list[0])
        num_pages = len(image_data_list)
    else:
        image_data_list = [image_data]
        image_type = get_image_type(image_data)
        num_pages = 1

    system_prompt_outside_message = None
    if chosen_prompt == "prompt-tei-gl":
        prompt_text = (
            "You are provided with a transcription of a handwritten poem from the 18th and 19th centuries. "
            "Annotate it in TEI-XML using these three examples:\n"
            "~~~~~~\nExample 1:"
            f"\n{_TEI_GL_EXAMPLE_1}\n~~~~~~\n"
            "~~~~~~\nExample 2:"
            f"\n{_TEI_GL_EXAMPLE_2}\n~~~~~~\n"
            "~~~~~~\nExample 3:"
            f"\n{_TEI_GL_EXAMPLE_3}\n~~~~~~\n"
            "Ignore the teiHeader. "
            "Use the Image as a reference for the correct annotation. "
            "Do not change the content of the text and do not return anything besides the xml."
        )
    elif chosen_prompt == "prompt-tei-ra":

        images_info = f"The document consists of {num_pages} image(s) provided in order. " if num_pages > 1 else ""
        pb_instruction = 'Insert <pb n="X"/> tags to     mark the start of each new image in sequential order (n="1" for first image, n="2" for second image, etc.)' if num_pages > 1 else 'The document has only one image'
        prompt_text = (
            f"You are provided with a transcription of a handwritten letter from the 18th and 19th centuries. "
            f"{images_info}"
            "Annotate it in TEI-XML using ONLY these allowed tags and structure:\n\n"
            "<text>\n"
            "<front>\n"
            '<div type="address"><note type="editorial">hat keine Adresse, ist geprüft</note></div>\n'
            "</front>\n"
            "<body>\n"
            '<div type="letter">\n'
            '<pb n="1"/>\n'
            "<opener>\n"
            "<dateline><lb/><placeName>Location</placeName> den\n"
            '<lb/><date when="YYYY-MM-DD">Date</date></dateline>\n'
            "<salute><lb/>Salutation</salute>\n"
            "</opener>\n"
            "<p>\n"
            "<lb/>Text\n"
            '<lb/><hi rendition="#latin">Latin text</hi>\n'
            '<g ref="#typoHyphen"/> (for hyphenation marks)\n'
            '<lb break="no"/> (for line breaks within a word)\n'
            "</p>\n"
            "<closer>\n"
            "<salute><lb/>Closing formula</salute>\n"
            "<signed><lb/>Signature</signed>\n"
            "</closer>\n"
            '<pb n="2"/>\n'
            "(Add additional <pb n=\"3\"/>, <pb n=\"4\"/>, etc. for each subsequent image in order)\n"
            "</div>\n"
            "</body>\n"
            "</text>\n\n"
            "Follow these examples precisely:\n"
            "~~~~~~\nExample 1:"
            f"\n{_TEI_RA_EXAMPLE_1}\n~~~~~~\n"
            "~~~~~~\nExample 2:"
            f"\n{_TEI_RA_EXAMPLE_2}\n~~~~~~\n"
            "~~~~~~\nExample 3:"
            f"\n{_TEI_RA_EXAMPLE_3}\n~~~~~~\n"
            "IMPORTANT RULES:\n"
            "1. Ignore the teiHeader\n"
            "2. Use ONLY the tags listed above\n"
            "3. Do not modify the text content\n"
            f"4. Use the image{'s' if num_pages > 1 else ''} as reference for correct annotation in the order provided\n"
            "5. Return only the TEI-XML markup, nothing else\n"
            "6. Place <lb/> tags at the beginning of each line\n"
            '7. Use <g ref="#typoHyphen"/> for hyphenation marks and <lb break="no"/> for continuing words\n'
            f"8. {pb_instruction}"
        )
    elif chosen_prompt == "prompt-tei-custom" and custom_prompt_text:
        prompt_text = custom_prompt_text

    system_prompt = "You are an expert in transcribing 18th- and 19th-century handwritten texts in German and French. Your task is to produce a faithful annotation in TEI-XML of the provided text."
    tei_content = f"{prompt_text}\n~~~~~~\n{merged_text}\n~~~~~~\n"

    system_prompt_outside_message = None
    if model in OPENAI_MODELS:
        content = [{"type": "text", "text": tei_content}]
        for img_data in image_data_list:
            content.append(_openai_image_block(image_type, img_data))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
    elif model in ANTHROPIC_MODELS:
        system_prompt_outside_message = system_prompt
        content = [{"type": "text", "text": tei_content}]
        for img_data in image_data_list:
            content.append(_anthropic_image_block(image_type, img_data))
        messages = [{"role": "user", "content": content}]
    elif model in GOOGLE_MODELS:
        system_prompt_outside_message = system_prompt
        content = [_google_image_part(image_type, img_data) for img_data in image_data_list]
        content.append(types.Part.from_text(text=tei_content))
        messages = content

    result = get_result_from_mllm(messages, model, system_prompt_outside_message, temperature, mode)
    cleaned_result = remove_accent(result)
    cleaned_result = remove_empty_lines(cleaned_result)

    return cleaned_result
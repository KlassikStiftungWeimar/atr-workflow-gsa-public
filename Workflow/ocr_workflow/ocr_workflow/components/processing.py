import base64
import re


def encode_image(image_file) -> str:
    """
    Encodes an image file to a Base64 string.

    Args:
        image_file (file): The image file to encode.

    Returns:
        str: The Base64-encoded image as a string.
    """
    image_content = image_file.read()
    encoded_image = base64.b64encode(image_content).decode("utf-8")
    return encoded_image


def remove_empty_lines(text_input: str) -> str:
    """
    Removes empty lines from the given text.

    Args:
        text_input (str): The input text from which empty lines should be removed.

    Returns:
        str: The cleaned text without empty lines.
    """
    output_clean = re.sub(r"\n\s*\n", "\n", text_input)
    return output_clean


def remove_xml_tags(text_input: str) -> str:
    """
    Removes XML tags from the given text.

    Args:
        text_input (str): The input text from which XML tags should be removed.

    Returns:
        str: The cleaned text without XML tags.
    """
    output_clean = re.sub(r"<[^>]+>", "", text_input)
    return output_clean


def remove_transkribus_metadata(text_input: str) -> str:
    """
    Removes Transkribus metadata elements from the given PAGE-XML.

    Args:
        text_input (str): The input text from which metadata elements should be removed.

    Returns:
        str: The cleaned text without Transkribus metadata.
    """
    output_clean = re.sub(r"<opener>.*?</opener>", "", text_input, flags=re.DOTALL)
    return output_clean


def strip_lines(text_input: str) -> str:
    """
    Strips leading and trailing whitespace from each line in the given text.

    Args:
        text_input (str): The input text whose lines should be stripped.

    Returns:
        str: The cleaned text with trimmed lines.
    """
    output_clean = "\n".join(line.strip() for line in text_input.splitlines())
    return output_clean


def remove_accent(text_input: str) -> str:
    """
    Removes Markdown code fences (```xml and ```) from the given text.

    Args:
        text_input (str): The input text from which code fences should be removed.

    Returns:
        str: The cleaned text without Markdown code fences.
    """
    output_clean = re.sub(r"```xml", "", text_input, flags=re.DOTALL)
    output_clean = re.sub(r"```", "", output_clean, flags=re.DOTALL)
    return output_clean


def remove_first_empty_line(text_input: str) -> str:
    """
    Removes the first line if it is empty.

    Args:
        text_input (str): The input text whose first line may be removed.

    Returns:
        str: The text without the first line if it was empty, otherwise unchanged.
    """
    lines = text_input.splitlines()
    if lines and lines[0].strip() == "":
        return "\n".join(lines[1:])
    return text_input


def remove_specific_string(text_input: str, string_to_remove: str = "hat keine Adresse, ist geprüft") -> str:
    """
    Removes a specific string from the text.

    Args:
        text_input (str): The input text from which the string should be removed.
        string_to_remove (str): The string to remove.

    Returns:
        str: The cleaned text without the specified string.
    """
    return text_input.replace(string_to_remove, "")


def get_image_type(base64_string: str) -> str:
    """
    Detects the image type from the Base64-encoded image data.

    Args:
        base64_string (str): The Base64-encoded image data.

    Returns:
        str: "jpeg" if the image is a JPEG, "png" otherwise.
    """
    if base64_string.startswith("/9j/"):
        return "jpeg"
    else:
        return "png"
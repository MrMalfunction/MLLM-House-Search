"""
Parser module for extracting structured descriptions from LLM output.
Handles delimiter-based text extraction with robust whitespace handling.
"""

import re


def parse_delimited_output(output_text):
    """
    Parse the delimiter-based output into separate fields.

    Args:
        output_text: Raw text output from the LLM

    Returns:
        Dictionary with parsed sections
    """
    result = {
        "short_description": "",
        "frontal_description": "",
        "kitchen_description": "",
        "bedroom_description": "",
        "bathroom_description": ""
    }

    try:
        # Helper function to extract content between delimiters
        def extract_section(text, start_pattern, end_pattern):
            """Extract text between start and end patterns."""
            # Use regex for more precise matching with word boundaries
            start_regex = re.escape(start_pattern)
            end_regex = re.escape(end_pattern)

            # Match start delimiter followed by content until end delimiter
            pattern = f"{start_regex}\\s*(.*?)\\s*{end_regex}"
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

            if match:
                return match.group(1).strip()
            return ""

        # Extract short description
        # Try with delimiters first
        short_desc = extract_section(output_text, "Short Description", "Short Description End")
        if not short_desc:
            # If no delimiters, grab text before first image section
            first_image = re.search(r"(Frontal|Kitchen|Bedroom|Bathroom)[\s\t]+Image", output_text, re.IGNORECASE)
            if first_image:
                short_desc = output_text[:first_image.start()].strip()
        result["short_description"] = short_desc

        # Extract individual room descriptions
        result["frontal_description"] = extract_section(output_text, "Frontal Image", "Frontal Image End")
        result["kitchen_description"] = extract_section(output_text, "Kitchen Image", "Kitchen Image End")
        result["bedroom_description"] = extract_section(output_text, "Bedroom Image", "Bedroom Image End")
        result["bathroom_description"] = extract_section(output_text, "Bathroom Image", "Bathroom Image End")

    except Exception as e:
        print(f"[Parser] Warning: Error parsing output: {e}")

    return result

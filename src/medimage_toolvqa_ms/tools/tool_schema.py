from __future__ import annotations

TOOL_SCHEMAS = [
    {
        "name": "Zoom-in",
        "description": "Crop a local region from an image by bbox.",
        "required": ["target_image", "bbox_2d"],
    },
    {
        "name": "BiomedParse",
        "description": "Segment a medical object using a text prompt.",
        "required": ["target_image", "captions"],
    },
    {
        "name": "SAM2",
        "description": "Segment an object using a bbox prompt.",
        "required": ["target_image", "bbox_2d"],
    },
]


def allowed_tool_names() -> set[str]:
    return {schema["name"] for schema in TOOL_SCHEMAS}

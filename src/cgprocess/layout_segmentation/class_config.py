"""Contains config for class names, labels and colors"""

cmap = [
    (1.0, 0.43843843843843844, 0.0),
    (0, 0.222, 0.222),
    (0.36036036036036045, 0.5, 0.5),
    (0.0, 1.0, 0.2389486260454002),
    (0.8363201911589008, 1.0, 0.0),
    (0.0, 0.5615942028985507, 1.0),
    (0.0422705314009658, 0.0, 1.0),
    (0.6461352657004831, 0.0, 1.0),
    (1.0, 0.73, 0.98),
    (0.5, 0.0, 0.0)
]

cmap_border_color = (0.2, 0.2, 1.0),

TOLERANCE = [
    5.0,  # "caption"
    5.0,  # "table"
    2.0,  # "article, paragraph"
    2.0,  # "heading"
    10.0,  # "header"
    1.0,  # "separator_vertical"
    1.0,  # "separator"
    5.0,  # "image",
    5.0,  # "inverted_text"
]

# The order dictates the priority in the drawing process. Eg. "image": 10 assigns label 10 to image regions, but the
# drawn region will be overwritten by tables, which are further down the dictionary.
LABEL_ASSIGNMENTS = {
    "TextLine": 0,
    "advertisement": 0,
    "UnknownRegion": 0,
    "image": 8,
    "Image": 8,
    "inverted_text": 9,
    "caption": 1,
    "table": 2,
    "article": 3,
    "article_": 3,
    "paragraph": 3,
    "heading": 4,
    "header": 5,
    "separator_vertical": 6,
    "separator_short": 7,
    "separator_horizontal": 7,
}

PADDING_LABEL = 255
PAGE_BORDER_CONTENT = "page_border_content"

LABEL_NAMES = [
    "caption",
    "table",
    "paragraph",
    "heading",
    "header",
    "separator_vertical",
    "separator_horizontal",
    "image",
    "inverted_text",
]

REGION_TYPES = {
    "UnknownRegion": "UnknownRegion",
    "caption": "TextRegion",
    "table": "TableRegion",
    "paragraph": "TextRegion",
    "heading": "TextRegion",
    "header": "TextRegion",
    "separator_vertical": "SeparatorRegion",
    "separator_horizontal": "SeparatorRegion",
    "image": "GraphicRegion",
    "inverted_text": "TextRegion",
}

VALID_TAGS = [
    "paragraph",
    "caption",
    "heading",
    "header",
    "image",
    "table",
    "inverted_text",
    "separator_vertical",
    "separator_horizontal",
    "UnknownRegion",
]


REDUCE_CLASSES = {3: [1, 5], 0: [8, 9]}

# REDUCE_CLASSES = {
#     6: [7]
# }

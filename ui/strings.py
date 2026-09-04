"""
ui/strings.py
-------------
Central registry for user-facing UI strings that live outside inline widget
code — the action buttons' tooltips to start. English is the source language.

To localize the app, translate the values in this module (or generate a Qt
.ts/.qm from it and feed it through a QTranslator later); every consumer
reads from here, so no widget code needs to change.

Keep each entry one translateable unit: full sentences, no mid-string
template splicing, and the same line breaks the UI should show (tooltips
render each sentence on its own line by convention in this app).
"""

# Inference page — action row.
T_MULTI_SELECT_MODE = (
    "Tick the models to run together as a batch.\n"
    "Every checked model gets its own run."
)

# Training page — action row.
T_EXPORT_WEIGHTS = (
    "Strip a training checkpoint down to the model weights\n"
    "(no optimizer / scheduler / metrics history) for inference."
)

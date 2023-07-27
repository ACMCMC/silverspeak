"""
Gradio app with a big textbox for input and a big textbox for output.
Also show tho checkboxes for replace_chars and replace_spaces.
"""
import gradio as gr

from silver_speak import rewrite_attack, EXAMPLE_TEXT

inputs = [
    gr.inputs.Textbox(lines=10, label="Input text"),
    gr.inputs.Checkbox(label="Replace characters", default=True),
    gr.inputs.Checkbox(label="Replace spaces", default=True)
]
outputs = gr.outputs.Textbox(label="Output text")
title = "Silver Speak"
description = "A tool to rewrite text to avoid AI detection systems."

gr.Interface(
    fn=rewrite_attack,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    allow_flagging=False,
    allow_screenshot=False,
    allow_download=False,
    allow_share=False,
    theme="huggingface",
    examples=[
        [EXAMPLE_TEXT, True, False],
        [EXAMPLE_TEXT, True, True],
    ],
    examples_per_page=6,
    layout="vertical",
).launch()

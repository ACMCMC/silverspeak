"""
Gradio app with a big textbox for input and a big textbox for output.
Also show tho checkboxes for replace_chars and replace_spaces.
"""

import gradio as gr

from silver_speak.homoglyphs.random_attack import random_attack as random_homoglyphs_attack
from silver_speak.deletion.random_attack import random_attack as random_deletion_attack

EXAMPLE_TEXT_1 = "Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures."
EXAMPLE_TEXT_2 = "What are the standards required of offered properties? Properties need to be habitable and must meet certain health and safety standards, which the local authority can discuss with you. These standards have been agreed by the Department of Housing, Local Government and Heritage. The local authority will assess your property to make sure it meets the standards. If the property does not meet the standards, the local authority will explain why and can discuss what could be done to bring the property up to standard. Some properties may not be suitable for all those in need of accommodation, due to location or other reasons. However, every effort will be made by the local authority to ensure that offered properties are matched to appropriate beneficiaries."

inputs = [
    gr.Textbox(lines=10, label="Input text"),
    gr.Checkbox(label="Homoglyphs", value=True),
    gr.Slider(
        minimum=0,
        maximum=0.3,
        step=0.01,
        value=0.05,
        label="Percentage of characters to replace",
    ),
    gr.Checkbox(label="Zero-Width", value=False),
]
outputs = gr.Textbox(label="Output text", show_copy_button=True)
title = "Silverspeak"
description = "A tool to rewrite text to avoid AI detection systems."

def attack_fn(text, replace_chars, percentage, replace_spaces):
    if replace_chars:
        text = random_homoglyphs_attack(
            text=text, percentage=percentage
        )
    if replace_spaces:
        text = random_deletion_attack(
            original_text=text, percentage=percentage
        )
    return text

gr.Interface(
    fn=attack_fn,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    allow_flagging="never",
    examples=[
        [EXAMPLE_TEXT_1, True, 0.05, False],
        [EXAMPLE_TEXT_1, True, 0.03, True],
        [EXAMPLE_TEXT_2, True, 0.1, False],
        [EXAMPLE_TEXT_2, False, 0.05, True],
    ],
    examples_per_page=2,
).launch()

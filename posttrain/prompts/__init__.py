"""Prompt templates for training and evaluation.

r1_zero.prompt        the DeepSeek-R1-Zero format. Its text is the template of the MIT-licensed Dr. GRPO code
                      (sail-sg/understand-r1-zero, apply_r1_template); see THIRD_PARTY_NOTICES.md.
question_only.prompt  the bare question.

Templates are read with surrounding whitespace stripped; {question} is filled with the problem text.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    return open(os.path.join(HERE, name + ".prompt"), encoding="utf-8").read().strip()

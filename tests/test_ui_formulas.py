"""LaTeX formulas in model answers are rewritten so st.markdown renders them."""

from ui import latex_for_streamlit


def test_inline_formula_becomes_dollar_math():
    text = r"Potential energy: \(PE = 12 \times 9.8 \times 2.5 = 294\ \text{J}\)."
    assert latex_for_streamlit(text) == r"Potential energy: $PE = 12 \times 9.8 \times 2.5 = 294\ \text{J}$."


def test_spaces_inside_the_delimiters_are_removed():
    assert latex_for_streamlit(r"\( m = 12\ \text{kg} \)") == r"$m = 12\ \text{kg}$"


def test_display_formula_becomes_double_dollar_math():
    text = "The work is\n\\[\nW = mgh\n\\]\nso it grows with h."
    assert latex_for_streamlit(text) == "The work is\n$$W = mgh$$\nso it grows with h."


def test_several_formulas_in_one_answer():
    text = r"(\(m = 12\)), (\(g = 9.8\)) and \[PE = mgh\]"
    assert latex_for_streamlit(text) == r"($m = 12$), ($g = 9.8$) and $$PE = mgh$$"


def test_plain_text_and_ordinary_brackets_are_unchanged():
    text = "Canberra (not Sydney) is the capital [source: Wikipedia]."
    assert latex_for_streamlit(text) == text


def test_code_blocks_are_left_alone():
    text = "Formula \\(x^2\\)\n```python\nprint(r'\\(x\\)')\n```\nand \\(y\\)"
    assert latex_for_streamlit(text) == "Formula $x^2$\n```python\nprint(r'\\(x\\)')\n```\nand $y$"

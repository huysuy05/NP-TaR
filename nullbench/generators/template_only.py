def generate_template_only(n: int, base_prompt: str = "Text: {text}"):
    """
    Keep the classification template but remove content.
    Example: base_prompt="Text: {text} Label:" → "Text:  Label:"
    """
    template = base_prompt.replace("{text}", "")
    return [template] * n
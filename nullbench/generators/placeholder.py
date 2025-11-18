def generate_placeholders(n : int):
    """
    Generate n short placeholders
    Example: "Text: NA" or "Text: NULL"
    """
    options = ["NA", "N/A", "NULL", "None", "No Content", "Nothing to see here"] #Could add more for testing purposes
    m = len(options)
    return [options[i % m] for i in range(n)]
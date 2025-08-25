def check_time_sensitive(query: str) -> bool:
    """
    Check if the query is time sensitive.
    """
    time_words = [
        "today",
        "now",
        "latest",
        "current",
        "price",
        "stock",
        "weather",
        "time",
        "yesterday",
        "tomorrow",
    ]
    return any(word in query.lower() for word in time_words)

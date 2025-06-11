def categorize_feature(name):
    if "mean" in name:
        return "mean"
    elif "se" in name or "error" in name:
        return "se"
    elif "worst" in name:
        return "worst"
    else:
        return "other"

def summarize_top_features(top_features):
    from collections import Counter
    categories = [categorize_feature(name) for name, _ in top_features]
    return Counter(categories)

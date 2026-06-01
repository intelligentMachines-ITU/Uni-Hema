import evaluate

def compute_rouge(predictions, references):
    """
    Compute ROUGE scores between predictions and references.
    
    Args:
        predictions (list of str): List of generated texts.
        references (list of str): List of reference texts.
    
    Returns:
        dict: ROUGE scores.
    """
    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=predictions, references=references)
    return result
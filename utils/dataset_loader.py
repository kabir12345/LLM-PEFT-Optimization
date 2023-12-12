from datasets import load_dataset

def load_imdb_dataset():
    """
    Load the IMDb movie reviews dataset for sentiment analysis.
    """
    return load_dataset("imdb")

def load_cnn_dailymail_dataset():
    """
    Load the CNN/Daily Mail dataset for summarization tasks.
    """
    return load_dataset("cnn_dailymail", "3.0.0")

def load_wikitext_dataset():
    """
    Load the WikiText dataset for text generation tasks.
    """
    return load_dataset("wikitext")

if __name__ == "__main__":

    imdb = load_imdb_dataset()
    cnn_dm = load_cnn_dailymail_dataset()
    wikitext = load_wikitext_dataset()
    print(f"IMDb dataset: {imdb}")
    print(f"CNN/Daily Mail dataset: {cnn_dm}")
    print(f"WikiText dataset: {wikitext}")

from datasets import load_dataset

def load_imdb_dataset():
    return load_dataset("imdb")

def load_cnn_dailymail_dataset():
    return load_dataset("cnn_dailymail", "3.0.0")

def load_wikitext_dataset():
    return load_dataset("wikitext")

if __name__ == "__main__":

    imdb = load_imdb_dataset()
    cnn_dm = load_cnn_dailymail_dataset()
    wikitext = load_wikitext_dataset()

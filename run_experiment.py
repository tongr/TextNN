
def main():
    import fire
    from textnn.dataset.imdb import ImdbClassifier
    from textnn.dataset.amazon import AmazonReviewClassifier
    fire.Fire({
        "imdb": ImdbClassifier,
        "amazon": AmazonReviewClassifier,
    })


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=getattr(logging, "INFO"), format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
    main()

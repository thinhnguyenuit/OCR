import os
import pytesseract
import argparse
import logging

logger = logging.getLogger()


def ocr():
    """
    Driver function
    """
    logger.setLevel("INFO")
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s : %(name)s : %(levelname)s : %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Parse arguments
    args = arguments_parser()
    path = args["path"]
    lang = args["lang"]

    logger.info(f"args: {path} - {lang}")

    return character_rocognition(path, lang)


def character_rocognition(path, lang):
    if not os.path.exists(path):
        logger.error(f"Can not find inpput image: {path}")
        raise Exception(f"No such file or directory {path}")
    return pytesseract.image_to_string(path, lang=lang)


def arguments_parser() -> dict:
    """
    Parse arguments from command line
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", type=str, help="path to input image")
    ap.add_argument("-l", "--lang", type=str, default="eng", help="language to perfome OCR")
    return vars(ap.parse_args())


if __name__ == "__main__":
    texts = ocr()

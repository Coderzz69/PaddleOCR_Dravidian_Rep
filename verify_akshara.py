import numpy as np
import regex

from ppocr.data.imaug.label_ops import CTCLabelEncode, NRTRLabelEncode
from ppocr.postprocess.rec_postprocess import CTCLabelDecode


DICT_PATH = "akshara/dict.txt"
MAX_TEXT_LENGTH = 25
SAMPLES = [
    "தமிழ்",
    "தரிசனத்தின்",
    "வணக்கம்",
    "க்ஷேத்திரம்",
    "ஸ்ரீ",
    "போலும்.....!",
]


def load_dict(path):
    with open(path, "r", encoding="utf-8") as fin:
        return [line.rstrip("\n").rstrip("\r") for line in fin]


def main():
    dictionary = load_dict(DICT_PATH)
    print(f"dictionary path: {DICT_PATH}")
    print(f"dictionary tokens: {len(dictionary)}")

    ctc_encoder = CTCLabelEncode(
        max_text_length=MAX_TEXT_LENGTH,
        character_dict_path=DICT_PATH,
        use_space_char=False,
    )
    nrtr_encoder = NRTRLabelEncode(
        max_text_length=MAX_TEXT_LENGTH,
        character_dict_path=DICT_PATH,
        use_space_char=False,
    )
    decoder = CTCLabelDecode(character_dict_path=DICT_PATH, use_space_char=False)

    print(f"ctc vocabulary size: {len(ctc_encoder.character)}")
    print(f"nrtr vocabulary size: {len(nrtr_encoder.character)}")

    for text in SAMPLES:
        tokens = regex.findall(r"\X", text)
        encoded = ctc_encoder.encode(text)
        nrtr_encoded = nrtr_encoder.encode(text)
        if encoded is None:
            raise AssertionError(f"CTC encoder rejected: {text!r}")
        if nrtr_encoded is None:
            raise AssertionError(f"NRTR encoder rejected: {text!r}")

        decoded = decoder.decode(np.array([encoded], dtype=np.int64))[0][0]
        if decoded != text:
            raise AssertionError(
                f"round-trip failed for {text!r}: decoded {decoded!r}"
            )

        print(f"\ntext: {text}")
        print(f"tokens: {tokens}")
        print(f"encoded IDs: {encoded}")
        print(f"decoded text: {decoded}")

    print("\nAkshara encode -> decode round-trip: OK")


if __name__ == "__main__":
    main()

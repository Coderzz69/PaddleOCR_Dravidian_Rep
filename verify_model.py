import argparse
import copy

import paddle
import yaml

from ppocr.data.imaug.label_ops import CTCLabelEncode, NRTRLabelEncode
from ppocr.losses.rec_ctc_loss import CTCLoss
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process


DICT_PATH = "akshara/dict.txt"
SAMPLE_TEXT = "தமிழ்"


def load_config(path, dict_path):
    with open(path, "r", encoding="utf-8") as fin:
        config = yaml.safe_load(fin)
    config["Global"]["character_dict_path"] = dict_path
    return config


def configure_head(config):
    post_process = build_post_process(config["PostProcess"], config["Global"])
    char_num = len(post_process.character)
    if config["Architecture"]["Head"]["name"] == "MultiHead":
        out_channels_list = {"CTCLabelDecode": char_num}
        second_loss = list(config["Loss"]["loss_config_list"][1].keys())[0]
        if second_loss == "SARLoss":
            out_channels_list["SARLabelDecode"] = char_num + 2
        elif second_loss == "NRTRLoss":
            out_channels_list["NRTRLabelDecode"] = char_num + 3
        config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
    else:
        config["Architecture"]["Head"]["out_channels"] = char_num
    return post_process, char_num


def ctc_classifier_dim(model):
    ctc_head = model.head.ctc_head
    fc = getattr(ctc_head, "fc", None)
    if fc is not None:
        return int(fc.weight.shape[-1])
    return int(ctc_head.fc2.weight.shape[-1])


def make_batch(config):
    max_text_length = config["Global"]["max_text_length"]
    image_shape = config["Global"].get("d2s_train_image_shape", [3, 48, 320])
    ctc_encoder = CTCLabelEncode(
        max_text_length=max_text_length,
        character_dict_path=config["Global"]["character_dict_path"],
        use_space_char=config["Global"].get("use_space_char", False),
    )
    nrtr_encoder = NRTRLabelEncode(
        max_text_length=max_text_length,
        character_dict_path=config["Global"]["character_dict_path"],
        use_space_char=config["Global"].get("use_space_char", False),
    )

    ctc_data = ctc_encoder({"label": SAMPLE_TEXT})
    nrtr_data = nrtr_encoder({"label": SAMPLE_TEXT})
    if ctc_data is None or nrtr_data is None:
        raise AssertionError(f"encoders rejected sample: {SAMPLE_TEXT!r}")

    image = paddle.randn([1] + image_shape, dtype="float32")
    label_ctc = paddle.to_tensor(ctc_data["label"][None, :], dtype="int64")
    label_gtc = paddle.to_tensor(nrtr_data["label"][None, :], dtype="int64")
    length = paddle.to_tensor([int(ctc_data["length"])], dtype="int64")
    valid_ratio = paddle.to_tensor([1.0], dtype="float64")
    return [image, label_ctc, label_gtc, length, valid_ratio]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml",
        help="PP-OCRv5 recognition config to verify.",
    )
    parser.add_argument("--dict", default=DICT_PATH, help="Akshara dictionary path.")
    args = parser.parse_args()

    paddle.set_device("cpu")
    config = load_config(args.config, args.dict)
    post_process, vocab_size = configure_head(config)
    model = build_model(copy.deepcopy(config["Architecture"]))
    model.train()

    batch = make_batch(config)
    preds = model(batch[0], data=batch[1:])
    ctc_loss = CTCLoss()(preds["ctc"], batch[:2] + batch[3:])["loss"]

    print(f"config: {args.config}")
    print(f"dictionary: {args.dict}")
    print(f"vocabulary size: {vocab_size}")
    print(f"classifier output dimension: {ctc_classifier_dim(model)}")
    print(f"dummy input shape: {list(batch[0].shape)}")
    print(f"ctc output shape: {list(preds['ctc'].shape)}")
    print(f"ctc loss: {float(ctc_loss.numpy())}")
    print("model forward + CTC loss: OK")


if __name__ == "__main__":
    main()

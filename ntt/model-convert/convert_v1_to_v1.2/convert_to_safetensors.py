import os
import argparse
import logging
from collections import defaultdict
import shutil

import torch
import json
from tqdm import tqdm
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


OLD_INDEX = "pytorch_model.bin.index.json"
NEW_INDEX = "model.safetensors.index.json"

def _get_filenames(dir_path):
    try:
        files = os.listdir(dir_path)
        return files
    except FileNotFoundError as e:
        logger.error(f" Directry not found: {dir_path}")
        raise e
    except Exception as e:
        raise e


def _get_pytorch_model_bin_list(old_model_path):
    filename = "pytorch_model.bin.index.json"
    with open(f"{old_model_path}/{filename}", "r") as f:
        data = json.load(f)

        # 複数のブロックで同じbinが参照されているためsetでユニークにする
        model_bin_list = set(data["weight_map"].values())

    return model_bin_list


def _change_ext_to_safetensors(filename):

    # 拡張子チェック
    if filename.endswith(".bin"):
        new_filename = os.path.splitext(filename)[0] + ".safetensors"
        new_filename = new_filename.replace("pytorch_model", "model")
        return new_filename
    else:
        raise ValueError("入力したモデルの形式がpytorch*.binではありません。")


def copy_files(old_model_path, new_model_path):
    file_list = [
        "adapt_tokenizer.py",
        "config.json",
        "configuration_mpt.py",
        "generation_config.json",
        "modeling_mpt.py",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    os.makedirs(new_model_path, exist_ok=True)
    for filename in file_list:
        source_file = f"{old_model_path}/{filename}"
        target_file = f"{new_model_path}/{filename}"
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            logger.info(f"Copied: {source_file} -> {target_file}")
        else:
            logger.warn(f"File not found: {source_file}")


def _convert_to_safetensors(pt_model_path, sf_model_path):
    loaded = torch.load(pt_model_path, map_location="cpu")
    dirname = os.path.dirname(sf_model_path)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_model_path, metadata={"format": "pt"})


def _convert_single(old_model_path, new_model_path):
    pt_filename = f"{old_model_path}/pytorch_model.bin"
    sf_filename = _change_ext_to_safetensors("pytorch_model.bin")
    sf_filename = os.path.join(new_model_path, sf_filename)
    _convert_to_safetensors(pt_filename, sf_filename)


def _convert_multi(old_model_path, new_model_path):
    # new_filenames = []

    with open(f"{old_model_path}/{OLD_INDEX}", "r") as f:
        old_index_data = json.load(f)
        # 複数のブロックで同じbinが参照されているためsetでユニークにする
        pt_model_bin_list = set(old_index_data["weight_map"].values())

    for pt_model_filename in tqdm(pt_model_bin_list):
        pt_filename = f"{old_model_path}/{pt_model_filename}"
        sf_filename = _change_ext_to_safetensors(pt_model_filename)
        sf_filename = os.path.join(new_model_path, sf_filename)
        # safetensorsに変更しながら保存
        _convert_to_safetensors(pt_filename, sf_filename)
        # new_filenames.append(sf_filename)

    # model.safetensors.index.json"作成
    new_index_path = f"{new_model_path}/{NEW_INDEX}"
    with open(new_index_path, "w") as f:
        new_index_data = {k: v for k, v in old_index_data.items()}
        # safetensorsに名前が置き換わったmapを作る
        new_map = {
            k: _change_ext_to_safetensors(v)
            for k, v in old_index_data["weight_map"].items()
        }
        new_index_data["weight_map"] = new_map
        json.dump(new_index_data, f, indent=4)


def convert_to_safetensors(old_model_path, new_model_path):

    flag = 0
    filenames = _get_filenames(old_model_path)

    # pytorch_model.bin.index.jsonがある場合
    if OLD_INDEX in filenames:
        _convert_multi(old_model_path, new_model_path)
        flag = 1
    # pytorch_model.binがある場合
    if "pytorch_model.bin" in filenames:
        _convert_single(old_model_path, new_model_path)
        flag = 1

    if flag == 1:
        logger.info("all model files have been successfully converted.")
    else:
        logger.error("failed to convert model.")

def arg_parse():
    parser = argparse.ArgumentParser(description="tsuzumiモデルタイプ変換ツール")
    parser.add_argument(
        "--input-model", type=str, required=True, help="変換前モデルパス"
    )
    parser.add_argument(
        "--output-model", type=str, required=True, help="変換後モデルパス"
    )

    return parser.parse_args()


def main():
    args = arg_parse()
    convert_to_safetensors(args.input_model, args.output_model)


if __name__ == "__main__":
    main()

# モデル変換スクリプト

本ツールはtsuzumiのモデルをbin形式からsafetensors形式に変換するツールである。

# 構成

* 構成は以下の通り。

```bash
- REMEAD.md   # 本ファイル
- convert_to_safetensors.py # convertスクリプト本体
- convert_to_safetensors.sh # convertスクリプトのラッパー
- tsuzumi_7B_v1_2_template  # tsuzumi7B v1.2のテンプレート格納ディレクトリ
```

# 使い方

## convert_to_safetensors.sh

* コマンド
```bash
./convert_to_safetensors.sh --input-model <input_model_dir> --output-model <output_model_dir> --template-model <template_model_dir>
```

### 引数

* `--input-model` : 入力するモデルのディレクトリ。bin形式のモデルのディレクトリを指定する
* `--output-model`: 出力するモデルのディレクトリ。safetensors形式のモデルが出力されるディレクトリを指定する
* `--template-model`: テンプレートモデルのディレクトリ。ここで指定されたディレクトリに存在する設定ファイル類を`--output-model` ディレクトリにコピーする。
    * コピーするファイルは下記の通り
        - `README.md`
        - `__init__.py`
        - `adapt_tokenizer.py`
        - `attention.py`
        - `blocks.py`
        - `config.json`
        - `configuration_mpt.py`
        - `custom_embedding.py`
        - `fc.py,ffn.py`
        - `generation_config.json`
        - `modeling_mpt.py`
        - `norm.py`
        - `param_init_fns.py`
        - `special_tokens_map.json`
        - `tokenizer.json`
        - `tokenizer_config.json`

### 実行例

* 以下に実行例を記載する。
```bash
$ ./convert_to_safetensors.sh --input-model v1_02_7b-instruct-hf --output-model v1_02_7b-instruct-hf_convert --template-model tsuzumi_7B_v1_2_template
2024-09-18 06:38:09 [INFO] Starting processing...
2024-09-18 06:38:09 [INFO] Input model directory: v1_02_7b-instruct-hf
2024-09-18 06:38:09 [INFO] Output model directory: v1_02_7b-instruct-hf_convert
2024-09-18 06:38:09 [INFO] Template model directory: tsuzumi_7B_v1_2_template
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [05:08<00:00, 154.40s/it]
2024-09-18 06:43:19,980 - __main__ - INFO - all model files have been successfully converted.
```

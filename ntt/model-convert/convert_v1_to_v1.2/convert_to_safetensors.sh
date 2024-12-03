#!/bin/bash -e
# 変換スクリプト .bin -> .safetensors

log() {
    local type="$1"
    shift
    printf '%s [%s] %s\n' "$(date +'%Y-%m-%d %H:%M:%S')" "$type" "$*"
}

# 使用方法
usage() {
    echo "Usage: $0 --input-model <input_model_dir> --output-model <output_model_dir> --template-model <template_model_dir>"
    exit 1
}

# 引数の解析
INPUT_DIR=""
OUTPUT_DIR=""
TEMPLATE_DIR=""

while [[ "$1" != "" ]]; do
    case $1 in
        --input-model )        shift
                         INPUT_DIR=$1
                         ;;
        --output-model )       shift
                         OUTPUT_DIR=$1
                         ;;
        --template-model )     shift
                         TEMPLATE_DIR=$1
                         ;;
        * )              usage
                         ;;
    esac
    shift
done

# 必要な引数が指定されているか確認
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$TEMPLATE_DIR" ]]; then
    usage
fi

# ディレクトリの存在確認
if [[ ! -d "$INPUT_DIR" ]]; then
    log "ERROR" "input model '$INPUT_DIR' does not exist."
    exit 1
fi

if [[ -d "$OUTPUT_DIR" ]]; then
    log "ERROR" "output model dir '$OUTPUT_DIR' exist."
    exit 1
fi

if [[ ! -d "$TEMPLATE_DIR" ]]; then
    log "ERROR" "Template model dir '$TEMPLATE_DIR' does not exist."
    exit 1
fi

# 必要な処理の実行
log "INFO" "Starting processing..."
log "INFO" "Input model directory: $INPUT_DIR"
log "INFO" "Output model directory: $OUTPUT_DIR"
log "INFO" "Template model directory: $TEMPLATE_DIR"

python convert_to_safetensors.py --input-model ${INPUT_DIR} \
                                 --output-model ${OUTPUT_DIR}

# template dirから必要なファイルのコピー

cp ${TEMPLATE_DIR}/{README.md,__init__.py,adapt_tokenizer.py,attention.py,blocks.py,config.json,configuration_mpt.py,custom_embedding.py,fc.py,ffn.py,generation_config.json,modeling_mpt.py,norm.py,param_init_fns.py,special_tokens_map.json,tokenizer.json,tokenizer_config.json} ${OUTPUT_DIR}/

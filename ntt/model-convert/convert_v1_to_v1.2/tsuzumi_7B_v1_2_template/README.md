# tsuzumi 7B v1 hf 版

tsuzumi_7B_v1のモデル実装からhuggingface's transformersで呼び出す部分だけを取り出し、llm-foundryコンテナへの依存をなくした実装です。

## モデルの重み

ディレクトリにモデルの重みは含まれていません。モデルファイルをコピーしてご使用ください。

```
cp /model/tsuzumi-v1-02/model-00001-of-00003.safetensors .
cp /model/tsuzumi-v1-02/model-00002-of-00003.safetensors .
cp /model/tsuzumi-v1-02/model-00003-of-00003.safetensors .
cp /model/tsuzumi-v1-02/model.safetensors.index.json .
```

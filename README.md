## 概要
- tsuzumiをKSサーバ上で動かすためのsingularity環境構築用の資材です。
- 公式のDockerFileをsingularity pythonで変換し、recipeファイル (tsuzumi.def)を作成しています。
  - 参考: https://docs.abci.ai/ja/containers/
- 推論だけであればAPI版がどこかで動いている気がするので、そちらの利用が可能かもしれません。

## 準備
- このリポジトリを20241001_tsuzumi7B-v1.2/sample/tools/に配置
- dockerディレクトリからnttをコピー
  ~~~
  cp -r ../docker/ntt .
  ~~~

## KSサーバの利用方法
1. ログイン用サーバに移動
   ~~~
   ssh ks000 or ks001
   ~~~
   ※ID/PASSは他INET上のマシンと同様
2. 計算用 (GPU付)サーバに移動
   ~~~
   srun -p gpu-v100 --gres gpu:1 --pty /bin/bash
   ~~~
   ※オプションの詳細な意味などは[ksサーバのページ](http://khn-wiki.cslab.kecl.ntt.co.jp/index.php?Slurm)を参照ください  
   ※v1.2は推論に14GB程度のGPUメモリが必要なのでGPUはV100相当くらいがおすすめ (-p gpu-v100)

## Singularityコンテナのビルドと実行
### 計算機
- 計算用サーバに移動
  - ビルドだけならログイン用サーバでも可能ですが、計算負荷の軽減のため計算用サーバでの実施を推奨
  - 実行は計算用サーバが必要
- ビルドは最初の1回のみ実施

### 実施方法
1. コンテナをビルド
   ~~~
   ./build_singularity.sh
   ~~~
2. 実行 (Singularityコンテナ内に移動)
   ~~~
   ./launch_singularity.sh
   ~~~
   ※プロンプトの先頭が Singularity>に変わります

### 推論方法
- 準備
  - 今のところKSサーバのGPUにてflash-attentionがうまく動かないので設定を変更する必要あり (要修正)
  - コンテナ内かホストのmodelsディレクトリに移動 (共有されているのでどちらでも問題ありません)
    ~~~
    # コンテナ内の場合:
    cd /models/tsuzumi-7b-v1_2-8k-instruct

    # ホスト側の場合
    cd {somewhere}/20241001_tsuzumi7B-v1.2/models/tsuzumi-7b-v1_2-8k-instruct
    ~~~
  - config.jsonlファイルを開き、編集
    ~~~
    # 9行目
    "attn_impl": "flash" -> "attn_impl": "torch"
    ~~~
- 実施
  ~~~
  python3.10 /work/inference.py
  ~~~
  ※Python3.10上に要求されるパッケージがインストールされており、現状明示的に指定が必要 (python or python3 で実行したい場合はコンテナ内でPATHを設定してください)
- 指示文の変更
  - inference.pyの先頭に指示文が書かれていますので、まずはこちらを編集いただければと思います。
  - ホスト側のworkディレクトリと共有されていますので、どちらを編集しても問題ありません。

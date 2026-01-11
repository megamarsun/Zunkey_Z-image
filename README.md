# Zunkey_Z-image v1.0.0

Tongyi-MAI/Z-Image-Turbo を使って「1行1シチュエーション」をまとめて生成するバッチGUIツールです。

## 特徴
- prompts.txt：1行 = 1シチュエーション（コピペ運用）
- 共通プロンプト：スタイル等を別欄に固定
- 画像 `{prefix}_0000.png` と同名の `{prefix}_0000.txt` に full_prompt とメタ情報を自動保存
- プレビュー機能（←/→ ボタン & キーボード ←/→）
- 保存フォルダをエクスプローラー等で開く「開く」ボタン
- 詳細設定（普段は触らなくてOK）を折りたたみで隔離

※ 学習済みモデルは同梱していません。初回起動時に Hugging Face から自動ダウンロードされます。モデルの利用条件は各自で確認してください。

## セットアップ（Pythonで動かす場合）
Python 3.10+ 推奨。

```bash
pip install -r requirements.txt
```

PyTorch は環境に合わせてインストールしてください（CUDA例）  
（例：CUDA 12.1）
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

起動：
```bash
python Zunkey_Z_image_v1_0_0.py
```

## 使い方（最短）
1. 左上に「1行1シチュ」を貼る
2. 共通プロンプトにスタイルを書く（例：lineart）
3. 保存フォルダと prefix を設定
4. 「バッチ開始」
5. 右側で画像と full_prompt を確認（←/→ or キーボード）

## GitHub公開向けメモ
- `tmp/`, `output_batch/`, `prompts_backup/`, `prompts.txt`, `zunkey_z_image_config.json` は実行時生成なので `.gitignore` に入れています。

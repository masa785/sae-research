from datasets import load_dataset
import json

print("Hugging Faceから本物のデータをストリーミングで取得開始...")

# streaming=True で繋ぎ、必要な分だけ抜き出す
ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

# Macの中に「mini_pile.jsonl」という小さなファイルを作る
with open("mini_pile.jsonl", "w", encoding="utf-8") as f:
    for i, row in enumerate(ds):
        if i >= 1000: # 1000件（約数MB）でストップ！ここを変えれば量は自由自在です
            break
        f.write(json.dumps(row) + "\n")
        if i % 100 == 0:
            print(f"{i}件 取得完了...")

print("完了！Macのローカルに『mini_pile.jsonl』を保存しました！")
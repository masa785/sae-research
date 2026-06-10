# Windows CUDA / CPU 実行メモ

このコードは `--device cpu` と `--device cuda` の両方を選べます。

ただし、`--device cuda` を使うには CUDA 対応版の PyTorch が入っている必要があります。CPU版 PyTorch のままだと CUDA は使えません。

## 現状確認

PowerShell で実行します。

```powershell
uv run python -c "import torch; print('torch', torch.__version__); print('cuda build', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

CUDA が使える状態なら、`cuda build` が `12.x` のような値になり、`cuda available` が `True` になります。

## CUDA版 PyTorch に切り替える

```powershell
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

インストール後、もう一度「現状確認」を実行してください。

## CPU版 PyTorch に戻す

```powershell
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio
```

## CUDA 実行例

```powershell
uv run TestPlay/examples/main.py `
  --model-name "EleutherAI/pythia-160m-deduped" `
  --tau 20 `
  --buffer-size 0.1m `
  --total-tokens 1m `
  --topk 25 `
  --noise-mode lap `
  --out-batch-ratio 0.1 `
  --lr 0.01 `
  --seed 456 `
  --layer 8 `
  --z-dim 768 `
  --optimizer adam `
  --device cuda `
  --buffer-device cpu `
  --llm-device cuda `
  --data-source hf `
  --refresh-batch-size 16 `
  --results-dir "C:\r"
```

`--buffer-device cpu` は推奨設定です。activation buffer まで CUDA に置くと GPU メモリを多く使います。

## CPU 実行例

```powershell
uv run TestPlay/examples/main.py `
  --model-name "EleutherAI/pythia-160m-deduped" `
  --tau 20 `
  --buffer-size 0.1m `
  --total-tokens 1m `
  --topk 25 `
  --noise-mode lap `
  --out-batch-ratio 0.1 `
  --lr 0.01 `
  --seed 456 `
  --layer 8 `
  --z-dim 768 `
  --optimizer adam `
  --device cpu `
  --buffer-device cpu `
  --llm-device cpu `
  --data-source hf `
  --refresh-batch-size 16 `
  --results-dir "C:\r"
```


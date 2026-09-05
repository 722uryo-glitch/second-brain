# Second Brain v1

ローカルLLM(Ollama) + 長期記憶 + 自発内省(DMN) + Whisper文字起こし を1つにした最小構成です。
OpenAI/Claude APIは不要です。

## できること

- Ollama上のローカルLLMと会話
- 会話をSQLiteへ長期記憶として保存
- Ollama embeddingsで意味検索し、過去記憶を自動想起
- 30分ごとに最近の記憶を読んで内省(DMN)
- 手動で重要メモを保存
- Whisperで音声を文字起こしし、そのまま長期記憶へ保存
- ブラウザUI: http://127.0.0.1:8765
- n8nを後から接続可能

## 1. 必要なもの

- Windows 10/11
- Python 3.11推奨
- Ollama
- ffmpeg（Whisperで音声を扱うため）
- GPUは推奨だが必須ではありません

## 2. Ollama

Ollamaをインストール後、PowerShellで:

```powershell
cd <このフォルダ>
.\setup_ollama.ps1
```

デフォルトでは次を取得します。

- 会話: `qwen3:8b`
- embeddings: `nomic-embed-text`

PC性能に合わせて `.env` で変更できます。

## 3. 起動

`.env.example` を `.env` にコピーしてから:

```powershell
Copy-Item .env.example .env
.\run.ps1
```

起動後:

http://127.0.0.1:8765

## 4. Whisper

Whisperは初回文字起こし時にモデルをダウンロードします。
`.env` の `WHISPER_MODEL` を変更可能です。

- tiny / base: 軽い
- small: デフォルト
- medium / large: 高品質だが重い

ffmpegがPATHに必要です。

## 5. DMN（自発内省）

デフォルト30分ごとです。

```env
DMN_INTERVAL_MINUTES=30
DMN_ENABLED=true
```

内省は最近25件を読み、価値がある時だけ `reflection` として長期記憶に保存します。

## 6. n8n

Docker Desktopがある場合:

```powershell
docker compose -f docker-compose.n8n.yml up -d
```

http://127.0.0.1:5678

n8nからSecond Brainを呼ぶ場合はHTTP Requestノードで以下を使えます。

- `POST http://host.docker.internal:8765/api/chat`
  - JSON: `{ "message": "..." }`
- `POST http://host.docker.internal:8765/api/reflect`
- `GET http://host.docker.internal:8765/api/memories?limit=50`

### 例

Schedule Trigger → HTTP Request `/api/reflect`

これだけで、n8n側から定期的な内省を実行できます。

## 7. データ

記憶は `second_brain.db` に保存されます。
外部クラウドへ送らず、Ollamaを使う限りLLM処理はローカルです。

## 現在のv1で未実装

- 知識グラフ可視化
- 記憶の自動統合/忘却
- タスク実行エージェント
- ブラウザ操作
- メール/GitHub連携
- GPT等への難問エスカレーション
- 権限/承認システム

これらはv2以降で追加できます。

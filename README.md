# Second Brain V2

Windows上で動く、永続記憶・世界情報収集・調査・検証・自己監査を統合した個人向けAI基盤です。

## 現在の構成

- **Ollama**: ローカル/プライベート処理、embeddings、フォールバック
- **UnoRouter**: 公開情報の推論・検証・複雑な成果物生成を必要に応じて自動ルーティング
- **SQLite**: 記憶、外部情報、Claims/Evidence、実行履歴の機械的な正本
- **SearXNG**: ローカルの一般Webメタ検索。`run.ps1` がDockerで自動起動を試みる
- **Trafilatura**: 検索結果ページから本文を抽出
- **Google News / GDELT / GitHub / Bluesky / Reddit / Mastodon / RSS / primary feeds**: 常時情報収集
- **Obsidian**: 人が読める知識ビュー
- **n8n**: 定期処理・外部自動化

## Second Brainの実行パイプライン

通常の複雑な依頼は、単にLLMへ転送されません。

```text
依頼
→ 目的理解 / タスク分解
→ 必要なら長期記憶を検索
→ 調査課題を分割
→ SQLite蓄積知識 + Claimsを検索
→ SearXNG / Google Newsで追加探索
→ 上位ページ本文を抽出
→ 証拠不足を判定
→ 不足していれば追加検索（複数ラウンド）
→ 適切なAIモデルで推論・成果物生成
→ 別の検証モデル + deterministic auditで自己監査
→ 問題があれば自動修正
→ 実行履歴・会話・ durable memory を保存
```

## 記憶

- 最近の会話: short-term context
- preference / user_fact / decision / task / goal: durable memory
- embeddingsによる意味検索
- semantic duplicate抑制
- 古い会話がprompt windowから落ちても、rolling conversation stateで目標・決定・未完了事項を保持
- assistantの過去発言よりuser/manual由来の記憶を優先

## 調査・検証

- SQLite FTS5（対応環境ではtrigram優先）+ LIKE fallbackのハイブリッド検索
- 新しさだけでなく、質問との関連度でEvidenceを順位付け
- SearXNGで一般Web検索
- Google Newsをニュース補完として併用
- 重要なページはTrafilaturaで本文抽出
- Evidence不足をAIが意味的に判定し、必要な追加検索語を自動生成
- 「需要」「供給不足」「競合の少なさ」など別種の主張を混同しないよう監査
- 同一wire記事・転載らしいEvidenceを別の独立ソースとして水増ししないlineage collapse
- 失敗したネットワーク/LLM/本文取得はbounded retry + exponential backoff

## AI Router

役割によってモデルを自動で切り替えます。

- `fast_cloud`: planner / 軽量会話
- `reasoning`: 複雑な調査・成果物
- `verify`: fact-check / evidence-gap / critic
- `local`: 個人記憶・private processing

UnoRouterモデルが失敗・rate limitになった場合は、再試行、モデル切替、cooldownを行い、最後はOllamaへフォールバックします。

## 起動

初回のみ `.env.example` を `.env` へコピーします。

```powershell
Copy-Item .env.example .env
.\run.ps1
```

普段は:

```powershell
.\update.ps1
.\run.ps1
```

UI:

```text
http://127.0.0.1:8765
```

SearXNGはDocker Desktopが動いていれば `run.ps1` が自動起動します。手動確認:

```powershell
.\setup_searxng.ps1
```

## 状態確認

- `/api/health` — 全体
- `/api/ai-router/status` — モデルルーター
- `/api/research/status` — SearXNG / retrieval / runtime state
- `/api/executive/status` — 司令塔
- `/api/executive/runs?steps=true` — PLAN / MEMORY / RESEARCH / DRAFT / REVIEW / REVISE の実行履歴
- `/api/intelligence/queues` — document / fact-check backlog

## 外部情報収集

バックグラウンドで多言語ニュース、一次情報、GitHub、SNS等を収集し、Claims/Evidenceへ正規化します。X公式recent searchは `X_BEARER_TOKEN` がある場合のみ動作します。

本文取得失敗は最大4回まで指数バックオフで再試行します。

## プライバシー

`.env`、SQLite DB、Obsidian vaultはGitへ入れない構成です。個人記憶を含む処理は原則local laneを使います。`UNOROUTER_PRIVATE_CHAT=true` を明示した場合のみ、personal-memoryを使う会話でcloud利用を許可します。

**このFastAPIポートをインターネットへ直接公開しないでください。** 現在のローカルUI/n8n構成では認証ゲートを前提にしていません。

## まだ外部条件が必要なもの

- Xの完全な検索: 公式API credentialが必要
- 有料SEO検索ボリューム/CPC/広告競合など: 専用データプロバイダ契約が必要
- メール送信、SNS投稿、EC操作など: 各サービスの認証/connectorが必要
- PC電源OFF中の24/7稼働: サーバー/VPS/常時稼働マシンが必要

これらは「推論能力」の不足ではなく、外部API・認証・実行環境の問題です。

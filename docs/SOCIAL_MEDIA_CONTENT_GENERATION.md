# Social Media Content Generation Solution — Technical Specification

This document describes every feature of the obelisk-stamps social-media content generation and publishing pipeline in enough detail that a developer can re-implement the same system on a different platform or in a different language. It is intentionally platform-agnostic about *language* (Python is the reference but never required), and specific about *contracts* — external API endpoints, prompt structures, database schemas, status protocols, file formats, and algorithms.

All `app.py:NNNN` references point to the reference implementation in this repository.

---

## Table of Contents

1. [Overview & High-Level Architecture](#1-overview--high-level-architecture)
2. [Data Model](#2-data-model)
3. [Article CMS](#3-article-cms)
4. [Content Generation Pipeline](#4-content-generation-pipeline)
   - 4.1 [Carousel Image Generator](#41-carousel-image-generator)
   - 4.2 [Cinemagraph Generator](#42-cinemagraph-generator)
   - 4.3 [Narrated Video Generator](#43-narrated-video-generator)
5. [Caption System](#5-caption-system)
6. [Short URL & UTM Attribution](#6-short-url--utm-attribution)
7. [Auto-Publish Architecture](#7-auto-publish-architecture)
8. [Per-Network Workers](#8-per-network-workers)
9. [Scheduling System](#9-scheduling-system)
10. [Cron Endpoints](#10-cron-endpoints)
11. [Engagement Tracking](#11-engagement-tracking)
12. [Activity Log & Posting Log](#12-activity-log--posting-log)
13. [Settings & Secrets Store](#13-settings--secrets-store)
14. [Admin Dashboards](#14-admin-dashboards)
15. [Error Handling & Observability](#15-error-handling--observability)
16. [Reimplementation Notes](#16-reimplementation-notes)

---

## 1. Overview & High-Level Architecture

The solution turns a single editorial article into a complete cross-network social-media campaign:

```
                          ┌──────────────────┐
                          │   Article CMS    │
                          │  (title, body,   │
                          │   cover, slug)   │
                          └────────┬─────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
              ▼                    ▼                    ▼
       ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
       │ Carousel (10 │    │ Cinemagraph  │    │   Narrated   │
       │   images)    │ ─▶ │  (image→     │ ─▶ │     Video    │
       │ gpt-image-1  │    │   video)     │    │  (GPT+TTS+   │
       └──────────────┘    │ Luma/Runway  │    │   Whisper+   │
                           └──────────────┘    │   FFmpeg)    │
                                               └──────┬───────┘
                                                      │
                                                      ▼
                                         ┌────────────────────────┐
                                         │   Auto-Publish Runner  │
                                         │ Sequential per-network │
                                         │   posting with status  │
                                         │ tracking & activity log│
                                         └───┬──────┬──────┬──────┘
                                             │      │      │
                                          IG/FB/  YouTube  ...12+
                                          X/...   Shorts  networks
                                             │      │      │
                                             ▼      ▼      ▼
                                         ┌────────────────────────┐
                                         │      Posting Log       │
                                         │  (ML training record)  │
                                         └────────────────────────┘
                                             │
                                             ▼
                                         ┌────────────────────────┐
                                         │   Engagement Polling   │
                                         │ + Bulk metrics import  │
                                         │  → Analytics Dashboard │
                                         └────────────────────────┘
```

Core invariants the rest of the system depends on:

- **An article is the unit of campaign.** All assets, captions, and posts are keyed by `article_id`.
- **Every long-running job is a background thread** with status reported via a key-value `site_settings` row (`*_status_*`) and live-streamed to a per-article activity log.
- **Every successful post writes to `posting_log`** so we can compute "which networks has this article been on" without trusting any individual platform's API.
- **Auto-publish is a sequential, network-by-network sweep** that reads a saved action list and ALWAYS includes YouTube (force-injected if missing).
- **Scheduling fires on a fixed slot system** (10:00 and 14:00 UTC by default), filled by a cron-driven pre-pass that auto-schedules every "ready" draft into the next free slot.

External dependencies the reference uses:

- OpenAI: `gpt-4.1-mini` (script + captions + shortening), `gpt-image-1` (carousel images), `tts-1` / `tts-1-hd` (narration audio), `whisper-1` (word-level caption timing)
- Luma Labs `ray-2` and/or Runway `gen4_turbo` (cinemagraph image→video) — both optional, only one is required
- FFmpeg + ffprobe (local) — for video assembly
- Google Cloud Storage (or any S3-compatible bucket) — for asset storage with local-filesystem fallback
- MySQL — for relational state (articles, engagement, posting log, settings)

---

## 2. Data Model

Six tables are central to the content-generation pipeline.

### 2.1 `articles`

| Column | Type | Notes |
|---|---|---|
| `id` | INT PK | |
| `title`, `subtitle`, `slug`, `excerpt`, `content` | TEXT | `content` is HTML |
| `is_published` | BOOL | |
| `published_at`, `scheduled_publish_at`, `created_at`, `updated_at` | DATETIME | UTC |
| `image_url` | VARCHAR | Cover image (GCS URL or `/static/...`) |
| `show_slideshow` | BOOL | Whether to render the carousel on the public article page |
| **Carousel** | | |
| `carousel_images` | JSON TEXT | `["url1", "url2", ...]`; indices 0–9 are live, 10+ archived |
| `carousel_prompts` | JSON TEXT | Image prompts, parallel to `carousel_images` |
| `carousel_punchlines` | JSON TEXT | One-line captions per slide, parallel array |
| `carousel_style` | TEXT | Last-used style suffix appended to prompts |
| `carousel_created_at` | JSON TEXT | ISO-8601 timestamps per slide |
| **Cinemagraph** | | |
| `carousel_cinemagraphs` | JSON TEXT | MP4 URLs per slide (or `null` if no cinemagraph) |
| `carousel_cinemagraph_prompts` | JSON TEXT | Motion prompts per slide |
| `carousel_cinemagraph_created_at` | JSON TEXT | Timestamps per slide |
| `carousel_cinemagraph_log` | TEXT | Last run log (capped at 65 KB) |
| `carousel_cinemagraph_archived` | JSON TEXT | Historical clips |
| **Narrated video** | | |
| `video_narrated_url` | TEXT | Latest run URL (GCS or `/static/...`) |
| `video_narrated_script` | JSON TEXT | Latest run segment array |
| `video_narrated_runs` | JSON TEXT | All runs `[{ts, ts_label, url, params, segments, cinemagraph_slides}, ...]` |
| `video_narrated_status` | VARCHAR | `running:*` / `error:*` / NULL |
| `video_narrated_log` | TEXT | Last run log (capped at 65 KB) |
| `video_ai_url`, `video_ai_status` | TEXT / VARCHAR | Optional AI cinemagraph (Runway/Luma) — separate from narrated |

### 2.2 `article_engagement`

Append-only point-in-time snapshots. Latest per `(article_id, platform)` is taken via `MAX(fetched_at)` correlated subquery — never aggregate across multiple snapshots or you'll double-count.

| Column | Type | Notes |
|---|---|---|
| `id` | INT PK | |
| `article_id` | INT | FK→articles |
| `platform` | VARCHAR(20) | `yt`, `ig`, `fb`, `x`, `bluesky`, `tiktok`, `pinterest`, `threads`, `linkedin`, `reddit`, `telegram`, `vimeo`, `mastodon`, `vk`, `tumblr` |
| `content_type` | VARCHAR(20) | `narrated_video`, `carousel`, `cinemagraph`, `manual` |
| `post_id`, `permalink` | VARCHAR | Post identifiers on the target network |
| `likes`, `views`, `shares`, `comments`, `saves`, `clicks`, `impressions`, `reach` | INT | Network-specific metric set |
| `fetched_at` | DATETIME UTC | Snapshot time |
| `created_at` | DATETIME | Auto |

### 2.3 `posting_log`

Canonical record. Every platform worker writes exactly one row on success. This is the source of truth for "did we post this?".

| Column | Type | Notes |
|---|---|---|
| `id` | INT PK | |
| `article_id` | INT | |
| `platform` | VARCHAR(20) | Same codes as `article_engagement` |
| `content_type` | VARCHAR(20) | `narrated`, `carousel`, `cine`, etc. |
| `post_id`, `permalink`, `caption`, `hashtags` | TEXT | Hashtags extracted via `#\w+` regex from caption |
| `posted_at` | DATETIME UTC | |
| `posted_day_of_week`, `posted_hour` | TINYINT | For ML feature engineering |
| `posted_is_weekend` | BOOL | |
| `article_title`, `article_slug`, `article_word_count`, `image_count`, `video_duration_seconds` | mixed | Snapshotted from article at post time |

Indexes on `article_id`, `platform`, `posted_at`, `(posted_day_of_week, posted_hour)`.

### 2.4 `short_links`

| Column | Type | Notes |
|---|---|---|
| `id` | INT PK | |
| `code` | VARCHAR(8) UNIQUE | 6-char alphanumeric, deterministic per (article_id, platform) |
| `article_id` | INT | |
| `platform` | VARCHAR(20) | The network this short was minted for |
| `target_url` | VARCHAR(1000) | Full URL with UTM params already appended |
| `click_count` | INT | Total clicks (incl. bots) |
| `click_count_human` | INT | Bot-filtered |
| `created_at` | DATETIME | |

Unique constraint on `(article_id, platform)` — exactly one short per article+platform pair so the same `/a/{code}` always resolves to the same target.

### 2.5 `scheduled_publish_audit`

Optional but useful: a row for every schedule mutation (set/cleared/published) with `source` (manual, cron, auto), `old_at`, `new_at`, `note`, and `created_at`. Enables auditing why an article fired when it did.

### 2.6 `site_settings`

Generic key-value store. PRIMARY KEY `(key)`, LONGTEXT value. Used for:

- Tokens / API credentials cached after OAuth flows (`youtube_refresh_token`, `pinterest_access_token`, etc.)
- Global prompts (`ig_caption_prompt`) and per-article overrides (`ig_caption_prompt_{article_id}`)
- Schedule trackers (`daily_media_last_run_date`, `cron_last_hit_at`)
- Auto-publish action sequence (`auto_publish_actions`)
- Per-run status/result keys for every worker (`*_status_{article_id}_{run_ts}`, `*_result_{article_id}_{run_ts}`, post-id markers like `ig_narrated_post_id_*`)
- Activity logs (`ig_activity_log_{article_id}` — JSON array capped to 50 entries)
- Cron logs (`cron_hit_log`, `daily_media_run_log`)

Accessors are `get_setting(key, default=None)` and `set_setting(key, value)` (which uses `INSERT … ON DUPLICATE KEY UPDATE`).

---

## 3. Article CMS

Articles are the seed of every campaign. The CMS has three states:

- **Pre-draft pipeline** — title/excerpt brainstormed, no body yet
- **Draft** (`is_published=0`) — content written, assets being generated
- **Published** (`is_published=1`) — live on the site, eligible for auto-publish

Key admin routes:

- `GET /admin/articles` — main list with tabs (Published / Drafts / Pre-Draft Pipeline)
- `POST /admin/articles/new` — create
- `GET /admin/articles/<id>/edit` — edit page (carousel + cinemagraph + narrated video panels inline)
- `POST /admin/articles/<id>/save` — save fields
- `POST /admin/articles/<id>/schedule-publish` — manually set `scheduled_publish_at`
- `POST /admin/articles/<id>/auto-schedule` — fill the next free slot (rejects if media incomplete)
- `POST /admin/articles/<id>/toggle-slideshow` — enable/disable public-page slideshow

Article eligibility for auto-publish (`_article_eligible_for_auto_publish_actions`, app.py:4486):

```
carousel_images decoded ≥ 10 items
AND video_narrated_url is non-empty
```

Published-state transitions go through `_sweep_scheduled_publishes()` (see §9.2).

---

## 4. Content Generation Pipeline

Three generators run independently and idempotently, each producing one columnar artifact set on the `articles` row.

### 4.1 Carousel Image Generator

Produces 10 square images per article, one per "scene".

**Entry point:** `POST /admin/articles/<id>/generate-carousel` (Server-Sent Events, streaming progress). Also invoked from the daily-media cron (§10.2) for drafts missing images.

**Two-phase algorithm:**

1. **Scene planning with GPT-4.1-mini.** A single chat completion produces a JSON array of N objects (`N` = number of empty slots, normally 10):

   *System prompt* — instructs the model to be a "visual storyteller and Instagram content strategist", break the article into N sequential scenes, output for each (a) a detailed DALL-E image prompt and (b) a punchline ≤15 words, and ensure all N prompts share a consistent art style.

   *User prompt* — article title + the article content (HTML stripped, max 6000 chars) + the request for N scenes.

   *Response format* — `{"type": "json_object"}` constrained, returning `[{"prompt": "...", "punchline": "..."}, ...]`. Prompts and punchlines are stored as `carousel_prompts` and `carousel_punchlines` JSON arrays.

2. **Image generation per slide.** For each slot index `i`:
   - Append a global style suffix to the prompt (configurable, default = a vintage-editorial / postage-stamp engraving aesthetic with a fixed palette: navy, burgundy, gold, ivory).
   - Call OpenAI Images API: `model=gpt-image-1`, `size=1024x1024`, `quality=medium` (≈ $0.04/image), `n=1`.
   - Persist bytes to GCS at `articles/{article_id}/carousel/image_{i+1}.png` and/or local fallback `static/articles/{article_id}/carousel/image_{i+1}.png`.
   - Persist each image's URL into `carousel_images[i]` immediately (incremental save survives connection interruption).
   - On `BadRequestError` (content policy): log per-image, continue. On `RateLimitError`: halt the batch. On other errors: log and continue.

**Single-slide regenerate:** `POST /admin/articles/<id>/regenerate-carousel-image` with `{index, prompt, punchline}`. Archives the old image/prompt/punchline to the end of the arrays (indices ≥10 are the archive), generates a new image with a timestamped filename, and updates the live slot.

**Style override system:** A per-article `carousel_style` column stores the last-used style suffix so regenerations match.

### 4.2 Cinemagraph Generator

Optional layer that turns each static carousel image into a 5-second looping video clip for the narrated video to use instead of a Ken Burns pan.

**Entry point:** `POST /admin/articles/<id>/generate-cinemagraphs` with `{video_model: "luma"|"runway", global_prompt, prompts: []}`. Spawns a background thread; status is reported via `site_settings.cinemagraph_status_{article_id}`.

**Provider abstractions:**

- **Luma `ray-2`** (default): POST to `https://api.lumalabs.ai/dream-machine/v1/generations` with `{model: "ray-2", aspect_ratio: "1:1", duration: "5s", keyframes: {frame0: {type: "image", url}}, prompt}`. Poll on `state ∈ {queued, dreaming, completed, failed}`; MP4 URL is at `assets.video`. Bearer token auth.
- **Runway `gen4_turbo`**: POST `{model: "gen4_turbo", duration: 5, ratio: "960:960", promptImage, promptText}`. Poll on `status ∈ {PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED}`; MP4 at `output[0]`.

Both providers are wrapped to a unified `submit(image_url, prompt) → task_id` and `poll(task_id) → (state, mp4_url, failure_reason)` interface.

**Prompt resolution hierarchy:**

1. Per-slide prompt in the request body
2. Request-level `global_prompt`
3. `site_settings.cinemagraph_prompt_{article_id}` (per-article override)
4. `site_settings.cinemagraph_prompt` (global default)
5. Hard-coded default: *"Subtle gentle atmospheric motion, cinematic still life, minimal movement, soft breathing effect, loop-friendly"*

**Storage:** MP4 files at `articles/{article_id}/cinemagraph/slide_{N}_{unix_ts}.mp4` (GCS) with `/static/articles/{article_id}/cinemagraph/...` local fallback. URLs land in `carousel_cinemagraphs[i]`.

**Runtime characteristics:**

- 6-second poll interval between status checks
- Stops after **3 consecutive slide failures** to cap costs
- Aborts immediately on auth or billing errors (don't burn credits retrying)
- Cancellable via a `cinemagraph_cancel_{article_id}` flag checked between slides
- Each slide regeneration auto-archives the prior clip into `carousel_cinemagraph_archived` (JSON array of `{url, prompt, created_at, archived_at, slot_source}`)
- Run log appended to `carousel_cinemagraph_log` (DB, capped 65 KB) and full-fidelity copy uploaded to GCS as `articles/{id}/cinemagraph/run_{ts}.log`

### 4.3 Narrated Video Generator

The heaviest piece. Turns 10 carousel images + 10 (optional) cinemagraphs into a single vertical MP4 with voice-over and burned-in captions, ready for Reels/Shorts/TikTok.

**Entry point:** `POST /admin/articles/<id>/generate-narrated-video` with an optional config dict overriding any of the defaults below. Also invoked from the daily-media cron for drafts with carousel but no video.

**Default configuration:**

```
format:        "vertical"      # → 720x1280; "square" → 720x720
voice:         "onyx"          # OpenAI TTS voice
tts_model:     "tts-1"         # or "tts-1-hd"
script_len:    "medium"        # short=20-40w / medium=40-70w / long=70-100w per slide
kb_speed:      "slow"          # Ken Burns zoom rate: slow=0.0010 / medium=0.0015 / fast=0.0025
speed:         1.35            # final playback speed multiplier (clamped to [0.5, 3.0])
crf:           23              # H.264 quality (lower = better)
fps:           25              # output frame rate
captions:      true            # burn-in subtitles
caption_style: "tiktok"        # "tiktok" or "minimal"
```

Status is reported through `articles.video_narrated_status` with the protocol:
`running:script` → `running:tts:N/10` → `running:whisper:N/10` → `running:clip:N/10` → `running:final` → `NULL` (success) or `error:<reason>` (failure). The route refuses to spawn a new worker if status already starts with `running` (prevents double-fires).

**Six-stage pipeline:**

**Stage 1 — Narration script (GPT-4.1-mini).** One chat completion. Prompt:

```
You are writing a voiceover script for a {n}-slide Instagram carousel about
the article: "{title}".

For each slide, write a narration segment of {word_range} words that matches
the scene description and punchline. The narration should sound natural when
read aloud as a voiceover.

Return ONLY a JSON object in this exact format: {"segments": [...]}

Slide descriptions:
Slide 1: {carousel_prompt_1} — Punchline: {carousel_punchline_1}
...
```

Response constrained via `response_format={"type": "json_object"}`. Fallback chain: if `segments` key missing, use first list-valued key; if still not a list, use first string as a single segment; if a segment has <3 words, substitute the corresponding punchline; pad/truncate to exactly `n_slides`.

**Stage 2 — TTS per segment.** For each segment text, call OpenAI `audio.speech.create(model=tts_model, voice=voice, input=seg, response_format="mp3")`. Persist to `static/articles/{id}/video/audio/segment_{N}.mp3`. Probe duration via ffprobe (fallback: `ffmpeg -i` regex on `Duration:`).

**Stage 3 — Whisper transcription** (only if `captions=true`). For each MP3, call `audio.transcriptions.create(model="whisper-1", response_format="verbose_json", timestamp_granularities=["word"])`. Store word arrays `[{word, start, end}, …]` per segment. If Whisper is disabled or fails, captions get evenly-distributed estimated timings.

**Stage 4 — Per-clip FFmpeg render.** For each of the 10 slides:

- **Build the ASS subtitle file** if captions are on. Sentences are split on `[.!?]\s+`, grouped in pairs (so each caption stays on-screen long enough to read), and timed against the Whisper word timestamps. Style differs by `caption_style`:
  - `tiktok`: Arial 40pt (vertical) / 34pt (square), bold, white text with 3px black outline + 2px shadow, fade-in `{\fad(80,60)}`, bottom margin 18% (vertical) / 10% (square).
  - `minimal`: Same but smaller (32/28pt), no bold, 2px outline, no shadow, no fade.

- **Cinemagraph branch** (slide has a working cinemagraph URL):
  ```
  ffmpeg -y -stream_loop -1 -t {dur} -i {cine.mp4} -i {audio.mp3}
    -filter_complex "[0:v]fps=12,scale={S}:{S}:flags=fast_bilinear,crop={W}:{H},setsar=1{subs}[v]"
    -map [v] -map 1:a -c:v libx264 -preset ultrafast -crf {crf}
    -c:a aac -shortest clip_{i}.mp4
  ```
  where `S = max(W, H)` and `{subs}` is `,subtitles='{escaped_ass_path}'` if captions on.

- **Ken Burns branch** (still image):
  ```
  ffmpeg -y -framerate 12 -loop 1 -t {dur} -i {image.png} -i {audio.mp3}
    -filter_complex "[0:v]fps=12,scale={ZW}:{ZH}:force_original_aspect_ratio=increase:flags=fast_bilinear,
                     crop={ZW}:{ZH},crop={W}:{H}:x='{px}':y='{py}',setsar=1{subs}[v]"
    -map [v] -map 1:a -c:v libx264 -preset ultrafast -tune stillimage -crf {crf}
    -c:a aac -shortest clip_{i}.mp4
  ```
  Zoom factor is capped at 1.3×; the crop pan equations interpolate from zoomed-in to zoomed-out across the audio duration.

Per-clip timeout 120s (Ken Burns) or 240s (cinemagraph). Subtitle escaping is non-obvious — on Windows, drive colons and backslashes need libavfilter-specific escaping: `path.replace("\\", "/").replace(":", r"\:")` then wrap the whole thing in single quotes inside the filter argument.

**Stage 5 — Concat.** Write a concat-demuxer manifest:
```
file '/absolute/path/clip_1.mp4'
file '/absolute/path/clip_2.mp4'
...
```
Then `ffmpeg -y -f concat -safe 0 -i manifest.txt -c copy narrated_{ts}.mp4`. Stream-copy (no re-encode).

**Stage 6 — Speed-up + upload.** If `speed != 1.0`, re-encode with `setpts=PTS/{speed}` + `atempo={speed}` (atempo preserves pitch). Upload to GCS at `articles/{id}/video/narrated_{ts}.mp4`. Append a run object to `video_narrated_runs`:

```json
{
  "ts": 1716574818,
  "ts_label": "26 May 2026, 14:30",
  "url": "https://storage.googleapis.com/.../narrated_1716574818.mp4",
  "params": {"format":"vertical","voice":"onyx",...},
  "segments": ["…","…",…],
  "cinemagraph_slides": [1, 3, 5]
}
```

Set `video_narrated_url` to the new URL, clear `video_narrated_status`, persist the log to DB (capped 65 KB) and to GCS (full).

All intermediate files (per-segment MP3s, ASS subtitle files, per-slide MP4s, the manifest, and pre-speedup MP4 if speed≠1) are cleaned up in a `finally` block.

---

## 5. Caption System

### 5.1 Long-form caption generation

`_generate_caption_for_article(article_id)` produces one Instagram-style caption per article. Used as-is by Instagram, Facebook, YouTube, TikTok, LinkedIn, Reddit, Telegram, and Vimeo; passed through the shortener for X, Threads, Bluesky, Mastodon, and Pinterest.

- Model: `gpt-4.1-mini`, `max_tokens=600`
- System prompt: from `site_settings.ig_caption_prompt_{article_id}` (per-article override) or `site_settings.ig_caption_prompt` (global default). The default prompt is provided at deployment time and instructs the model on voice, hashtag policy, and CTA framing.
- User message: `Article title: {title}\n\nArticle summary: {plain}` where `plain` is the article body, HTML stripped, whitespace normalized, truncated to 500 chars.
- After generation: strip any trailing "Want to read the full article?" outro the model added, strip any bare `{site_url}/articles/{slug}` URL, then append `\n\n{short_url}` where `short_url` comes from `make_short_url(article_id, "ig")` (§6).
- Fallback: if OpenAI is unavailable, return `{title}\n{site_url}/articles/{slug}`.

### 5.2 Per-platform shortening

```
_SHORTEN_PLATFORM_LIMITS = {
    "x":         280,
    "bluesky":   300,
    "threads":   500,
    "mastodon":  500,
    "pinterest": 800,
}
```

`_shorten_caption_for_platform_inline(article_id, source, platform)`:

- Model: `gpt-4.1-mini`, `max_tokens=600`
- Reserves space for the CTA: `target_body_chars = max(80, max_chars - len("\n\n👉 {short}") - 5)`
- Strips existing "Want to read…" CTAs and bare URLs from the source first
- System prompt instructs: preserve voice/emojis/hashtags, drop hashtags first if needed, never include URLs, output ≤ `target_body_chars` chars, return ONLY caption text
- User message: `Platform: {p}\nCharacter limit: {max}\nBody budget: {body}\n\n---\n{source}`
- Hard-truncate output to `target_body_chars` if the model exceeded its budget
- Append CTA: `caption + "\n\n👉 " + short_url`
- Hard-truncate final to `max_chars` as a last safety net
- On any error: return source unchanged

Shortened captions are cached per-run in the auto-publish runner to avoid paying for the same generation twice when two tight-budget platforms run in the same sweep.

---

## 6. Short URL & UTM Attribution

A small in-house bitly clone, tightly coupled with the per-platform attribution UTM scheme.

### 6.1 Short code generation

`make_short_url(article_id, platform_key)`:

1. Check `short_links` for an existing row keyed by `(article_id, platform_key)` (unique constraint). If found, return `{site_url}/a/{code}`.
2. Otherwise, fetch the article slug and synthesize a deterministic 6-char code: SHA-256 hash of `f"{article_id}:{platform_key}"`, then map 6 pairs of hex digits into the 62-character alphanumeric alphabet (`A–Z a–z 0–9`).
3. Build the target URL: `{site_url}/articles/{slug}?utm_source={mapped}&utm_medium=social&utm_campaign={slug}` where `mapped` comes from `_UTM_PLATFORM_MAP` (e.g. `yt → youtube`, `x → twitter`, `ig → instagram`).
4. `INSERT` the new row. On unique-key collision (race), re-query to fetch the winner's code.

The deterministic hash means short codes are stable: regenerating the same `(article_id, platform)` always yields the same `/a/...`. This matters because captions are baked into platform-stored posts — you can't change the short URL after publishing without breaking already-posted content.

### 6.2 Resolution endpoint

`GET /a/<code>`:

1. Look up the `short_links` row by code; 404 if missing.
2. Increment `click_count` always.
3. Check the User-Agent against `_BOT_UA_PATTERNS` (a list including all major search-engine crawlers, social-preview fetchers like `facebookexternalhit`/`telegrambot`, AI scrapers like `gptbot`/`claudebot`, headless-browser indicators like `puppeteer`/`playwright`, dev tools like `curl`/`wget`/`python-requests`, and monitoring like `uptime`/`pingdom`). If not a bot, also increment `click_count_human`.
4. Parse UTM params off the target URL and stash them in the user's session for purchase attribution.
5. Issue a 302 redirect to `target_url`.

Engagement dashboards only ever report `click_count_human` for the "Link Clicks" column — bot traffic is tracked but invisible to editorial users.

---

## 7. Auto-Publish Architecture

The auto-publish runner (`_run_auto_publish_actions(article_id)`, app.py:4578) is the orchestrator. It is invoked from:

1. The scheduled-publish sweep (§9.2) for each article it just published, **if** the article is eligible (≥10 carousel images + narrated video URL).
2. A manual admin button (for retrying after editing the article or adding a network).

### 7.1 Saved action sequence

The runner reads `site_settings.auto_publish_actions`, a JSON array such as:

```json
[
  {"text": "Post to YouTube",  "enabled": true},
  {"text": "Post to Instagram","enabled": true},
  {"text": "Post to Facebook", "enabled": true},
  {"text": "Post to X",        "enabled": true},
  ...
]
```

Network detection is text-based (`"post to <network>"` substring match, case-insensitive), normalized against `_AUTO_PUB_WORKER_MAP` keys. Any unrecognized action is ignored.

**Force-inject policy:** YouTube is always added to the sequence if not already present. The YouTube branch is a safe no-op when there is no `youtube_refresh_token` configured (logs "skipped" and continues), so force-inject does no harm.

### 7.2 Execution

The runner:

1. Reads the latest narrated run from `video_narrated_runs[0]` to get `run_ts` and `video_url`.
2. Generates the long-form caption once via `_generate_caption_for_article(article_id)`.
3. Creates a per-run cache `{platform: shortened_caption}` filled lazily on first need.
4. Logs the queued network sequence to the activity log with title `"Auto-Publish Sequence Started"`.
5. Iterates the network list **sequentially** (synchronous; one network blocks the next). For each network:
   - Compose status_key as `{prefix}{article_id}` (for `youtube`) or `{prefix}{article_id}_{run_ts}` (for the others), where `prefix` comes from `_AUTO_PUB_STATUS_PREFIX`. Delete any stale value.
   - Pick the caption (full or shortened, depending on `_SHORTEN_PLATFORM_LIMITS` membership).
   - Log `Auto-Publish → {net}` start event to the activity log.
   - Dispatch to the worker — for YouTube call `_upload_to_youtube_worker(article_id, video_url, title, desc, run_ts, refresh_token)` directly; for everyone else call `globals()[_AUTO_PUB_WORKER_MAP[net]](article_id, video_url, caption, run_ts)`.
   - When the worker returns, read the final status_key value, log `Auto-Publish → {net} done` with that value.
   - On any exception: log `Auto-Publish → {net} crashed` with traceback and continue to the next network. **One platform failure never aborts the rest of the sequence.**
6. After the loop: log `Auto-Publish Sequence Complete`.

### 7.3 Worker contract

Every per-network worker conforms to this contract:

```
def _post_narrated_<net>_worker(article_id, video_url, caption, run_ts):
    # Set status_key = {prefix}{article_id}_{run_ts}
    # Update status_key as work progresses: running:download, running:upload, etc.
    # On success:
    #   - Write site_settings[{platform}_narrated_post_id_{article_id}_{run_ts}] = post_id
    #   - log_social_post(article_id, '{platform_code}', 'narrated', post_id, permalink, caption)
    #   - Delete the status_key
    # On failure:
    #   - Set status_key = "error:<reason>"
    #   - Do NOT call log_social_post
```

The result key naming convention (`{platform}_narrated_post_id_*`) is what makes the published-networks detection (§14.1) work. The status key is what the UI can poll to display "in flight" state.

---

## 8. Per-Network Workers

Detailed contract per network. All workers receive `(article_id, video_url, caption, run_ts)` and post a narrated video. Where the network has a separate carousel posting path (Instagram primarily), it's noted.

### 8.1 YouTube (`_upload_to_youtube_worker`, app.py:8442)

- **Auth:** OAuth 2.0 with stored refresh token. `youtube_refresh_token` lives in `site_settings`, written after the admin completes the OAuth dance. App credentials in env: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`.
- **Endpoints:**
  - `POST https://oauth2.googleapis.com/token` (grant_type=refresh_token) → access token
  - `POST https://www.googleapis.com/upload/youtube/v3/videos?uploadType=resumable&part=snippet,status` with metadata body → returns `Location: <upload_uri>` header
  - `PUT <upload_uri>` with binary MP4 → returns `{id: ...}`
- **Metadata:** Title (≤100 chars, taken from article title), description (≤5000 chars, full caption), category 26 (Howto & Style), tags `["stamps", "philately", "shorts"]`, privacy `public`.
- **Status keys:** `yt_status_{article_id}` with values `running:token / running:download / running:upload / running:publish / running:poll`.
- **Result key:** `yt_result_{article_id}_{run_ts}` containing `done:{youtube_url}` or `error:{reason}`.
- **Post-id markers:** `youtube_video_id_{article_id}_{run_ts}` written separately (used by the engagement poller).
- **Title / description persistence (two key conventions — read with fallback):**
  - The manual `POST /admin/articles/<id>/post-to-youtube` route writes the admin-entered title/description to `yt_title_{article_id}_{run_ts}` and `yt_desc_{article_id}_{run_ts}` BEFORE invoking the worker.
  - The worker, after a successful upload, writes the *actual* posted title/description to `yt_narrated_title_{article_id}_{run_ts}` and `yt_narrated_caption_{article_id}_{run_ts}` (the `*_narrated_*` convention matches every other narrated-platform worker).
  - **Auto-publish does not go through the manual route**, so for auto-published videos only the `yt_narrated_*` keys exist.
  - Read paths (`/youtube-status`, `/delete-youtube-post`, `/archive-youtube-post`) MUST therefore check the manual key first and fall back to the worker key (`get_setting(yt_title) or get_setting(yt_narrated_title) or ""`). Otherwise the Edit-page panel renders empty Title/Description textareas for any auto-published video.
- **Failure modes:** Token refresh failure surfaces as `error:reconnect:…` and the admin sees a "reconnect YouTube" prompt. Upload errors propagate the YouTube error body.
- **Force-injected** into every auto-publish sequence (§7.1).

### 8.2 Instagram Reel (`_post_narrated_reel_worker`, app.py:9611)

- **Auth:** Long-lived Page Access Token from a Meta Business System User. Env: `IG_USER_ID`, `IG_ACCESS_TOKEN`.
- **Endpoints (Graph API v21.0):**
  - `POST /v21.0/{IG_USER_ID}/media` with `media_type=REELS&video_url={video_url}&caption={caption}` → returns `creation_id`
  - `GET /v21.0/{creation_id}?fields=status_code` → poll until `FINISHED`
  - `POST /v21.0/{IG_USER_ID}/media_publish?creation_id={id}` → returns `{id: post_id}`
- **Caption budget:** 2200 chars, full long-form caption.
- **Status keys:** `ig_narrated_status_{article_id}_{run_ts}` with `running:container / running:poll / running:publish`.
- **Result key:** `ig_narrated_post_id_{article_id}_{run_ts}`.
- **Polling:** 5-minute deadline, 5–30s intervals (exponential).
- **Carousel path is separate:** `_post_to_instagram_worker` posts up to 10 carousel slides (with optional cinemagraph videos) using a different two-step "create container per slide → create parent carousel container → publish" flow. It uses status key `ig_post_status_*` and result key `ig_carousel_post_id_*`.

### 8.3 Facebook (`_post_narrated_fb_worker`, app.py:10513)

- **Auth:** Page Access Token (env `FB_PAGE_ID`, `FB_PAGE_ACCESS_TOKEN`).
- **Endpoint:** `POST https://graph.facebook.com/v21.0/{FB_PAGE_ID}/videos` with `file_url={video_url}&description={caption}`.
- **Critical detail:** Uses `file_url` (Facebook fetches the video itself from GCS) rather than direct multipart upload. This avoids Cloud Run / serverless request-body size caps for the platform host.
- **Status key:** `fb_narrated_status_{id}_{run_ts}` (`running:upload`).
- **Result key:** `fb_narrated_post_id_{id}_{run_ts}`.
- **Caption budget:** Unbounded in practice; pass full caption.
- **No polling step** — the API returns immediately with the post ID.

### 8.4 X / Twitter (`_post_narrated_x_worker`, app.py:11086)

- **Auth:** OAuth 1.1a (4-credential: consumer key+secret + access token + secret). The reference uses the v2 Media Upload API (REST-style, not legacy v1.1 form-encoded).
- **Endpoints (host `api.x.com`):**
  - `POST /2/media/upload/initialize` with JSON body `{total_bytes, media_type:"video/mp4", media_category:"tweet_video"}` → `{data:{id:<media_id>, processing_info:{...}}}`
  - `POST /2/media/upload/{media_id}/append` with multipart form `{segment_index: N, media: <chunk bytes>}`. **Chunk size must be ≤ 1 MB** — v2 enforces a tighter limit than v1.1 (5 MB) and 5 MB chunks return HTTP 413.
  - `POST /2/media/upload/{media_id}/finalize` → `{data:{processing_info:{state, check_after_secs}}}` or 200 with no `processing_info` if processing is synchronous.
  - `GET /2/media/upload?command=STATUS&media_id={id}` → poll `processing_info.state`. The path-based variant `GET /2/media/upload/{id}` is the documented alternative but historically returned 404 in production. The worker tries the query-form first and falls back to the path form for robustness.
  - `POST https://api.twitter.com/2/tweets` with JSON body `{text, media:{media_ids:[media_id]}}`
- **Caption budget:** 280 chars (AI-shortened version, §5.2).
- **Status key:** `x_narrated_status_{id}_{run_ts}` (`running:download → running:upload_init → running:upload_append → running:upload_finalize → running:processing → running:tweeting`).
- **Result key:** `x_narrated_tweet_id_{id}_{run_ts}`.

**Critical: large-video processing wait pattern.** X's transcoder is asynchronous and takes 30–300s depending on file size (a typical 60-second 720p narrated video is ~80–150 MB and processes in 60–180s; up to 4 min during peak load). The wait pattern:

1. After FINALIZE, sleep `min(60, check_after_secs)` or 15s default (don't skip the initial sleep even if FINALIZE returned no `processing_info` — X often omits the block for asynchronous videos).
2. Poll STATUS for up to **5 minutes total**. On `succeeded` → proceed. On `failed` → surface as "Media processing failed" (don't retry; the video was rejected — common causes: codec not H.264, audio not AAC, duration > 140s, file > 512 MB).
3. If both STATUS URL forms return 404 twice in a row, abandon polling and fall through to a "blind wait + retry on `media invalid`" loop with the pattern [0, 30, 60, 90, 120]s on the tweet itself (~5 min of additional retries).

The **"media IDs are invalid"** tweet error from X is overloaded: it's used both for genuinely invalid IDs *and* as a not-ready-yet signal during async processing. Always classify it as a retry candidate until either STATUS reports `failed` or the retry budget is exhausted. Total worst-case timeline is ~10 minutes (5 min STATUS poll + 5 min tweet retries) which catches even queued-during-peak videos.

### 8.5 Threads (`_post_narrated_threads_worker`, app.py:11907)

- **Auth:** Threads Graph API access token (env `THREADS_USER_ID`, `THREADS_ACCESS_TOKEN`).
- **Endpoints:**
  - `POST /v1.0/{user_id}/threads` with `media_type=VIDEO&video_url={url}&text={caption[:500]}` → container_id
  - `GET /v1.0/{container_id}?fields=status,error_message` → poll until `FINISHED`
  - `POST /v1.0/{user_id}/threads_publish?creation_id={id}` → post_id
- **Caption budget:** 500 chars (shortened).
- **Polling:** 60 attempts × 5s = 5 min deadline.
- **Result key:** `threads_narrated_post_id_{id}_{run_ts}`.

### 8.6 Pinterest (`_post_narrated_pinterest_worker`, app.py:12481)

- **Auth:** OAuth 2.0 access token (env `PINTEREST_ACCESS_TOKEN`, `PINTEREST_BOARD_ID`; OAuth client `PINTEREST_CLIENT_ID`/`SECRET` for refresh).
- **Endpoints (API v5):**
  - `POST /v5/media` with `{media_type: "video"}` → `{media_id, upload_url, upload_parameters: {...}}`
  - `POST {upload_url}` with `multipart/form-data` body composed from the returned form fields + the binary video (S3 pre-signed upload)
  - `GET /v5/media/{media_id}` → poll `status` until `succeeded`
  - `POST /v5/pins` with `{board_id, title, description, media_source: {source_type: "video_id", cover_image_url, video_id}}`
- **Caption budget:** 800 chars (shortened).
- **Polling:** 90 × 5s = 7.5 min deadline.
- **Result key:** `pinterest_narrated_pin_id_{id}_{run_ts}`.

### 8.7 TikTok (`_post_narrated_tiktok_worker`, app.py:12990)

- **Auth:** OAuth 2.0 access token (env `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET`, `TIKTOK_ACCESS_TOKEN`).
- **Endpoints (Content Posting API v2):**
  - `POST /v2/post/publish/video/init/` with `{post_info: {title, privacy_level, …}, source_info: {source: "PULL_FROM_URL", video_url}}` → `publish_id`
  - `GET /v2/post/publish/status?publish_id=…` → poll until `PUBLISH_COMPLETE`
- **Caption budget:** Title 150 chars; full caption used as description (not enforced by worker).
- **Source mode:** `PULL_FROM_URL` (TikTok fetches the MP4 itself from GCS).
- **Polling:** 120 × 5s = 10 min deadline.
- **Unaudited-app constraint:** Until your TikTok dev app passes audit, only allowlisted accounts can post — init may fail with a permission error.
- **Result key:** `tiktok_narrated_video_id_{id}_{run_ts}`.

### 8.8 LinkedIn (`_post_narrated_linkedin_worker`, app.py:13453)

- **Auth:** OAuth 2.0 access token (env `LINKEDIN_ACCESS_TOKEN`, `LINKEDIN_ORG_ID`).
- **Endpoints (REST API, headers: `LinkedIn-Version: 202407`, `X-Restli-Protocol-Version: 2.0.0`):**
  - `POST /rest/videos?action=initializeUpload` → upload instructions (chunked URLs)
  - `PUT {upload_url_chunk}` per chunk
  - `POST /rest/videos?action=finalizeUpload` with `{value: {video, uploadToken, uploadedPartIds}}`
  - Poll `GET /rest/videos/{urn}?viewerCountryCode=US` until processed
  - `POST /rest/posts` with the post body referencing the video URN
- **Caption:** Full (no AI shortening).
- **Polling:** 60+ × 10s = 10+ min for transcode.
- **Result key:** `linkedin_narrated_post_id_{id}_{run_ts}`.

### 8.9 Bluesky (`_post_narrated_bluesky_worker`, app.py:14083)

- **Auth:** AT Protocol — handle + app password → session JWT (`com.atproto.server.createSession`). Env `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD`.
- **Two-step service auth for video upload (tricky):**
  - `GET {pds_url}/xrpc/com.atproto.server.getServiceAuth?aud={user_did}&lxm=com.atproto.repo.uploadBlob` → service token
  - `POST https://video.bsky.app/xrpc/app.bsky.video.uploadVideo` with the service token + MP4 bytes → blob ref
- **Post creation:** `POST {pds_url}/xrpc/com.atproto.repo.createRecord` with `{repo: did, collection: "app.bsky.feed.post", record: {text, embed: {$type: "app.bsky.embed.video", video: <blob>, alt}, ...}}`
- **Caption budget:** 300 chars (shortened).
- **Result key:** `bluesky_narrated_post_uri_{id}_{run_ts}` (full AT-URI, not just rkey).

### 8.10 Reddit (`_post_narrated_reddit_worker`, app.py:14625)

- **Auth:** OAuth 2.0 refresh token (env `REDDIT_CLIENT_ID/SECRET/REFRESH_TOKEN/SUBREDDIT`).
- **Endpoint:** `POST https://oauth.reddit.com/api/submit` with form data `{kind: "link", sr: SUBREDDIT, title: caption[:300], url: video_url, ...}`.
- **Posts as a link post** to the configured subreddit (not native video upload).
- **Caption budget:** 300-char title.
- **Result key:** `reddit_narrated_post_id_{id}_{run_ts}`.

### 8.11 Telegram (`_post_narrated_telegram_worker`, app.py:15036)

- **Auth:** Bot token (env `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`).
- **Endpoint:** `POST https://api.telegram.org/bot{token}/sendVideo` with `{chat_id, video: video_url, caption: caption[:1024]}`.
- **Caption budget:** 1024 chars.
- **Result key:** `telegram_narrated_post_id_{id}_{run_ts}`.

### 8.12 Vimeo (`_post_narrated_vimeo_worker`, app.py:15310)

- **Auth:** OAuth 2.0 access token (env `VIMEO_ACCESS_TOKEN`).
- **Endpoints (API v3.4, headers `Accept: application/vnd.vimeo.*+json;version=3.4`):**
  - `POST /me/videos` with `{upload: {approach: "tus", size: bytes}, name, description}` → `{upload: {upload_link}}`
  - `PATCH {upload_link}` with `Upload-Offset` (tus resumable protocol) repeating until full
  - `GET /videos/{id}?fields=transcode.status` → poll until `complete`
- **Caption:** name=first 128 chars, description=full caption.
- **Polling:** 120 × 5s = 10 min for transcode.
- **Result key:** `vimeo_narrated_video_id_{id}_{run_ts}`.

### 8.13 Mastodon (`_post_narrated_mastodon_worker`, app.py:15671)

- **Auth:** Instance access token (env `MASTODON_INSTANCE_URL`, `MASTODON_ACCESS_TOKEN`).
- **Endpoints:**
  - `POST {instance}/api/v2/media` (multipart with `file` = MP4) → `{id, url, ...}`
  - `GET {instance}/api/v1/media/{id}` → poll (returns 206 while processing, 200 when ready)
  - `POST {instance}/api/v1/statuses` with `{status, media_ids:[…], visibility: "public"}`
- **Caption budget:** 500 chars (shortened).
- **Polling:** 60 × 5s = 5 min.
- **Result key:** `mastodon_narrated_post_id_{id}_{run_ts}`.

### 8.14 VK / Tumblr

Workers exist (`_post_narrated_vk_worker`, `_post_narrated_tumblr_worker`) but are **not** in the default auto-publish sequence — they're invoked only from manual admin routes. VK uses an OAuth access token + group ID + a two-step `video.save → upload` flow; Tumblr uses OAuth and the NPF (Neue Post Format) API.

---

## 9. Scheduling System

### 9.1 Publish slots

```
_DAILY_PUBLISH_SLOTS_UTC = [(10, 0), (14, 0)]   # 10:00 and 14:00 UTC
```

Two posts per day. Time zones are UTC throughout (the public site converts to viewer-local in the browser).

### 9.2 Next-free-slot allocator

`_next_free_publish_slot()`:

1. Read the set of `scheduled_publish_at` values from currently-unpublished drafts (normalized to `YYYY-MM-DD HH:MM`).
2. Starting from today, iterate up to 120 days forward. For each day, try `(10, 0)` then `(14, 0)`. Return the first slot that is both strictly in the future AND not in the taken set.
3. Return `None` if no slot found in 120 days (rare; means the queue is 240+ articles deep).

### 9.3 Scheduled-publish sweep

`_sweep_scheduled_publishes()`:

1. Single atomic `UPDATE articles SET is_published=1, published_at=COALESCE(scheduled_publish_at, NOW()), scheduled_publish_at=NULL WHERE is_published=0 AND scheduled_publish_at <= NOW()`.
2. For each newly-published article, if `_article_eligible_for_auto_publish_actions(id)` is true, spawn a daemon thread running `_run_auto_publish_actions(id)`.
3. Append an entry to `scheduled_publish_audit` for each published article.
4. Return the list of `(id, title)` tuples for caller logging.

Idempotent — running the sweep twice in the same second does nothing on the second run.

### 9.4 Ready-draft pre-pass

`_schedule_all_ready_drafts()` (added as a runner pre-pass — see §10.2):

1. Query all unpublished drafts with `scheduled_publish_at IS NULL` AND fully ready (≥10 carousel images + narrated video URL), ordered by `updated_at ASC` (oldest first).
2. For each, call `_next_free_publish_slot()` and assign the result via UPDATE.
3. Log every assignment to `scheduled_publish_audit` with source `_schedule_all_ready_drafts`.

This is the key to filling the schedule weeks ahead: every cron tick, any "media-complete" drafts that don't already have a date instantly get one.

---

## 10. Cron Endpoints

Two public, unauthenticated cron endpoints designed to be hit by an external uptime monitor (UptimeRobot, Pingdom, etc.). Both return plain text and support both `GET` and `HEAD` — many free monitors default to HEAD.

### 10.1 `/cron/publish-scheduled`

- Updates `site_settings.cron_last_hit_at` to current UTC ISO.
- Calls `_sweep_scheduled_publishes()` on every hit (idempotent — safe to call from HEAD).
- Appends to `cron_hit_log` (JSON array in site_settings, capped at 100 entries) the time, method, and list of `{id, title}` for articles published this hit.
- Returns `OK` (text) for GET, empty body for HEAD.
- No auth — worst-case impact of an unauthorised hit is "scheduled articles fire on time".

Recommended frequency: every 5 minutes (so a scheduled article fires within ~5 min of its slot).

### 10.2 `/cron/daily-media-generation`

Once-per-UTC-day media-generation orchestrator. Hit endpoint:

- If `site_settings.daily_media_last_run_date` is already today's UTC date (`YYYY-MM-DD`) AND `?force=1` is not set: respond `ALREADY_RAN_TODAY` and exit.
- Otherwise: write today's date to `daily_media_last_run_date` (so concurrent monitor hits don't re-fire), spawn `_daily_media_generation_runner()` in a background thread, respond `STARTED`.

The runner:

1. **Pre-pass:** `_schedule_all_ready_drafts()` (§9.4).
2. **Generation pass:** `_pick_drafts_needing_media(_DAILY_MEDIA_MAX_ARTICLES)` returns up to 8 drafts (configurable) where `carousel_count < 10 OR video_narrated_url is empty`. Drafts already scheduled fire first (`ORDER BY (scheduled_publish_at IS NULL), scheduled_publish_at ASC, updated_at ASC`).
3. For each picked draft:
   - If missing carousel: run carousel generator synchronously.
   - If missing narrated video but carousel ready: run narrated worker synchronously with `_DAILY_MEDIA_NARRATED_DEFAULT_CFG`.
   - If both complete and no scheduled_publish_at: assign next free slot (catches any draft the pre-pass missed because it newly became ready).
   - If carousel just generated and no cover image set: set cover to slide 2 (slide 1 is sometimes the title intro; slide 2 reads better as a thumbnail). Enable slideshow if not already enabled.
4. Append run summary to `site_settings.daily_media_run_log` (JSON array capped at 30 entries).

Recommended frequency: hit once per day (cron at e.g. 00:30 UTC). The endpoint's self-throttling means hitting it more often is also safe.

---

## 11. Engagement Tracking

Three input mechanisms; one query model.

### 11.1 Per-platform pollers

`admin_engagement_poll_all` iterates all published articles and calls `fetch_all_engagement(article_id)`, which in turn calls per-platform pollers:

- **YouTube:** `GET https://www.googleapis.com/youtube/v3/videos?part=statistics&id=<video_id>` for each `youtube_video_id_*` key found in site_settings. Stats fields `viewCount`, `likeCount`, `commentCount` map to engagement columns.
- **Instagram:** `GET /v21.0/{post_id}/insights?metric=…` for each `ig_narrated_post_id_*` key. Reels metrics: `plays`, `likes`, `comments`, `shares`, `saved`.
- **Facebook:** Similar via Graph API with the FB post id.
- **X:** Tweet lookup via `GET /2/tweets/:id?tweet.fields=public_metrics` for each tweet id.
- **Threads / Pinterest / TikTok / LinkedIn / Bluesky:** Per-platform insights endpoints, all keyed off the result-key markers stored after auto-publish.

Each poll inserts a fresh row into `article_engagement` with `fetched_at=now`. The table is append-only; no UPDATEs.

### 11.2 Manual bulk import

`/admin/engagement/bulk-import` accepts pasted spreadsheet rows for platforms where automated polling is unavailable (or to backfill historical periods).

Request: `{platform: "yt", commit: true|false, rows: [{title, views, likes, comments, shares}, ...]}`

Title matching is the heart of this endpoint. Algorithm:

1. **Normalize:** lowercase, replace any non-`[a-z0-9 ]` with space, strip.
2. **Tokenize:** split on whitespace, drop common stopwords (`the, a, an, of, and, or, to, in, on, for, with, from, by, at, is, as, how, its, it, this, that, through, into, be, are, was, were`), drop tokens of length ≤ 1.
3. **Match tiers in order:**
   - **Exact** — normalized form is a unique key in the article index.
   - **Substring** — exactly one article's normalized title contains the input (or vice versa).
   - **Fuzzy** — token-set scoring: `score = (jaccard + coverage) / 2` where `jaccard = |A∩B| / |A∪B|` and `coverage = |A∩B| / |A|`. Accept the top candidate only if `score ≥ 0.45` AND `(top - runner_up) ≥ 0.10`.
   - **Ambiguous** — more than one substring candidate.
   - **No match** — none of the above.
4. **Dry-run** (commit=false) returns the preview: `{matched: [...], unmatched: [...], ambiguous: [...]}`. Match types are tagged so the UI can show `exact` / `fuzzy` badges.
5. **Commit** (commit=true) inserts one `article_engagement` row per matched article with `content_type='manual'` and current UTC `fetched_at`. **All-zero rows are skipped** to avoid creating no-op snapshots that would muddle the latest-per-platform query.

### 11.3 Per-article metrics modal

`GET /admin/articles/<id>/metrics` returns the latest snapshot per platform for the per-article form. `POST /admin/articles/<id>/metrics` accepts `{entries: [{platform, views, likes, comments, shares}, …]}` and appends one snapshot row per platform.

### 11.4 Latest-snapshot query pattern

This is the single read shape every dashboard uses:

```sql
SELECT … FROM article_engagement e
INNER JOIN (
    SELECT article_id, platform, MAX(fetched_at) AS mf
    FROM article_engagement
    GROUP BY article_id, platform
) latest
ON latest.article_id = e.article_id
   AND latest.platform = e.platform
   AND latest.mf = e.fetched_at
WHERE …
```

Aggregations always go through this subquery; raw `SUM(e.likes)` over the full table double-counts history.

---

## 12. Activity Log & Posting Log

### 12.1 Per-article activity log

`_add_activity_log(article_id, title, body, component=None)` appends to `site_settings.ig_activity_log_{article_id}`:

```json
[
  {"timestamp": "2026-05-26T14:30:00Z", "title": "...", "content": "...", "component": "carousel"},
  ...
]
```

Capped at 50 entries (oldest evicted). `content` truncated to 4000 chars. `component ∈ {carousel, narrated, cinemagraph}` lets the UI tab-filter the log modal.

Used by every long-running worker to mirror progress so the editor can watch live.

### 12.2 Posting log

`log_social_post(article_id, platform, content_type, post_id, permalink, caption)` (app.py:1890):

- Extracts hashtags via `re.findall(r'#\w+', caption)`.
- Looks up article: title, slug, HTML-stripped word count.
- For carousel/cinemagraph content_types, counts images.
- Computes `posted_day_of_week` (Monday=0), `posted_hour` (0–23), `posted_is_weekend`.
- INSERTs one row.

Every successful platform worker calls this exactly once. This is the canonical "did this article publish on this network?" record — used by the Articles list's network-icon column (§14.1).

---

## 13. Settings & Secrets Store

`site_settings` is a generic kv store. Important conventions:

- **Tokens are stored here, not in env**, because most OAuth flows produce tokens at runtime (admin clicks "Connect YouTube") and they need to be refreshable across server restarts. Env vars are limited to credentials seeded at deploy time (client IDs/secrets, page access tokens that never expire, GCS bucket name).
- **Per-article overrides** use the convention `{key}_{article_id}` (e.g. `ig_caption_prompt_42`). The resolver always checks the per-article override first, falls back to the global key, falls back to a hard-coded default.
- **Per-run markers** use `{prefix}_{article_id}_{run_ts}` so multiple narrated runs of the same article don't collide.
- **Logs/state arrays** stored as JSON in the `value` column. Cap with a `_LOG_MAX` constant and roll off the oldest.

Reference helpers:

```
get_setting(key, default=None)        # SELECT value WHERE `key`=%s
set_setting(key, value)               # INSERT … ON DUPLICATE KEY UPDATE
delete_setting(key)                   # DELETE WHERE `key`=%s
```

---

## 14. Admin Dashboards

### 14.1 Published-networks indicator (Articles list + Edit page)

The same indicator appears in two surfaces:

- **Articles list** (`/admin/articles`, Published tab) — one icon row under each article title, so editors can scan dozens of articles and see distribution at a glance.
- **Article edit page** (`/admin/articles/<id>/edit`) — one icon row in the page header next to the title, so editors looking at a single article can confirm where it landed without going back to the list.

Both surfaces render the same shared partial `templates/_published_networks_icons.html`, which takes a `networks` iterable of canonical network names and emits one Font Awesome brand icon per network with the platform's brand colour and a "Published to {Network}" hover title. The partial is the single source of icon truth — adding a new network (e.g. when a new platform worker is wired up) only requires extending the partial's `_net_icons` dict.

Network detection uses **two unioned signals** (run server-side, once per page load):

```sql
SELECT DISTINCT platform FROM posting_log         WHERE article_id = ?
UNION
SELECT DISTINCT platform FROM article_engagement  WHERE article_id = ?
```

The Articles list runs a batched variant (no `WHERE article_id`) and joins by `article_id` in Python; the Edit page runs the per-article version above. The platform code is then mapped to a canonical network name via `_PLAT_CODE_TO_NET`:

```
yt→youtube, ig→instagram, fb→facebook, x→x, bluesky→bluesky,
tiktok→tiktok, pinterest→pinterest, threads→threads,
linkedin→linkedin, reddit→reddit, telegram→telegram, vimeo→vimeo,
mastodon→mastodon, vk→vk, tumblr→tumblr
```

`posting_log` is the canonical signal — every successful platform worker calls `log_social_post` (§12.2). `article_engagement` is the fallback signal for articles whose metrics were imported via bulk import but never had a `log_social_post` row (e.g. historical content published before the system existed, or content posted manually from outside the platform).

**Per-run icons on the edit page** are a separate concern: the Narrated Video Generator panel shows one accordion item per narrated run (see `video_narrated_runs` in §4.3), with a small row of icons in each run's collapsed header. Those icons are live-coloured during posting via JS but are *not* restored on page load — they will be grey for past runs until a backfill from the per-run site-settings markers (`{platform}_narrated_post_id_{article_id}_{run_ts}`, etc.) is wired up. The article-level header indicator above is the source of truth for "did this article ever reach network X".

### 14.2 Drafts list

Columns: Title, Scheduled (UTC, with a `data-utc=` attribute the browser converts to local), Carousel (count 0/10), Narrated (Yes/—), Edit.

When a draft has full media but no schedule, an "Auto-schedule" button appears — clicking it calls `POST /admin/articles/<id>/auto-schedule` which assigns the next free slot.

A "Publish scheduled now" button manually triggers `_sweep_scheduled_publishes()` (same code path as the cron, useful when an admin wants to fire a scheduled-for-tomorrow article immediately).

### 14.3 Analytics dashboard

Two cards: **Platform Performance** (totals per platform, latest snapshot semantics) and **Article Performance** (per-article totals + last-fetched timestamp + "Edit metrics" button opening the per-article modal). Article order matches the Articles list (`ORDER BY updated_at DESC`) so editors can correlate visually.

Top-right: **Bulk Import** button (opens the modal described in §11.2) and **Refresh All Articles** (kicks off `admin_engagement_poll_all`).

### 14.4 Settings page

Contains the auto-publish action editor (drag-and-drop list of "Post to X" rows that gets saved as `auto_publish_actions`), OAuth-connect buttons for every per-network app, manual cron triggers, and editable caption/script prompts.

---

## 15. Error Handling & Observability

A few invariants the system follows everywhere:

- **Background workers never raise to the caller.** Always wrap in `try/except`, log the traceback, mirror to the activity log, and set a status value like `error:<reason>` (truncated to 200 chars).
- **Status keys are always cleared (set to NULL or deleted) on terminal states.** The UI polls the status key; absence means "idle/done", presence means "in flight or errored".
- **Activity logs flush incrementally** so a long-running job's progress is visible while it's still running.
- **Run logs are dual-stored:** truncated to 65 KB in the DB column for quick UI display, full-fidelity copy uploaded to GCS as a `.log` artifact for forensics.
- **Cleanup runs in `finally`:** every worker that creates temp files in the static dir cleans them up regardless of success/failure.
- **Idempotency is preferred over locking.** The publish sweep uses a single atomic UPDATE; the daily cron self-throttles via a date marker; short-link generation handles the unique-key race by re-querying.

---

## 16. Reimplementation Notes

Things to get right when reimplementing on another platform/language.

**Concurrency model:** Every long-running job (carousel, cinemagraph, narrated, every platform worker) MUST run in the background — never in a request handler thread. Whether you use threads, an async runtime, a job queue (Celery / Sidekiq / Hangfire), or a serverless function fan-out is up to you, but the contract has to be the same: synchronous status updates via a shared kv store that the UI can poll.

**Persistence of status:** Don't use in-memory state for run status. The reference uses `site_settings` rows because a server restart mid-narrated-render would otherwise leave the UI thinking it was still running. The reference clears `video_narrated_status` matching `running:%` at startup as a belt-and-braces measure.

**Caption character budgets are platform-specific and change.** Treat `_SHORTEN_PLATFORM_LIMITS` as data, not as code. Re-check the current limits for X, Threads, Mastodon, Bluesky, Pinterest at re-implementation time (they have changed at least three times in the last 18 months).

**OAuth refresh tokens vs page access tokens:**
- YouTube, Reddit, Pinterest, TikTok, LinkedIn, Vimeo: refresh-token grant — store the refresh token, exchange for an access token at use time.
- Facebook + Instagram Reels: long-lived **Page Access Token** derived from a Meta Business System User. This token never expires and bypasses the 60-day refresh problem of user tokens. Use it.
- Threads: long-lived access token via the Meta Threads Graph dance, ~60-day lifetime.
- Bluesky: handle + app password → session JWT every call (don't try to cache the JWT for long).
- Telegram: simple bot token, never expires.
- Mastodon: instance-issued access token, never expires.
- X: OAuth 1.1a 4-credential (consumer key/secret + access token/secret) — set up once and never expires unless revoked.

**File assembly is local, expensive, and CPU-bound.** FFmpeg is single-threaded for most filter graphs at `-preset ultrafast` and uses ~1 core per render. Plan compute accordingly. A 10-slide vertical narrated render takes roughly 2–4 minutes on a 2-vCPU container.

**GCS / object storage as the asset bus.** Almost every per-network worker passes a URL (not bytes) to the target API. This avoids container-host request-body size caps and lets the network fetch on its own bandwidth. Make sure your bucket has public-read or signed URLs your APIs can hit.

**The publish slot system is deliberately rigid.** Two slots/day, 4 hours apart, UTC. Don't make it configurable per article. The fixed cadence is a feature: it means a downstream cron monitor only needs to fire every ~5 minutes to catch every slot.

**The fuzzy-matching thresholds (0.45 floor, 0.10 margin) were tuned empirically against ~50 published articles.** If you reimplement on a different content domain, hand-test against a representative sample and re-tune. The math is symmetric and the algorithm is generic; only the constants are corpus-specific.

**Force-inject YouTube** in the auto-publish runner. Editors forget. The cost of force-inject is zero when YouTube isn't connected (the worker no-ops). The cost of forgetting it is one article per network missing from your most valuable platform.

**ASS subtitle escaping is platform-specific.** On Windows the libavfilter `subtitles=` filter needs `path.replace('\\', '/').replace(':', r'\:')` and the whole thing wrapped in single quotes. On POSIX the path can usually be passed verbatim. Detect and branch.

**The `posting_log` table is your friend.** It's the only place that reliably captures "we posted X to Y at time Z". Engagement tables can be polluted by retries and imports; `auto_publish_actions` can be edited after the fact. `posting_log` is append-only and never lies. Build all your "did we post this?" UI off it.

**Avoid divergent key conventions between manual and auto-publish write paths.** If a platform has both (a) a per-platform manual button on the Edit page that calls a `/post-to-<network>` route AND (b) an auto-publish worker that posts the same content without going through that route, both code paths must persist post metadata (title, caption, post id) under the same `site_settings` keys. Otherwise the Edit page restore logic will silently work for one path and silently fail for the other — symptom is "the caption I see on the actual network is missing from the panel after page reload". The YouTube worker historically wrote `yt_narrated_title_*` / `yt_narrated_caption_*` while the manual route wrote `yt_title_*` / `yt_desc_*`; the read endpoints had to be patched to check both with fallback. When reimplementing: pick **one** key convention per platform and use it from both paths.

---

*End of specification.*

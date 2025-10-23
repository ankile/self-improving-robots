from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote


def _require_pyarrow_dataset():
    try:
        import pyarrow.dataset as ds  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "pyarrow is required for the dataset viewer. Install with "
            "`pip install self-improving-robots[viewer]` or add pyarrow manually."
        ) from exc
    return ds


def _require_flask_components():
    try:
        from flask import Flask, abort, jsonify, render_template_string, send_file
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Flask is required for the dataset viewer. Install with "
            "`pip install self-improving-robots[viewer]` or add Flask manually."
        ) from exc
    return Flask, abort, jsonify, render_template_string, send_file


@dataclass
class EpisodeVideoSegment:
    video_key: str
    relative_path: Path
    from_timestamp: float
    to_timestamp: float

    def as_dict(self) -> Dict[str, Any]:
        start = float(self.from_timestamp)
        end = float(self.to_timestamp)
        duration = max(0.0, end - start)
        rel_posix = self.relative_path.as_posix()
        return {
            "video_key": self.video_key,
            "from_timestamp": start,
            "to_timestamp": end,
            "duration_seconds": duration,
            "url": f"/video/{quote(rel_posix, safe='/')}",
        }


@dataclass
class EpisodeRecord:
    episode_index: int
    length: int
    dataset_from_index: int
    dataset_to_index: int
    tasks: List[str]
    extras: Dict[str, Any]
    video_segments: Dict[str, EpisodeVideoSegment]

    def as_dict(self) -> Dict[str, Any]:
        payload = {
            "episode_index": self.episode_index,
            "length": self.length,
            "dataset_from_index": self.dataset_from_index,
            "dataset_to_index": self.dataset_to_index,
            "duration_seconds": self.extras.get("duration_seconds"),
            "tasks": self.tasks,
            "video_segments": {key: seg.as_dict() for key, seg in self.video_segments.items()},
        }
        for key in ("success", "source", "notes"):
            if key in self.extras:
                payload[key] = self.extras[key]
        return payload


class DatasetIndex:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root.expanduser().resolve()
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_root}")
        info_path = self.dataset_root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"Expected metadata at {info_path}. Ensure this is a LeRobot dataset directory."
            )
        with info_path.open("r", encoding="utf-8") as f:
            self.info: Dict[str, Any] = json.load(f)
        self.fps = self.info.get("fps", 30)
        self.video_keys = [
            key for key, feature in self.info.get("features", {}).items() if feature.get("dtype") == "video"
        ]
        if not self.video_keys:
            raise ValueError("Dataset metadata does not list any video features to visualize.")
        self.video_template = self.info.get("video_path")
        if not self.video_template:
            raise ValueError("Dataset metadata did not include a `video_path` template.")

        self.episodes = self._load_episodes()

    def _load_episodes(self) -> List[EpisodeRecord]:
        ds = _require_pyarrow_dataset()
        episodes_dir = self.dataset_root / "meta" / "episodes"
        if not episodes_dir.exists():
            raise FileNotFoundError(f"Episode metadata directory missing: {episodes_dir}")

        dataset = ds.dataset(str(episodes_dir), format="parquet")
        table = dataset.to_table()
        rows = table.to_pylist()

        records: List[EpisodeRecord] = []
        for row in rows:
            episode_index = int(row.get("episode_index", len(records)))
            length = int(row.get("length", 0))
            dataset_from_index = int(row.get("dataset_from_index", 0))
            dataset_to_index = int(row.get("dataset_to_index", dataset_from_index + length))
            tasks_raw = row.get("tasks") or []
            if isinstance(tasks_raw, str):
                tasks = [tasks_raw]
            elif isinstance(tasks_raw, list):
                tasks = [str(task) for task in tasks_raw]
            else:
                tasks = []

            extras: Dict[str, Any] = {}
            if "success" in row and row["success"] is not None:
                extras["success"] = bool(row["success"])
            if "source" in row and row["source"] is not None:
                extras["source"] = row["source"]
            if "notes" in row and row["notes"]:
                extras["notes"] = row["notes"]

            duration_seconds = length / self.fps if self.fps else None
            extras["duration_seconds"] = duration_seconds

            segments: Dict[str, EpisodeVideoSegment] = {}
            for video_key in self.video_keys:
                chunk_key = f"videos/{video_key}/chunk_index"
                file_key = f"videos/{video_key}/file_index"
                start_key = f"videos/{video_key}/from_timestamp"
                end_key = f"videos/{video_key}/to_timestamp"

                chunk_idx = row.get(chunk_key)
                file_idx = row.get(file_key)

                if chunk_idx is None or file_idx is None:
                    continue

                rel_path_str = self.video_template.format(
                    video_key=video_key,
                    chunk_index=int(chunk_idx),
                    file_index=int(file_idx),
                )
                rel_path = Path(rel_path_str)
                start_ts = float(row.get(start_key, 0.0) or 0.0)
                end_ts = float(row.get(end_key, start_ts + duration_seconds if duration_seconds else 0.0) or start_ts)

                segments[video_key] = EpisodeVideoSegment(
                    video_key=video_key,
                    relative_path=rel_path,
                    from_timestamp=start_ts,
                    to_timestamp=end_ts,
                )

            records.append(
                EpisodeRecord(
                    episode_index=episode_index,
                    length=length,
                    dataset_from_index=dataset_from_index,
                    dataset_to_index=dataset_to_index,
                    tasks=tasks,
                    extras=extras,
                    video_segments=segments,
                )
            )
        return records

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "dataset_name": self.dataset_root.name,
            "fps": self.fps,
            "total_episodes": len(self.episodes),
            "video_keys": self.video_keys,
        }


INDEX_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <title>{{ dataset_name }} – LeRobot Dataset Viewer</title>
  <style>
    :root {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      color: #1f2933;
      background: #f7fafc;
    }
    body {
      margin: 0;
      display: flex;
      flex-direction: column;
      min-height: 100vh;
    }
    header {
      background: #111827;
      color: #f9fafb;
      padding: 1rem 2rem;
      box-shadow: 0 2px 6px rgba(15, 23, 42, 0.2);
    }
    header h1 {
      margin: 0;
      font-size: 1.4rem;
      font-weight: 600;
    }
    main {
      flex: 1;
      display: flex;
      padding: 1rem 2rem 2rem;
      gap: 1.5rem;
    }
    #episode-list {
      width: 280px;
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    #episode-list h2 {
      margin: 0;
      padding: 1.1rem 1.3rem;
      border-bottom: 1px solid #e2e8f0;
      font-size: 1rem;
      letter-spacing: 0.01em;
    }
    #episode-buttons {
      overflow-y: auto;
      flex: 1;
    }
    .episode-button {
      width: 100%;
      text-align: left;
      padding: 0.85rem 1.3rem;
      border: none;
      background: transparent;
      cursor: pointer;
      transition: background 0.2s ease;
      border-bottom: 1px solid #f1f5f9;
    }
    .episode-button:hover {
      background: #f8fafc;
    }
    .episode-button.active {
      background: #1d4ed8;
      color: #ffffff;
    }
    #viewer {
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 1rem;
    }
    #episode-meta {
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
      padding: 1.25rem 1.5rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.75rem;
    }
    .meta-item {
      display: flex;
      flex-direction: column;
      font-size: 0.95rem;
    }
    .meta-item span {
      color: #64748b;
      font-size: 0.78rem;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      margin-bottom: 0.25rem;
    }
    #videos {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 1.25rem;
    }
    .video-card {
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
      padding: 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.6rem;
    }
    .video-card h3 {
      margin: 0;
      font-size: 1rem;
      color: #1e3a8a;
    }
    video {
      width: 100%;
      border-radius: 10px;
      background: #000;
    }
    #empty-state {
      margin-top: 4rem;
      text-align: center;
      color: #64748b;
    }
  </style>
</head>
<body>
  <header>
    <h1>{{ dataset_name }} · Interactive Episode Viewer</h1>
  </header>
  <main>
    <section id=\"episode-list\">
      <h2>Episodes</h2>
      <div id=\"episode-buttons\"></div>
    </section>
    <section id=\"viewer\">
      <div id=\"episode-meta\"></div>
      <div id=\"videos\"></div>
      <div id=\"empty-state\" style=\"display:none;\">No episodes found.</div>
    </section>
  </main>
  <script>
    const state = {
      metadata: null,
      episodes: [],
      activeEpisode: null,
    };

    function renderEpisodeList() {
      const container = document.getElementById('episode-buttons');
      container.innerHTML = '';
      if (state.episodes.length === 0) {
        document.getElementById('empty-state').style.display = 'block';
        return;
      }
      document.getElementById('empty-state').style.display = 'none';

      state.episodes.forEach((episode, idx) => {
        const btn = document.createElement('button');
        btn.className = 'episode-button' + (state.activeEpisode === idx ? ' active' : '');
        const duration = episode.duration_seconds ? ` · ${episode.duration_seconds.toFixed(1)}s` : '';
        const label = episode.tasks && episode.tasks.length ? episode.tasks.join(', ') : 'Episode';
        btn.textContent = `#${episode.episode_index} · ${label}${duration}`;
        btn.addEventListener('click', () => selectEpisode(idx));
        container.appendChild(btn);
      });
    }

    function selectEpisode(idx) {
      state.activeEpisode = idx;
      renderEpisodeList();
      renderEpisode(idx);
    }

    function renderEpisode(idx) {
      const episode = state.episodes[idx];
      if (!episode) return;

      const meta = document.getElementById('episode-meta');
      meta.innerHTML = '';

      const metaItems = [
        { label: 'Episode', value: `#${episode.episode_index}` },
        { label: 'Frames', value: episode.length },
        { label: 'Dataset Indices', value: `${episode.dataset_from_index} – ${episode.dataset_to_index}` },
      ];

      if (episode.duration_seconds) {
        metaItems.push({ label: 'Duration', value: `${episode.duration_seconds.toFixed(2)} s` });
      }
      if (episode.tasks && episode.tasks.length) {
        metaItems.push({ label: 'Tasks', value: episode.tasks.join(', ') });
      }
      if (episode.success !== undefined) {
        metaItems.push({ label: 'Success', value: episode.success ? '✅' : '❌' });
      }
      if (episode.source !== undefined) {
        metaItems.push({ label: 'Source', value: episode.source });
      }
      metaItems.forEach(item => {
        const span = document.createElement('div');
        span.className = 'meta-item';
        span.innerHTML = `<span>${item.label}</span><strong>${item.value}</strong>`;
        meta.appendChild(span);
      });

      const videos = document.getElementById('videos');
      videos.innerHTML = '';
      const segments = episode.video_segments || {};
      const keys = Object.keys(segments);
      if (keys.length === 0) {
        const empty = document.createElement('div');
        empty.textContent = 'No video segments available for this episode.';
        videos.appendChild(empty);
        return;
      }

      keys.forEach(key => {
        const segment = segments[key];
        const card = document.createElement('div');
        card.className = 'video-card';
        const heading = document.createElement('h3');
        heading.textContent = key;
        const timing = document.createElement('div');
        timing.style.fontSize = '0.85rem';
        timing.style.color = '#475569';
        timing.textContent = `t = ${segment.from_timestamp.toFixed(2)}s → ${segment.to_timestamp.toFixed(2)}s`;
        const video = document.createElement('video');
        video.controls = true;
        video.src = segment.url;
        video.dataset.start = segment.from_timestamp;
        video.dataset.end = segment.to_timestamp;
        video.addEventListener('loadedmetadata', () => {
          const start = parseFloat(video.dataset.start);
          if (!Number.isNaN(start)) {
            video.currentTime = start;
          }
        });
        video.addEventListener('timeupdate', () => {
          const end = parseFloat(video.dataset.end);
          if (!Number.isNaN(end) && video.currentTime > end) {
            video.pause();
            video.currentTime = end;
          }
        });
        card.appendChild(heading);
        card.appendChild(timing);
        card.appendChild(video);
        videos.appendChild(card);
      });
    }

    async function bootstrap() {
      const metaResponse = await fetch('/api/metadata');
      if (!metaResponse.ok) {
        throw new Error('Failed to load dataset metadata');
      }
      state.metadata = await metaResponse.json();

      const episodesResponse = await fetch('/api/episodes');
      if (!episodesResponse.ok) {
        throw new Error('Failed to load episode list');
      }
      const payload = await episodesResponse.json();
      state.episodes = payload.episodes || [];

      renderEpisodeList();
      if (state.episodes.length) {
        selectEpisode(0);
      }
    }

    bootstrap().catch(err => {
      console.error(err);
      const container = document.getElementById('episode-meta');
      container.innerHTML = `<div style=\"color:#b91c1c;\">${err.message}</div>`;
    });
  </script>
</body>
</html>
"""


def create_app(dataset_index: DatasetIndex):
    Flask, abort, jsonify, render_template_string, send_file = _require_flask_components()
    app = Flask(__name__)

    @app.route("/")
    def home():
        return render_template_string(INDEX_HTML, dataset_name=dataset_index.dataset_root.name)

    @app.route("/api/metadata")
    def metadata():
        return jsonify(dataset_index.as_metadata())

    @app.route("/api/episodes")
    def episodes():
        return jsonify({"episodes": [episode.as_dict() for episode in dataset_index.episodes]})

    @app.route("/video/<path:relpath>")
    def serve_video(relpath: str):
        file_path = (dataset_index.dataset_root / relpath).resolve()
        root = dataset_index.dataset_root
        try:
            file_path.relative_to(root)
        except ValueError:
            abort(404)
        if not file_path.is_file():
            abort(404)
        return send_file(file_path, mimetype="video/mp4")

    return app


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Interactive viewer for LeRobot datasets (video episodes).")
    parser.add_argument(
        "--dataset",
        "-d",
        default="data/square-dagger-v1",
        help="Path to a local LeRobot dataset directory (default: data/square-dagger-v1)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for the web server (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5001, help="Port for the web server (default: 5001)")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open the viewer in the default browser after startup.",
    )

    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    dataset_index = DatasetIndex(dataset_path)
    app = create_app(dataset_index)

    if args.open_browser:
        def _open():
            time.sleep(1.0)
            url = f"http://{args.host}:{args.port}/"
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":  # pragma: no cover
    main()

import json
import os
import uuid
import random
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from flask import Flask, render_template, request, redirect, url_for, session, send_file, abort
from flask_sqlalchemy import SQLAlchemy


# -----------------------------
# Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")
DB_PATH = os.path.join(BASE_DIR, "results.sqlite3")

N_TRIALS_PER_PARTICIPANT = 10

METRICS = [
    {
        "key": "metric_a",
        "name": "Visual Quality and Realism",
        "desc": "Short description of metric A."
    },
    {
        "key": "metric_b",
        "name": "Temporal Consistency",
        "desc": "Short description of metric B."
    },
    {
        "key": "metric_c",
        "name": "Controllability",
        "desc": "Short description of metric C."
    },
    {
        "key": "metric_d",
        "name": "Overall Quality Score",
        "desc": "Short description of metric D."
    }
]


# -----------------------------
# App / DB setup
# -----------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
# For production: set this via environment variable and keep it secret
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# -----------------------------
# Database model
# -----------------------------
class Rating(db.Model):
    __tablename__ = "ratings"

    id = db.Column(db.Integer, primary_key=True)

    participant_id = db.Column(db.String(64), index=True, nullable=False)
    created_at_utc = db.Column(db.String(64), nullable=False)

    trial_index = db.Column(db.Integer, nullable=False)

    set_name = db.Column(db.String(128), nullable=False)

    # Which methods were compared, and which video file was shown on each side
    method_a = db.Column(db.String(128), nullable=False)
    method_b = db.Column(db.String(128), nullable=False)

    video_a = db.Column(db.String(512), nullable=False)  # relative path under static/
    video_b = db.Column(db.String(512), nullable=False)

    # To support counterbalancing / auditing
    left_label = db.Column(db.String(8), nullable=False)   # "A"
    right_label = db.Column(db.String(8), nullable=False)  # "B"

    # Three integer metrics 0-10
    metric_a_A = db.Column(db.Integer, nullable=False)
    metric_b_A = db.Column(db.Integer, nullable=False)
    metric_c_A = db.Column(db.Integer, nullable=False)
    metric_d_A = db.Column(db.Integer, nullable=False)

    metric_a_B = db.Column(db.Integer, nullable=False)
    metric_b_B = db.Column(db.Integer, nullable=False)
    metric_c_B = db.Column(db.Integer, nullable=False)
    metric_d_B = db.Column(db.Integer, nullable=False)

# -----------------------------
# Manifest / sampling utilities
# -----------------------------
def load_manifest() -> Dict[str, Dict[str, List[str]]]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict) or len(manifest) < 1:
        raise ValueError("manifest.json must be a dict of sets -> methods -> [video paths]")

    for set_name, methods in manifest.items():
        if not isinstance(methods, dict) or len(methods) < 2:
            raise ValueError(f"Set '{set_name}' must map to a dict with at least 2 methods")

        for method_name, vids in methods.items():
            if not isinstance(vids, list) or len(vids) == 0:
                raise ValueError(f"Set '{set_name}', method '{method_name}' must map to a non-empty list")

            for rel_path in vids:
                abs_path = os.path.join(BASE_DIR, "static", rel_path)
                if not os.path.isfile(abs_path):
                    raise FileNotFoundError(
                        f"Missing video file for set '{set_name}', method '{method_name}': {abs_path}"
                    )

    return manifest


def generate_trials(
    manifest: Dict[str, Dict[str, List[str]]],
    n_trials: int,
    seed: int,
    allow_video_repeats_within_participant: bool = False,
    counterbalance_sides: bool = True,
) -> List[dict]:
    rng = random.Random(seed)

    set_names = list(manifest.keys())
    if len(set_names) == 0:
        raise ValueError("Manifest has no sets")

    # Tracks used videos to reduce repeats within a participant
    used_videos = set()

    trials = []
    for t in range(n_trials):
        # 1) choose set for this trial
        set_name = rng.choice(set_names)

        methods_dict = manifest[set_name]
        method_names = list(methods_dict.keys())
        if len(method_names) < 2:
            raise ValueError(f"Set '{set_name}' has < 2 methods; cannot create A/B trial")

        # 2) choose two different methods from same set
        method_left, method_right = rng.sample(method_names, 2)

        # 3) choose one video from each method within the set
        vid_left = pick_video(
            rng,
            methods_dict[method_left],
            used_videos,
            allow_video_repeats_within_participant,
        )
        vid_right = pick_video(
            rng,
            methods_dict[method_right],
            used_videos,
            allow_video_repeats_within_participant,
        )

        # 4) counterbalance which method goes on left/right if desired
        # (Here "left" and "right" are UI sides; you still label them A/B in UI.)
        if counterbalance_sides and (rng.random() < 0.5):
            method_left, method_right = method_right, method_left
            vid_left, vid_right = vid_right, vid_left

        trials.append({
            "trial_index": t,
            "set": set_name,
            "left": {"label": "A", "method": method_left, "video": vid_left},
            "right": {"label": "B", "method": method_right, "video": vid_right},
        })

    return trials


def pick_video(
    rng: random.Random,
    candidates: List[str],
    used_videos: set,
    allow_repeats: bool
) -> str:
    if allow_repeats:
        return rng.choice(candidates)

    # Try a few times to find an unused candidate.
    # If exhausted, fall back to allowing repeats for this pick.
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    for v in shuffled:
        if v not in used_videos:
            used_videos.add(v)
            return v

    # All used; fallback
    v = rng.choice(candidates)
    used_videos.add(v)
    return v


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


# -----------------------------
# Routes
# -----------------------------
@app.before_request
def ensure_participant():
    # Assign a participant_id once and keep it in session cookie.
    if "participant_id" not in session:
        session["participant_id"] = uuid.uuid4().hex


@app.route("/", methods=["GET"])
def start():
    return render_template("start.html", n_trials=N_TRIALS_PER_PARTICIPANT, metrics=METRICS)


@app.route("/begin", methods=["POST"])
def begin():
    manifest = load_manifest()

    # You can also accept a "participant_code" from the user here if you prefer
    participant_id = session["participant_id"]

    # Deterministic per participant, so refreshes won’t reshuffle mid-study.
    seed = int(participant_id[:8], 16)

    trials = generate_trials(
        manifest=manifest,
        n_trials=N_TRIALS_PER_PARTICIPANT,
        seed=seed,
        allow_video_repeats_within_participant=False,
        counterbalance_sides=True,
    )

    session["trials"] = trials
    session["current_trial"] = 0
    return redirect(url_for("trial"))


@app.route("/trial", methods=["GET"])
def trial():
    trials = session.get("trials")
    if not trials:
        return redirect(url_for("start"))

    idx = int(session.get("current_trial", 0))
    if idx >= len(trials):
        return redirect(url_for("done"))

    t = trials[idx]

    app.logger.info(
        "Serving trial %d: set=%s | left=%s | right=%s",
        idx,
        t.get("set"),
        t["left"]["video"],
        t["right"]["video"],
    )

    return render_template(
        "trial.html",
        trial_index=idx,
        n_trials=len(trials),
        left=t["left"],
        right=t["right"],
        metrics=METRICS,
    )


@app.route("/submit", methods=["POST"])
def submit():
    trials = session.get("trials")
    if not trials:
        return redirect(url_for("start"))

    idx = int(session.get("current_trial", 0))
    if idx >= len(trials):
        return redirect(url_for("done"))

    t = trials[idx]
    participant_id = session["participant_id"]

    # Parse metric integers 0..10
    scores_A = {}
    scores_B = {}

    for m in METRICS:
        key = m["key"]

        raw_A = request.form.get(f"{key}_A")
        raw_B = request.form.get(f"{key}_B")

        if raw_A is None or raw_B is None:
            abort(400, f"Missing score for {key}")

        val_A = int(raw_A)
        val_B = int(raw_B)

        if not (0 <= val_A <= 10 and 0 <= val_B <= 10):
            abort(400, "Scores must be between 0 and 10")

        scores_A[key] = val_A
        scores_B[key] = val_B

    # Store one row per trial
    row = Rating(
        participant_id=participant_id,
        created_at_utc=utc_now_str(),
        trial_index=idx,
        set_name=t["set"],

        method_a=t["left"]["method"],
        method_b=t["right"]["method"],
        video_a=t["left"]["video"],
        video_b=t["right"]["video"],

        left_label=t["left"]["label"],
        right_label=t["right"]["label"],

        metric_a_A=scores_A["metric_a"],
        metric_b_A=scores_A["metric_b"],
        metric_c_A=scores_A["metric_c"],
        metric_d_A=scores_A["metric_d"],

        metric_a_B=scores_B["metric_a"],
        metric_b_B=scores_B["metric_b"],
        metric_c_B=scores_B["metric_c"],
        metric_d_B=scores_B["metric_d"],
    )
    db.session.add(row)
    db.session.commit()

    session["current_trial"] = idx + 1
    if session["current_trial"] >= len(trials):
        return redirect(url_for("done"))
    return redirect(url_for("trial"))


@app.route("/done", methods=["GET"])
def done():
    return render_template("done.html")


@app.route("/export.csv", methods=["GET"])
def export_csv():
    import csv
    from io import StringIO, BytesIO

    sio = StringIO()
    writer = csv.writer(sio)

    writer.writerow([
        "participant_id", "created_at_utc", "trial_index", "set_name",
        "method_left", "video_left", "method_right", "video_right",
        "metric_a_left", "metric_b_left", "metric_c_left", "metric_d_left",
        "metric_a_right", "metric_b_right", "metric_c_right", "metric_d_right",
    ])

    rows = Rating.query.order_by(Rating.participant_id, Rating.trial_index).all()
    for r in rows:
        writer.writerow([
            r.participant_id, r.created_at_utc, r.trial_index, r.set_name,
            r.method_a, r.video_a, r.method_b, r.video_b,
            r.metric_a_A, r.metric_b_A, r.metric_c_A, r.metric_d_A,
            r.metric_a_B, r.metric_b_B, r.metric_c_B, r.metric_d_B,
        ])

    # Convert to bytes for send_file
    bio = BytesIO(sio.getvalue().encode("utf-8"))
    bio.seek(0)

    return send_file(
        bio,
        mimetype="text/csv; charset=utf-8",
        as_attachment=True,
        download_name="results.csv",
    )



def init_db():
    with app.app_context():
        db.create_all()


if __name__ == "__main__":
    init_db()
    # In production, run behind gunicorn/uvicorn + reverse proxy
    app.run(host="0.0.0.0", port=5000, debug=True)

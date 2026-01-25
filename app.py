import json
import os
import uuid
import random
from datetime import datetime, timezone
from typing import Dict, List, Tuple

from flask import Flask, render_template, request, redirect, url_for, session, send_file, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError


# -----------------------------
# Configuration
# -----------------------------
# local deployment
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))
# Where to store persistent files (SQLite) in production.
PERSISTENT_STORAGE_DIR = os.environ.get("PERSISTENT_STORAGE_DIR", LOCAL_DIR)   # local dev defaults to repo dir
STATIC_DIR = os.path.join(PERSISTENT_STORAGE_DIR, "static") # where static files are stored in production
DB_PATH = os.path.join(PERSISTENT_STORAGE_DIR, "results.sqlite3") # SQLite DB path
MANIFEST_PATH = os.path.join(STATIC_DIR, "manifest.json")
EXPORT_TOKEN = os.environ.get("EXPORT_TOKEN", "0")  # set EXPORT_TOKEN env variable to a secret value in production
CONTACT_INFO= os.environ.get("CONTACT_INFO", "CONTACT")
N_TRIALS_PER_PARTICIPANT = int(os.environ.get("N_TRIALS_PER_PARTICIPANT", 10))

# Demographics
DEMOGRAPHICS = [
    {
        "key": "ai_exp",
        "question": "Do you work or study in AI or machine learning?",
        "options": [
            "Yes",
            "No",
        ],
    },
    {
        "key": "game_exp",
        "question": "How familiar are you with video games?",
        "options": [
            "Not at all",
            "Casual player",
            "Regular player",
        ],
    },
    {
        "key": "minecraft_exp",
        "question": "How much experience do you have with Minecraft?",
        "options": [
            "I've never heard of Minecraft",
            "I know what Minecraft is but have never played it",
            "Less than 10 hours",
            "10–100 hours",
            "More than 100 hours",
        ],
    },
    {
        "key": "minetest_exp",
        "question": "What is your experience with Luanti (formerly known as Minetest)?",
        "options": [
            "I have played it",
            "I have heard of it but never played it",
            "I have never heard of it",
        ],
    },
]

METRICS = [
    {
        "key": "metric_a",
        "name": "Visual Quality and 3D consistency",
        "desc": "Visual quality and spatial consistency of the 3D scene. Deduct points for visual artifacts such as blurriness, color shifts, texture inconsistencies, or objects whose appearance changes unnaturally when viewed from different viewpoints."
    },
    {
        "key": "metric_b",
        "name": "Temporal Consistency",
        "desc": "Stability and consistency of the scene over time. Deduct points if parts of the scene do not remain consistent as the video progresses, particularly when the player revisits areas that were shown earlier."
    },
    {
        "key": "metric_c",
        "name": "Motion and Interaction Plausibility",
        "desc": "Plausibility of player motion and interactions with the environment. Deduct points for unrealistic movement (e.g. clipping through solid objects, floating, or missing collisions). Also deduct points if the environment’s dynamics appear to degrade or become unnaturally simplified (e.g. a transition from a complex scene towards overly flat or uniform terrain)."
    },
    {
        "key": "metric_d",
        "name": "Overall Quality Score",
        "desc": "Your subjective rating of the overall quality of the video."
    }
]


# -----------------------------
# App / DB setup
# -----------------------------
app = Flask(__name__, static_folder=STATIC_DIR, template_folder="templates")
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

    __table_args__ = (
        db.UniqueConstraint("participant_id", "trial_index", name="uq_participant_trial"),
    )

    id = db.Column(db.Integer, primary_key=True)

    participant_id = db.Column(db.String(64), index=True, nullable=False)
    created_at_utc = db.Column(db.String(64), nullable=False)

    trial_index = db.Column(db.Integer, nullable=False)

    set_name = db.Column(db.String(128), nullable=False)

    # Which methods were compared, and which video file was shown on each side
    method_a = db.Column(db.String(128), nullable=False)
    method_b = db.Column(db.String(128), nullable=False)

    video_a = db.Column(db.String(512), nullable=False)
    video_b = db.Column(db.String(512), nullable=False)

    # To support counterbalancing / auditing
    left_label = db.Column(db.String(8), nullable=False)   # "A"
    right_label = db.Column(db.String(8), nullable=False)  # "B"

    # Three integer metrics 0-5
    metric_a_A = db.Column(db.Integer, nullable=False)
    metric_b_A = db.Column(db.Integer, nullable=False)
    metric_c_A = db.Column(db.Integer, nullable=False)
    metric_d_A = db.Column(db.Integer, nullable=False)

    metric_a_B = db.Column(db.Integer, nullable=False)
    metric_b_B = db.Column(db.Integer, nullable=False)
    metric_c_B = db.Column(db.Integer, nullable=False)
    metric_d_B = db.Column(db.Integer, nullable=False)

# -----------------------------
# Demographics model
# -----------------------------
class DemographicResponse(db.Model):
    __tablename__ = "demographics"

    id = db.Column(db.Integer, primary_key=True)

    participant_id = db.Column(db.String(64), unique=True, nullable=False)
    created_at_utc = db.Column(db.String(64), nullable=False)

    # JSON blob of answers
    responses_json = db.Column(db.Text, nullable=False)


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
                abs_path = os.path.join(STATIC_DIR, rel_path)
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


def participant_seed(participant_id: str) -> int:
    return int(participant_id[:8], 16)


def participant_progress(participant_id: str) -> int:
    # Number of completed trials
    return Rating.query.filter_by(participant_id=participant_id).count()


def has_demographics(participant_id: str) -> bool:
    return DemographicResponse.query.filter_by(participant_id=participant_id).first() is not None


def is_completed(participant_id: str) -> bool:
    return participant_progress(participant_id) >= N_TRIALS_PER_PARTICIPANT


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_export_token():
    if not EXPORT_TOKEN:
        abort(500, "EXPORT_TOKEN is not set on server")
    # token can be passed as a query param or header
    token = request.args.get("token") or request.headers.get("X-Export-Token")
    if token != EXPORT_TOKEN:
        abort(403)


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
    return render_template("start.html", n_trials=N_TRIALS_PER_PARTICIPANT, metrics=METRICS, contact_info=CONTACT_INFO)

@app.route("/begin", methods=["POST"])
def begin():
    participant_id = session["participant_id"]

    if is_completed(participant_id):
        return redirect(url_for("done"))

    session["seed"] = participant_seed(participant_id)
    # If demographics already filled, do not ask again
    if has_demographics(participant_id):
        return redirect(url_for("trial"))

    return redirect(url_for("demographics"))

@app.route("/demographics", methods=["GET", "POST"])
def demographics():
    participant_id = session["participant_id"]

    if is_completed(participant_id):
        return redirect(url_for("done"))

    # If already answered, skip
    if request.method == "GET" and has_demographics(participant_id):
        return redirect(url_for("trial"))

    if request.method == "POST":
        responses = {}

        for q in DEMOGRAPHICS:
            key = q["key"]
            val = request.form.get(key)
            if val is not None:
                responses[key] = val

        row = DemographicResponse(
            participant_id=participant_id,
            created_at_utc=utc_now_str(),
            responses_json=json.dumps(responses),
        )

        # one row per participant
        db.session.merge(row)
        db.session.commit()

        return redirect(url_for("trial"))

    return render_template(
        "demographics.html",
        demographics=DEMOGRAPHICS,
    )


@app.route("/trial", methods=["GET"])
def trial():
    participant_id = session.get("participant_id")
    if not participant_id:
        return redirect(url_for("start"))

    if is_completed(participant_id):
        return redirect(url_for("done"))

    manifest = load_manifest()

    seed = session.get("seed", participant_seed(participant_id))
    trials = generate_trials(
        manifest=manifest,
        n_trials=N_TRIALS_PER_PARTICIPANT,
        seed=seed,
        allow_video_repeats_within_participant=False,
        counterbalance_sides=True,
    )

    idx = participant_progress(participant_id)
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
        n_trials=N_TRIALS_PER_PARTICIPANT,
        left=t["left"],
        right=t["right"],
        metrics=METRICS,
    )

@app.route("/submit", methods=["POST"])
def submit():
    participant_id = session.get("participant_id")
    if not participant_id:
        return redirect(url_for("start"))

    if is_completed(participant_id):
        return redirect(url_for("done"))

    manifest = load_manifest()
    seed = session.get("seed", participant_seed(participant_id))
    trials = generate_trials(
        manifest=manifest,
        n_trials=N_TRIALS_PER_PARTICIPANT,
        seed=seed,
        allow_video_repeats_within_participant=False,
        counterbalance_sides=True,
    )

    idx = participant_progress(participant_id)
    t = trials[idx]

    # Parse metric integers 0..5
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
        if not (0 <= val_A <= 5 and 0 <= val_B <= 5):
            abort(400, "Scores must be between 0 and 5")
        scores_A[key] = val_A
        scores_B[key] = val_B

    # Prevent accidental duplicates (e.g., back button / double submit)
    existing = Rating.query.filter_by(participant_id=participant_id, trial_index=idx).first()
    if existing is not None:
        return redirect(url_for("trial"))

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

    try:
        db.session.add(row)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()

    return redirect(url_for("trial"))


@app.route("/done", methods=["GET"])
def done():
    return render_template("done.html")


# local only: reset participant_id to allow retaking the study
@app.route("/reset", methods=["GET"])
def reset():
    # Safety: only allow in local dev
    if not (request.host.startswith("127.0.0.1") or request.host.startswith("localhost")):
        abort(404)

    session.clear()  # clears participant_id, seed, etc.
    return redirect(url_for("start"))


@app.route("/export.csv", methods=["GET"])
def export_csv():
    import csv
    import json
    from io import StringIO, BytesIO

    require_export_token()

    # Demographic columns are defined by the global DEMOGRAPHICS list
    demo_keys = [q["key"] for q in DEMOGRAPHICS]

    # Load demographics once, build a participant_id -> dict map
    demo_rows = DemographicResponse.query.all()
    demo_by_pid = {}
    for d in demo_rows:
        try:
            demo_by_pid[d.participant_id] = json.loads(d.responses_json or "{}")
        except json.JSONDecodeError:
            demo_by_pid[d.participant_id] = {}

    sio = StringIO()
    writer = csv.writer(sio)

    writer.writerow([
        "participant_id", "created_at_utc", "trial_index", "set_name",
        "method_left", "video_left", "method_right", "video_right",
        "metric_a_left", "metric_b_left", "metric_c_left", "metric_d_left",
        "metric_a_right", "metric_b_right", "metric_c_right", "metric_d_right",
        *demo_keys,  # one column per demographic key
    ])

    rows = Rating.query.order_by(Rating.participant_id, Rating.trial_index).all()
    for r in rows:
        demo = demo_by_pid.get(r.participant_id, {})
        writer.writerow([
            r.participant_id, r.created_at_utc, r.trial_index, r.set_name,
            r.method_a, r.video_a, r.method_b, r.video_b,
            r.metric_a_A, r.metric_b_A, r.metric_c_A, r.metric_d_A,
            r.metric_a_B, r.metric_b_B, r.metric_c_B, r.metric_d_B,
            *[demo.get(k, "") for k in demo_keys],
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

init_db()

if __name__ == "__main__":
    init_db()
    # In production, run behind gunicorn/uvicorn + reverse proxy
    app.run(host="0.0.0.0", port=5000, debug=True)

# video_ab_study
Minimal webapp to run a user study on A/B random pairing of videos.

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Add video files to `static/videos/` and specify directory structure in `manifest.json`.

## Running the App locally
1. Run the app: `python app.py`
2. Access the app at `http://localhost:5000` in your web browser and fill in the survey.
3. After the study, generate `result.csv` by accessing `http://localhost:5000/export.csv`. It will be downloaded to your local machine.
4. Open `result.csv` to see the collected data or run `result_processing.py` to start processing it.
5. To reset the study for a new participant, go to `http://localhost:5000/reset`.

## Configuration
- `manifest.json`: Define the video pairings and methods here.
- `templates/`: HTML templates for the web pages.
- `app.py`: Main Flask application file. You can modify the `METRICS` and `N_TRIALS_PER_PARTICIPANT` variables to customize the study.

## Deployment Instructions

This application is designed to run as a standard Python web service with persistent storage for results.

### Requirements

* Python 3.10+
* A writable directory for persistent storage (used for the SQLite database)
* Ability to set environment variables
* A production WSGI server (e.g. Gunicorn)

---

### Required environment variables

The following environment variables must be set when running the application in production:

| Variable                 | Description                                                                     |
| ------------------------ | ------------------------------------------------------------------------------- |
| `SECRET_KEY`             | Secret key used to sign Flask session cookies. Must be a stable, private value. |
| `PERSISTENT_STORAGE_DIR` | Path to a writable directory where `results.sqlite3` will be stored.            |
| `EXPORT_TOKEN`           | Secret token required to access the `/export.csv` endpoint.                     |

Example:

```bash
export SECRET_KEY="long-random-string"
export PERSISTENT_STORAGE_DIR="/path/to/persistent/storage"
export EXPORT_TOKEN="another-long-random-string"
```

---

### Database

The application uses SQLite for storing study results.

* The database file is created automatically at:

  ```
  $PERSISTENT_STORAGE_DIR/results.sqlite3
  ```

* Tables are created on startup if they do not already exist.

* The database directory must exist and be writable by the application process.

Only the database is stored in persistent storage; static assets (videos, templates, code) are read from the application directory.

---

### Running the application

For production, the application should be run behind a WSGI server.

Example:

```bash
gunicorn app:app --bind 0.0.0.0:8000 --workers 1 --threads 4
```

Notes:

* A single worker is recommended when using SQLite to avoid write-locking issues.
* Multiple threads are safe and supported.

---

### Exporting results

Results can be downloaded as CSV via:

```
/export.csv
```

Access is protected by the `EXPORT_TOKEN`.

Example:

```bash
curl -H "X-Export-Token: <EXPORT_TOKEN>" https://<host>/export.csv -o results.csv
```

---

### Development vs production

* In local development, environment variables may be omitted.
* In production, all required variables **must** be set.

---


## Known behavior in corner cases (we're OK with these)

### Local deployment
- Participant quits before completing all trials: Their data is saved up to the point they quit. Downloading `export.csv` will include their partial data. They can resume the study by reloading `http://localhost:5000`

### Remote deployment
- Participant can run multiple sessions by opening multiple browsers, clearing cookies or using private/incognito mode. Each session is treated as a separate participant.

## TODOs
- [x] Update structure of manifest so that videos are stored in `setA/method1/..., setB/method1/..., setC/method1/...`. For each individual A/B trial both video A and video B should come from the same set but different methods.
- [x] Add remote deployment instructions and test.
- [x] Protect /export.csv endpoint prior to remote deployment.
- [x] When running locally, add option to rerun study with a new participant_id. how: go to .../reset to reset the session.
- [x] Add study info/description to start.html. Add consent checkbox.
- [ ] Add preliminary.html to explain what we mean by each metric, give a good/bad example for each. Give an example of both games.
- [x] Add demographics.html to ask some demographic questions about the user: Experience with video quality assessment, Prior experience playing Minecraft-like games.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

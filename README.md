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

## Configuration
- `manifest.json`: Define the video pairings and methods here.
- `templates/`: HTML templates for the web pages.
- `app.py`: Main Flask application file. You can modify the `METRICS` and `N_TRIALS_PER_PARTICIPANT` variables to customize the study.

## Deployment
TODO: deployment instructions.

## Known behavior in corner cases:

### Local deployment
- Participant quits before completing all trials: Their data is saved up to the point they quit. Downloading `export.csv` will include their partial data. They can resume the study by reloading `http://localhost:5000`
- Participant completes all trials: Their data is saved and included in `export.csv`. Reloading `http://localhost:5000` redirects them to `http://localhost:5000/done`. However killing and restarting the app lets them retake the study. In that case, new entries will be created in the database, with the same participant_id. Entries remain unique as the created_at_utc timestamp will be different.

## TODOs
- [ ] Add remote deployment instructions and test.
- [ ] Protect /export.csv endpoint prior to remote deployment.
- [ ] When running locally, add option to rerun study with a new participant_id to done.html.
- [ ] Add study info/description to start.html. Add consent checkbox.
- [ ] Add preliminary.html to explain what we mean by each metric, give a good/bad example for each. Give an example of both games.
- [ ] Add user_questions.html to ask some demographic questions about the user: Experience with video quality assessment, Prior experience playing Minecraft-like games.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

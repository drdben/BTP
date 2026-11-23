# Watching Together — Media Comfort Survey

A small survey for exploring how comfortable people feel watching entertainment clips alone or with different social groups. It collects each participant's name, so responses are identifiable to the researcher.

## Prepare the survey

1. Review the 11 clips in `clips.json` before inviting participants. Add or remove clips as needed and add content notes after previewing them.
2. Use a stable, unique `id` for each clip. Add its display `title`, URL, and an optional short `note` describing sensitive content.
3. Preview every clip and its platform embed before sharing the survey. YouTube links are embedded using YouTube's privacy-enhanced player. Reddit post links are embedded where Reddit allows it; other URLs open in a new tab.
4. Make sure your consent wording matches your institution's research requirements. The included short notice is a draft for a small, voluntary survey and does not replace any required ethics review or approved consent text.

Example `clips.json`:

```json
{
  "researchTitle": "Watching Together",
  "estimatedMinutes": 7,
  "clips": [
    {
      "id": "clip-01",
      "title": "Clip 1",
      "url": "https://www.youtube.com/shorts/VIDEO_ID",
      "note": "Contains strong language."
    },
    {
      "id": "clip-02",
      "title": "Clip 2",
      "url": "https://www.reddit.com/r/SUBREDDIT/comments/POST_ID/",
      "note": ""
    }
  ]
}
```

Each clip gets four 1–5 comfort ratings: on one's own, with parents, with children under 12 (or hypothetically), and with friends. Participants may skip individual ratings or an entire clip.

## Run locally

For the current survey, you can open `index.html` directly; the provided clip list is built in, so it also works when the browser blocks local `clips.json` access. If you edit `clips.json`, use the local server below so those edits load.

From the project folder:

```bash
../.venv/bin/python -m http.server 8000
```

Open `http://localhost:8000` in a browser. The local HTTP server lets the page load `clips.json`.

## Connect a Google Sheet

1. Create a Google Sheet for responses. Copy its spreadsheet ID from the URL: the value between `/d/` and `/edit`.
2. In the sheet, choose **Extensions → Apps Script**. Replace the editor contents with `Code.gs` from this repository, then replace `PASTE_GOOGLE_SHEET_ID_HERE` with the ID from step 1. Save.
3. Choose **Deploy → New deployment → Web app**. Set **Execute as** to your account and **Who has access** to **Anyone**, then deploy. Google may ask you to authorize the script. The web app needs public access so survey participants can submit without signing in; the script writes to the spreadsheet using your account.
4. Copy the deployed web app URL ending in `/exec`. In `app.js`, replace `PASTE_APPS_SCRIPT_WEB_APP_URL_HERE` in `GOOGLE_SHEETS_ENDPOINT` with that URL. Commit and push the change so Netlify deploys the updated site.
5. Open the Netlify site in a private/incognito window and submit a test response. Wait for the screen to say **Response received**, then check the `Responses` tab in the sheet. Each row has a submission ID, timestamp, participant name, and the complete response as JSON. Delete the test row before inviting participants.

If you change `Code.gs`, deploy a new version through **Deploy → Manage deployments → Edit → New version → Deploy**. The endpoint uses a unique submission ID so retrying the same response does not add a duplicate row. Participants can download a JSON backup at the submission screen; the existing researcher import tool can combine those backup files into a CSV.

The sheet contains names and survey responses. Limit access to the spreadsheet to the research team and follow your institution's approved consent and retention requirements. The web app URL is not a secret; do not put credentials in `app.js`.

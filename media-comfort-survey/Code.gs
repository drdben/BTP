// Google Apps Script web app for the Watching Together survey.
// Set SPREADSHEET_ID before deploying. The script creates the Responses tab.
const SPREADSHEET_ID = "PASTE_GOOGLE_SHEET_ID_HERE";
const SHEET_NAME = "Responses";
const HEADERS = ["Submission ID", "Submitted at (UTC)", "Participant name", "Response JSON"];

function doPost(e) {
  try {
    const response = JSON.parse(e.postData.contents);
    if (response.survey !== "Watching Together" || !response.submissionId || !Array.isArray(response.ratings)) {
      throw new Error("Invalid survey response");
    }
    const lock = LockService.getScriptLock();
    lock.waitLock(10000);
    try {
      const sheet = getResponseSheet_();
      const ids = sheet.getLastRow() > 1
        ? sheet.getRange(2, 1, sheet.getLastRow() - 1, 1).getDisplayValues().flat()
        : [];
      if (!ids.includes(response.submissionId)) {
        sheet.appendRow([
          response.submissionId,
          response.submittedAt || new Date().toISOString(),
          safeCell_(response.demographics && response.demographics.participantName || ""),
          JSON.stringify(response)
        ]);
      }
    } finally {
      lock.releaseLock();
    }
    return jsonOutput_({ ok: true, submissionId: response.submissionId });
  } catch (error) {
    return jsonOutput_({ ok: false, error: String(error) });
  }
}

// JSONP status check lets the browser confirm receipt despite Apps Script's
// cross-origin redirect behavior. It returns only whether this random ID exists.
function doGet(e) {
  const callback = String(e.parameter.callback || "");
  const id = String(e.parameter.check || "");
  if (!/^sheetStatus_[a-zA-Z0-9]+$/.test(callback) || !id) {
    return ContentService.createTextOutput("Invalid request").setMimeType(ContentService.MimeType.TEXT);
  }
  const sheet = getResponseSheet_();
  const ids = sheet.getLastRow() > 1
    ? sheet.getRange(2, 1, sheet.getLastRow() - 1, 1).getDisplayValues().flat()
    : [];
  const result = { received: ids.includes(id) };
  return ContentService.createTextOutput(callback + "(" + JSON.stringify(result) + ");")
    .setMimeType(ContentService.MimeType.JAVASCRIPT);
}

function getResponseSheet_() {
  if (SPREADSHEET_ID.indexOf("PASTE_") === 0) throw new Error("Set SPREADSHEET_ID in Code.gs");
  const spreadsheet = SpreadsheetApp.openById(SPREADSHEET_ID);
  let sheet = spreadsheet.getSheetByName(SHEET_NAME);
  if (!sheet) sheet = spreadsheet.insertSheet(SHEET_NAME);
  if (sheet.getLastRow() === 0) {
    sheet.appendRow(HEADERS);
    sheet.setFrozenRows(1);
  }
  return sheet;
}

function jsonOutput_(value) {
  return ContentService.createTextOutput(JSON.stringify(value))
    .setMimeType(ContentService.MimeType.JSON);
}

function safeCell_(value) {
  const text = String(value);
  return /^[=+@\-]/.test(text) ? "'" + text : text;
}

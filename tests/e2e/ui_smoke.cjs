// End-to-end UI smoke test (optional; needs Node + Playwright: npm i -g playwright && npx playwright install chromium).
// Scenario 1 (BASE_EMPTY): a running app WITHOUT trained checkpoints.
// Scenario 2 (BASE_TRAINED): a running app WITH trained checkpoints. In the Stage-3 audit these were
//   SYNTHETIC checkpoints trained for seconds on noise images - a software test only, results meaningless.
// Env: BASE_EMPTY, BASE_TRAINED, SAMPLES_DIR (sample_xray.png >=256 px, notes.txt, one RSNA <patientId>.dcm), SHOTS_DIR
const { chromium, devices } = require("playwright");
const fs = require("fs");
const S = process.env.SAMPLES_DIR || __dirname;
const out = process.env.SHOTS_DIR || `${S}/shots`;
fs.mkdirSync(out, { recursive: true });
const routes = ["", "analysis", "explainability", "models", "experiments", "datasets", "about"];
const vis = async (loc) => { try { await loc.first().waitFor({ state: "visible", timeout: 15000 }); return true; } catch { return false; } };
const results = [];
const ok = (name, cond, extra = "") => { results.push(`${cond ? "PASS" : "FAIL"} ${name} ${extra}`); };

async function visitAll(browser, base, tag, ctxOpts) {
  const ctx = await browser.newContext(ctxOpts);
  const page = await ctx.newPage();
  const errors = [];
  page.on("console", (m) => m.type() === "error" && !/TUNNEL|fonts\.g|status of 415/.test(m.text() + (m.location()?.url || "")) && errors.push(m.text()));
  page.on("pageerror", (e) => errors.push(String(e)));
  for (const r of routes) {
    await page.goto(`${base}/#/${r}`);
    await page.waitForLoadState("networkidle");
    await page.waitForTimeout(600);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    ok(`${tag} /${r} no horizontal overflow`, overflow <= 1, `(${overflow}px)`);
    await page.screenshot({ path: `${out}/${tag}_${r || "overview"}.png`, fullPage: true });
  }
  ok(`${tag} no console errors`, errors.length === 0, errors.slice(0, 3).join(" | "));
  await ctx.close();
}

(async () => {
  const browser = await chromium.launch();
  const desktop = { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 };
  const phone = { ...devices["iPhone 13"] };

  // ---------------- Scenario 1: no trained models
  const A = process.env.BASE_EMPTY || "http://localhost:4173";
  await visitAll(browser, A, "s1_desktop", desktop);
  await visitAll(browser, A, "s1_phone", phone);
  {
    const ctx = await browser.newContext(desktop);
    const p = await ctx.newPage();
    await p.goto(`${A}/#/`); await p.waitForLoadState("networkidle");
    ok("s1 overview says models not trained", await vis(p.getByText("Models not trained yet.").first()));
    ok("s1 overview no experiment results", await vis(p.getByText("No experiment results available").first()));
    await p.goto(`${A}/#/analysis`); await p.waitForLoadState("networkidle");
    ok("s1 analysis shows training instructions", await vis(p.getByText("Models not trained yet - analysis is disabled")));
    await p.setInputFiles('input[type="file"]', `${S}/notes.txt`);
    ok("s1 invalid file rejected client-side", await vis(p.getByText(/Unsupported file type/)));
    await p.setInputFiles('input[type="file"]', `${S}/sample_xray.png`);
    await p.waitForTimeout(300);
    ok("s1 preview shows dimensions", await vis(p.getByText("512 × 512 px")));
    ok("s1 analyse button disabled", await p.getByRole("button", { name: "Analyse X-ray" }).isDisabled());
    await p.goto(`${A}/#/models`); await p.waitForLoadState("networkidle");
    await vis(p.getByText("Not trained", { exact: true }));
    ok("s1 six+ not-trained badges", (await p.getByText("Not trained", { exact: true }).count()) >= 6);
    ok("s1 use-for-analysis disabled", await p.getByRole("button", { name: "Use for analysis" }).first().isDisabled());
    await ctx.close();
  }
  // API down handling: frontend pointed at a dead backend
  {
    const ctx = await browser.newContext(desktop);
    const p = await ctx.newPage();
    await p.route("**/api/**", (route) => route.abort());
    await p.goto(`${A}/#/datasets`); await p.waitForTimeout(1500);
    ok("api-down error state shown", await vis(p.getByText(/Cannot reach the analysis server/).first()));
    await p.screenshot({ path: `${out}/api_down.png` });
    await ctx.close();
  }

  // ---------------- Scenario 2: synthetic trained checkpoints
  const B = process.env.BASE_TRAINED || "http://localhost:4174";
  await visitAll(browser, B, "s2_desktop", desktop);
  {
    const ctx = await browser.newContext(desktop);
    const p = await ctx.newPage();
    const errors = [];
    p.on("pageerror", (e) => errors.push(String(e)));
    p.on("console", (m) => m.type() === "error" && !/TUNNEL|fonts\.g|status of 415/.test(m.text() + (m.location()?.url || "")) && errors.push(m.text()));
    // model selection through Model Lab
    await p.goto(`${B}/#/models`); await p.waitForLoadState("networkidle");
    const cardB = p.locator("section", { hasText: "Hybrid Model B" }).filter({ has: p.getByRole("button", { name: /analysis/ }) }).first();
    await cardB.getByRole("button", { name: /Use for analysis|Selected for analysis/ }).click();
    await p.waitForTimeout(1500);
    ok("s2 model B selected", await vis(p.getByRole("button", { name: "Selected for analysis" }).first()));
    ok("s2 A vs B comparison table", await vis(p.getByText("DeLong paired test")));
    await p.screenshot({ path: `${out}/s2_models_after_select.png`, fullPage: true });

    // analysis
    await p.goto(`${B}/#/analysis`); await p.waitForLoadState("networkidle");
    await p.setInputFiles('input[type="file"]', `${S}/sample_xray.png`);
    // server-side validation: a text file renamed to .png passes the browser check but must be rejected by the API
    fs.writeFileSync(`${out}/fake.png`, "this is not an image");
    await p.getByRole("button", { name: "Remove file" }).click();
    await p.setInputFiles('input[type="file"]', `${out}/fake.png`);
    await p.getByRole("button", { name: "Analyse X-ray" }).click();
    await p.getByText(/does not match a PNG, JPEG or DICOM/).waitFor({ timeout: 20000 });
    ok("s2 server-side invalid file error shown", true);
    await p.getByRole("button", { name: "Remove file" }).click();
    await p.setInputFiles('input[type="file"]', `${S}/sample_xray.png`);
    // model selection from the top bar
    await p.getByLabel("Model used for analysis").selectOption("model_b");
    await p.getByRole("button", { name: "Analyse X-ray" }).click();
    ok("s2 loading state shown", await vis(p.getByText(/Analysing…|Uploading|Running model/).first()));
    await p.getByText("Model output", { exact: true }).waitFor({ timeout: 120000 });
    ok("s2 no configuration warning (calibration matches checkpoint)", !(await p.getByText("Configuration warning").isVisible()));
    ok("s2 analysis result rendered", await vis(p.getByText(/Model output only - not a diagnosis/)));
    ok("s2 calibrated probability shown", await vis(p.getByText("Calibrated probability")));
    for (const t of ["Localization heatmap", "Grad-CAM++", "Combined overlay"]) {
      await p.getByRole("tab", { name: t }).click();
      await p.waitForTimeout(200);
    }
    ok("s2 combined overlay has 2 layers", (await p.locator('img[alt$="overlay"]').count()) === 2);
    await p.getByRole("button", { name: "Zoom in" }).click();
    await p.getByRole("button", { name: /Compare with original/ }).click();
    await p.screenshot({ path: `${out}/s2_analysis_result.png`, fullPage: true });
    // report
    const [rep] = await Promise.all([ctx.waitForEvent("page"), p.getByRole("button", { name: "Generate report" }).click()]);
    await rep.waitForLoadState("networkidle");
    ok("s2 report opens with analysis id", (await rep.content()).includes("Analysis ID") && (await rep.content()).includes("Checkpoint identifier"));
    await rep.screenshot({ path: `${out}/s2_report.png`, fullPage: true });
    await rep.pdf({ path: `${out}/s2_report.pdf`, format: "A4" });

    // explainability with RSNA DICOM (auto reference lookup)
    await p.goto(`${B}/#/explainability`); await p.waitForLoadState("networkidle");
    await p.getByRole("button", { name: "Remove file" }).click().catch(() => {});
    const dcm = fs.readdirSync(S).find((f) => f.endsWith(".dcm"));
    await p.setInputFiles('input[type="file"]', `${S}/${dcm}`);
    await p.getByRole("button", { name: "Generate explanations" }).click();
    await p.getByText("What influenced this model's prediction?").waitFor({ timeout: 120000 });
    ok("s2 explain shows RSNA reference", await vis(p.getByText(/RSNA dataset/).first()));
    ok("s2 explain metrics table", await vis(p.getByRole("cell", { name: "Predicted localization" })));
    await p.screenshot({ path: `${out}/s2_explain.png`, fullPage: true });

    // experiments
    await p.goto(`${B}/#/experiments`); await p.waitForLoadState("networkidle"); await p.waitForTimeout(800);
    ok("s2 experiments classification table", await vis(p.getByText("A. Classification performance")));
    await vis(p.locator(".recharts-line"));
    ok("s2 roc chart drawn", (await p.locator(".recharts-line").count()) > 0);
    await p.screenshot({ path: `${out}/s2_experiments_full.png`, fullPage: true });
    // overview recent analyses
    await p.goto(`${B}/#/`); await p.waitForLoadState("networkidle");
    await p.getByRole("link", { name: "Report", exact: true }).first().waitFor({ timeout: 30000 }).catch(() => {});
    ok("s2 recent analyses listed", (await p.getByRole("link", { name: "Report", exact: true }).count()) >= 1);
    await p.screenshot({ path: `${out}/s2_overview_after.png`, fullPage: true });
    ok("s2 no console errors in flows", errors.length === 0, errors.slice(0, 3).join(" | "));
    await ctx.close();
  }
  {
    const ctx = await browser.newContext(phone);
    const p = await ctx.newPage();
    await p.goto(`${B}/#/analysis`); await p.waitForLoadState("networkidle");
    await p.setInputFiles('input[type="file"]', `${S}/sample_xray.png`).catch(() => {});
    await p.getByRole("button", { name: "Analyse X-ray" }).click();
    await p.getByText("Model output", { exact: true }).waitFor({ timeout: 120000 });
    const overflow = await p.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    ok("s2 phone analysis no overflow", overflow <= 1, `(${overflow}px)`);
    await p.screenshot({ path: `${out}/s2_phone_analysis.png`, fullPage: true });
    await p.getByRole("button", { name: "Open menu" }).click();
    ok("s2 phone menu opens", await vis(p.getByRole("link", { name: "Research Experiments" })));
    await ctx.close();
  }
  await browser.close();
  console.log(results.join("\n"));
  console.log(`\n${results.filter((r) => r.startsWith("PASS")).length}/${results.length} passed`);
})().catch((e) => { console.error(e); console.log(results.join("\n")); process.exit(1); });

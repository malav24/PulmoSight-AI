import { BookOpen, Compass, FlaskConical, HelpCircle, Lightbulb, Scale, ShieldCheck, Target, Telescope, TriangleAlert } from "lucide-react";
import { Card, CardHeader, Collapsible, PageHeader } from "../components/ui";

function Section({ icon, title, children, tech }) {
  return (
    <Card>
      <CardHeader icon={icon} title={title} />
      <div className="space-y-3 px-5 py-4 text-[14px] leading-relaxed text-navy-700">
        {children}
        {tech && <Collapsible title="For researchers">{tech}</Collapsible>}
      </div>
    </Card>
  );
}

export default function About() {
  return (
    <div className="space-y-6">
      <PageHeader title="About the research" subtitle="RIGHT FOR THE RIGHT REASON: Explanation-Guided Deep Learning for Pneumonia Detection and Localization in Chest X-Rays." />
      <div className="grid gap-6 lg:grid-cols-2">
        <Section icon={HelpCircle} title="Research problem"
          tech="Shortcut learning (Geirhos et al., 2020) and confounding in CXR models (e.g. Zech et al., 2018; DeGrave et al., 2021) show high internal AUC with poor transfer.">
          <p>Deep-learning models can detect pneumonia-related changes on chest X-rays with high accuracy scores. But a high score does not show <i>why</i> the model decided. Models have been shown to use “shortcuts” - hospital markers, text, image borders or the type of scanner - instead of the lungs. Such a model may look accurate in testing and fail on new patients.</p>
        </Section>
        <Section icon={Telescope} title="Research gap">
          <p>Most pneumonia models are judged mainly by accuracy, and their explanations (heatmaps) are produced only <i>after</i> training, as a check. Few studies use radiologists' annotations to <b>guide</b> where the model looks during training, and fewer still test whether that guidance improves faithfulness, robustness to shortcuts, calibration and performance on other datasets - under a controlled comparison with an identical unguided model.</p>
        </Section>
        <Section icon={Target} title="Aim">
          <p>To develop and evaluate an explanation-guided hybrid CNN-Transformer that detects and localizes pneumonia-related lung opacity on chest X-rays, and to test whether guiding its attention with radiologist annotations makes it “right for the right reason”.</p>
        </Section>
        <Section icon={Compass} title="Objectives">
          <ol className="list-decimal space-y-1 pl-5">
            <li>Build a hybrid EfficientNet-B0 + Transformer with a classification head and a localization head.</li>
            <li>Train it with and without attention guidance from RSNA radiologist boxes (Model B vs Model A).</li>
            <li>Compare against ResNet50, DenseNet121, EfficientNet-B0 and ViT-Small baselines.</li>
            <li>Evaluate classification, localization, explanation faithfulness, shortcut robustness, calibration and NIH external generalisation.</li>
            <li>Present results in an interface that clinicians can understand, with clear limitations.</li>
          </ol>
        </Section>
        <Section icon={Lightbulb} title="Research questions">
          <ol className="list-decimal space-y-1 pl-5">
            <li>Does attention guidance change how much of the model's evidence falls inside radiologist-annotated regions?</li>
            <li>Does it change classification performance on the RSNA test set?</li>
            <li>Are guided models less affected when non-lung content (borders, markers, background) is masked?</li>
            <li>Does guidance change the performance drop on the NIH ChestX-ray14 dataset?</li>
            <li>Are the guided model's probabilities better calibrated?</li>
          </ol>
        </Section>
        <Section icon={Scale} title="Hypotheses" tech="Primary comparison: Model A vs Model B (paired DeLong for AUC; bootstrap CIs for localization/explanation metrics; ≥3 seeds). Hypotheses are to be tested - no result is assumed.">
          <ul className="list-disc space-y-1 pl-5">
            <li><b>H1:</b> Model B places more attention and Grad-CAM++ evidence inside annotated regions than Model A.</li>
            <li><b>H2:</b> Model B's classification AUC is not lower than Model A's (non-inferiority).</li>
            <li><b>H3:</b> Model B's predictions change less under shortcut stress tests.</li>
            <li><b>H4:</b> Model B shows a smaller AUC drop from RSNA to NIH.</li>
          </ul>
          <p className="text-xs text-navy-500">These are hypotheses. The study may support or reject them.</p>
        </Section>
        <Section icon={FlaskConical} title="Proposed methodology"
          tech="L_total = L_cls + α·L_loc + λ·L_attn; L_attn = attention mass outside the rasterised box mask (or KL to box distribution); patient-level 70/15/15 split; threshold and temperature fitted on validation only.">
          <p>Images are standardised (contrast enhancement, 384-pixel size). A convolutional network extracts image features; a small Transformer relates regions to each other. One head scores the image, another draws a localization map. For Model B, an extra training signal penalises attention placed outside the radiologist boxes. All models share the same data split, training schedule and selection rule, so differences can be attributed to the guidance.</p>
        </Section>
        <Section icon={BookOpen} title="Expected contributions">
          <ul className="list-disc space-y-1 pl-5">
            <li>A controlled test of explanation guidance for pneumonia detection (identical architecture, one changed factor).</li>
            <li>An evaluation that goes beyond accuracy: faithfulness, shortcut robustness, calibration and external data.</li>
            <li>An open, reproducible research pipeline and a clinician-oriented interface that states its limits.</li>
          </ul>
        </Section>
        <Section icon={TriangleAlert} title="Dataset limitations">
          <ul className="list-disc space-y-1 pl-5">
            <li>RSNA labels mark lung opacity suspicious for pneumonia, drawn by radiologists - not confirmed clinical diagnoses. Boxes are coarse rectangles.</li>
            <li>NIH ChestX-ray14 labels were extracted automatically from reports and contain errors; “Pneumonia” is defined differently from RSNA.</li>
            <li>RSNA images were drawn from the NIH dataset, so overlapping images must be removed before NIH can serve as an external test.</li>
            <li>Both datasets come from one US hospital system; performance elsewhere is unknown. Only frontal views are used.</li>
          </ul>
        </Section>
        <Section icon={ShieldCheck} title="Ethical considerations">
          <ul className="list-disc space-y-1 pl-5">
            <li>Research and education only - not a medical device, not for diagnosis or treatment decisions.</li>
            <li>Only public, de-identified datasets are used, under their licences. Uploaded images are processed in memory; the original file is deleted and only a hash and derived 384-px images are kept for reports.</li>
            <li>The system never invents predictions or results; it abstains when uncertain, when the input looks unusual or fails quality checks.</li>
            <li>Explanations are presented as computational outputs, not as clinical reasoning. Possible biases across patient groups have not been assessed.</li>
          </ul>
        </Section>
      </div>
    </div>
  );
}

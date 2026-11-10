// Replace only the discussion slide, retaining the original charts and embedded audio.
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const { Presentation, PresentationFile } = await import(
  pathToFileURL(require.resolve('@oai/artifact-tool')).href
);
const JSZip = require('jszip');
const args = Object.fromEntries(process.argv.slice(2).map(arg => {
  const split = arg.indexOf('=');
  if (split < 0) throw new Error('Use --name=value arguments');
  return [arg.slice(2, split), arg.slice(split + 1)];
}));
for (const name of ['root', 'input', 'output', 'work-dir', 'skill-dir', 'python', 'report-pages']) {
  if (!args[name]) throw new Error('Missing --' + name);
}
const root = path.resolve(args.root), work = path.resolve(args['work-dir']);
await fs.mkdir(work, { recursive: true });
await fs.mkdir(path.dirname(path.resolve(args.output)), { recursive: true });
const destination = path.join(root, 'results/final_submission_20261110');
const data = JSON.parse(await fs.readFile(path.join(destination, 'presentation_data.json'), 'utf8'));
const claims = data.local_alignment;
if (!claims || claims.conditions.length !== 100) throw new Error('Verified alignment values required');
const readouts = ['single', 'ensemble', 'aligned_fixed', 'aligned_calibrated'];
const titles = ['Single prompt', 'Fixed ensemble', 'Local / preset', 'Local / calibrated*'];
const clean = name => claims.conditions.find(r => r.region === 'clean' && r.readout === name);
const same = name => claims.conditions.find(r => r.region === 'own' && r.readout === name &&
  r.condition === 'semantic_same' && Number(r.level_db) === 5);
const pct = x => (100 * Number(x)).toFixed(2);
const values = [['Readout', 'Clean\naccuracy (%)', 'Same\naccuracy (%)', 'Same\nmacro-F1 (%)', 'Captures', 'Other\nerrors', 'Target in\ntop two (%)']];
for (const [index, name] of readouts.entries()) {
  const row = same(name);
  values.push([titles[index], pct(clean(name).accuracy_mean), pct(row.accuracy_mean), pct(row.macro_f1_mean),
    row.captures, row.other_errors, pct(Number(row.target_in_top2) / Number(row.n))]);
}
const zeroFolds = Object.values(claims.selection.folds).filter(row =>
  row.choice.beta_text === 0 && row.choice.beta_audio === 0).length;
const presetChange = 100 * (Number(same('aligned_fixed').accuracy_mean) - Number(same('ensemble').accuracy_mean));
const calibratedChange = 100 * (Number(same('aligned_calibrated').accuracy_mean) - Number(same('ensemble').accuracy_mean));
const detected = claims.primary.filter(row => Number(row.holm_p) < 0.05).length;
const sign = value => (value >= 0 ? '+' : '') + value.toFixed(2);
const narration = `I test PAT-style local feedback with the same frozen LAION-CLAP. This adaptation is not official or full PAT. ` +
  `At own power and plus five decibels, preset alignment changes same-category accuracy by ${sign(presetChange)} percentage points relative to the fixed ensemble. ` +
  `Captures change from ${same('ensemble').captures} to ${same('aligned_fixed').captures}. ` +
  `Preset macro-F1 is ${pct(same('aligned_fixed').macro_f1_mean)} percent, compared with ${pct(same('ensemble').macro_f1_mean)} for the ensemble. ` +
  `Calibrated accuracy changes by ${sign(calibratedChange)} points. ` +
  `Calibration uses non-test clean labels and is not zero-shot. ` +
  (zeroFolds ? `${zeroFolds} of five folds select zero alignment weights. ` :
    'All five folds select nonzero feedback weights. ') +
  (detected ? `${detected} of four corrected tests detect a change. ` : 'None of four corrected tests establishes an accuracy change. ') +
  `An accuracy change is not automatically reduced capture. CPU timing excludes setup and cache hits.`;
const evidence = 'results/extensions_20261111/clap_local_alignment/conditions.csv; ' +
  'primary_comparisons.csv; independent_audit.json. PAT: https://aclanthology.org/2025.naacl-long.616/';

// Match the existing 13.333 by 7.5 inch deck, Arial type and dark discussion slide.
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const slide = presentation.slides.add();
slide.background.fill = '#172A2D';
function text(value, x, y, width, height, size, color = '#FFFFFF', bold = false) {
  const shape = slide.shapes.add({ geometry: 'textbox', position: { left: x, top: y, width, height },
    fill: 'none', line: { fill: 'none', width: 0 } });
  shape.text = value;
  shape.text.style = { typeface: 'Arial', fontSize: size, color, bold,
    autoFit: 'none', wrap: 'square', insets: 0 };
  return shape;
}
text('PAT-style local audio–text alignment', 57.6, 46, 1162, 86, 48, '#FFFFFF', true);
text('Same frozen LAION-CLAP · own / +5 dB · 2,000 same-category trials', 59.5, 137, 1152, 51, 22.7, '#CCDCD8');
const table = slide.tables.add({ rows: 5, columns: 7, left: 67.2, top: 213,
  width: 1146, height: 300, columnWidths: [270, 146, 146, 146, 146, 146, 146], values });
table.rows[0].height = 72;
for (let row = 1; row < 5; row++) table.rows[row].height = 57;
table.borders.assign({ style: 'solid', fill: '#536366', width: 0.7 });
for (let row = 0; row < 5; row++) for (let column = 0; column < 7; column++) {
  const cell = table.getCell(row, column);
  cell.fill = row === 0 ? '#315B52' : '#172A2D';
  cell.text.style = { typeface: 'Arial', fontSize: row === 0 ? 21 : 23,
    color: '#FFFFFF', bold: row === 0 || column === 0,
    alignment: column === 0 ? 'left' : 'center', verticalAlignment: 'middle', insets: 9 };
}
text('Accuracy vs ensemble: preset ' + sign(presetChange) + ' pp; calibrated ' +
  sign(calibratedChange) + ' pp.', 72, 548, 1136, 48, 24, '#CCDCD8');
text('* Calibrated uses non-test clean labels. Captures and other errors are counts over all 2,000 trials.',
  72, 612, 1136, 32, 17, '#CCDCD8');
text('ELEC5305  /  QIHAN ZHU  /  PROJECT PRESENTATION', 57.6, 664.3, 960, 24, 13.3, '#B8C9C7');
text('11', 1152, 664.3, 62, 24, 14.7, '#B8C9C7').text.alignment = 'right';
const authoredPath = path.join(work, 'authored-slide.pptx');
await (await PresentationFile.exportPptx(presentation)).save(authoredPath);
const authored = await JSZip.loadAsync(await fs.readFile(authoredPath));
const source = await fs.readFile(args.input);
const zip = await JSZip.loadAsync(source);
const newSlide = await authored.file('ppt/slides/slide1.xml').async('string');
if (/\br:(?:id|embed|link)=/.test(newSlide)) throw new Error('New slide references uncopied external parts');
zip.file('ppt/slides/slide11.xml', newSlide);
let finalSlide = await zip.file('ppt/slides/slide12.xml').async('string');
if (!finalSlide.includes('Released prompt-list ablation; full PAT is not reproduced') &&
    !finalSlide.includes('LAION local-feedback adaptation; not official PAT')) {
  throw new Error('The expected source-deck PAT description is absent');
}
finalSlide = finalSlide.replace('Released prompt-list ablation; full PAT is not reproduced',
  'LAION local-feedback adaptation; not official PAT');
finalSlide = finalSlide.replace(/\d+-page report/g, args['report-pages'] + '-page report');
zip.file('ppt/slides/slide12.xml', finalSlide);
const segments = JSON.parse(await fs.readFile(path.join(destination, 'narration_segments.json'), 'utf8'));
segments[10] = { slide: 11, title: 'PAT-style local audio–text alignment', narration, evidence };
segments[11].narration = segments[11].narration.replace('The eleven-page report', 'The final report');
const escape = value => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
for (const index of [10, 11]) {
  const segment = segments[index], name = `ppt/notesSlides/notesSlide${index + 1}.xml`;
  let replaced = false;
  const xml = (await zip.file(name).async('string')).replace(/<p:sp\b[\s\S]*?<\/p:sp>/g, block => {
    if (!block.includes('type="body"')) return block;
    replaced = true;
    const body = '<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang="en-US"/>' +
      '<a:t>' + escape(segment.narration + '\n\nEvidence: ' + segment.evidence) +
      '</a:t></a:r></a:p></p:txBody>';
    return block.replace(/<p:txBody>[\s\S]*?<\/p:txBody>/, body);
  });
  if (!replaced) throw new Error('Speaker-note body placeholder missing');
  zip.file(name, xml);
}
const candidate = path.join(work, 'candidate.pptx');
await fs.writeFile(candidate, await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
const { finalizePresentation } = await import(pathToFileURL(
  path.join(args['skill-dir'], 'container_tools/artifact_tool_utils.mjs')).href);
await finalizePresentation({ workspaceDir: work, candidatePath: candidate,
  finalPath: path.resolve(args.output), pythonExecutable: args.python,
  integrityValidatorPath: path.join(args['skill-dir'], 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(args['skill-dir'], 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-heading-fit',
    '--require-native-table-slide', '6', '--require-native-table-slide', '7', '--require-native-table-slide', '11'],
  explicitTotalSlideCount: 12, requiredNativeTableOwnerSlides: [6, 7, 11],
  requiredNativeChartOwnerSlides: [6, 8], verifyArtifactToolImport: true,
  fontPolicy: { basis: 'reference', families: ['Arial'], referencePath: path.resolve(args.input),
    referenceSha256: crypto.createHash('sha256').update(source).digest('hex') },
  receiptPath: path.join(work, 'validation.json') });
await fs.writeFile(path.join(work, 'narration_segments.json'), JSON.stringify(segments, null, 2) + '\n');
await fs.writeFile(path.join(work, 'speaker_notes.txt'), segments.map(segment =>
  `SLIDE ${segment.slide}\n${segment.narration}\n\nEvidence: ${segment.evidence}`).join('\n\n'));
let elapsed = 0;
const timing = segments.map(segment => {
  const words = segment.narration.trim().split(/\s+/).length;
  const extra = segment.slide === 10 ? 30 : 0;
  const seconds = Math.ceil(words * 60 / 145) + extra;
  const row = { slide: segment.slide, start_seconds: elapsed, seconds, words,
    extra_pause_seconds: extra };
  elapsed += seconds;
  return row;
});
const words = timing.reduce((sum, row) => sum + row.words, 0);
const range = [words / 150 + 0.5, words / 130 + 0.5];
const clock = seconds => String(Math.floor(seconds / 60)).padStart(2, '0') + ':' +
  String(seconds % 60).padStart(2, '0');
const introduction = [
  'ELEC5305 FINAL PRESENTATION - RECORDING SCRIPT',
  'Qihan Zhu | When Background Noise Has Meaning | 11 November 2026', '',
  `Matches the 12-slide concise deck and the ${args['report-pages']}-page report.`,
  `Narration: ${words} words. Speech at 130-150 words/minute plus 30 seconds`,
  `of demonstration pauses: approximately ${range[0].toFixed(1)}-${range[1].toFixed(1)} minutes.`,
  'Loading speed and personal delivery may add time. The supplied course requirements',
  'say brief without a minute limit; this is a production estimate, not a course rule.',
  'The single formal report contains five core figures; detailed data remain in the repository.',
  '', 'SETUP',
  'Use the final presentation and the run guide in docs/code_demo.md.',
  'On slide 10 switch to the local demonstration, press Run, explain the processing,',
  'and play the two five-second examples without speaking over them.',
  'The existing execution clip has no author narration. Identify playback as playback.',
  'Review these notes and explain them naturally in your own words.', '',
];
const script = introduction.join('\n') + '\n' + segments.map((segment, index) => {
  const time = timing[index];
  return `${clock(time.start_seconds)}-${clock(time.start_seconds + time.seconds)} | SLIDE ${segment.slide} | ${segment.title}\n` +
    `Visual: corresponding slide in the concise presentation.\n\nNarration:\n${segment.narration}\n\n` +
    `Evidence (not spoken): ${segment.evidence}\n`;
}).join('\n');
await fs.writeFile(path.join(work, 'video_narration_final.txt'), script);
await fs.writeFile(path.join(work, 'narration_timing.json'), JSON.stringify({ slides: 12, words,
  estimated_minutes_at_130_to_150_wpm: range.map(value => Number(value.toFixed(2))),
  timeline_seconds: elapsed, extra_demo_pause_seconds: 30, final_narrated_video_recorded: false,
  segments: timing }, null, 2) + '\n');
console.log(JSON.stringify({ slides: 12, edited_slides: [11, 12], primary_table_rows: 4 }));

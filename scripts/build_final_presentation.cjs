/* Editable, native-chart research deck. Result values come from audited tables. */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const PptxGenJS = require('pptxgenjs');
const JSZip = require('jszip');
const root = path.resolve(__dirname, '..');
const dest = path.join(root, 'results/final_submission_20261110');
const d = JSON.parse(fs.readFileSync(path.join(dest, 'presentation_data.json'), 'utf8'));
const p = new PptxGenJS();
p.layout = 'LAYOUT_WIDE';
p.author = 'Qihan Zhu';
p.subject = 'ELEC5305 concise research presentation and live code demonstration';
p.title = 'When Background Noise Has Meaning';
p.company = 'ELEC5305 · University of Sydney';
p.lang = 'en-AU';
p.theme = {headFontFace:'Arial', bodyFontFace:'Arial', lang:'en-AU'};
const C = {ink:'172A2D',muted:'536366',pale:'F0F5F4',white:'FFFFFF',svm:'0072B2',cnn:'D55E00',clap:'009E73',grid:'DCE4E3',grey:'BAC5C5'};
let count = 0;
const notes = [];
const briefNotes = ["An environmental recording can contain a target and another recognisable event. My main question is how three different audio representations fail under that competition, compared with generic noise at controlled power. I use the same mixtures and explicit evaluation rules. The report distinguishes measured behaviour from a claim that semantic meaning alone causes failure.", "The experiment retains all two thousand ESC-50 clips and the five official folds. Each target receives white noise, a DEMAND background and two kinds of competing event. The exact partner or noise segment remains fixed across the three levels. Zero decibels is an equal-power stress test with an ambiguous designated single label.", "The official ESC-50 record lists 39.6 percent accuracy for its MFCC and zero-crossing SVM, and 64.5 for Piczak’s CNN. Our 48.9 and 67.9 provide context, not a controlled improvement, because the pipelines differ. Between-class learning mixes sounds with proportional labels; our augmentation retains one target label. The revised CLAP paper reports 88 to 91 percent after overlap exclusions. Our 81.75 uses all two thousand clips, a specified checkpoint and a different prompt, so the gap cannot be attributed to prompts alone.", "The SVM uses ninety-two interpretable features, the compact CNN learns from sixty-four log-Mel bands, and CLAP compares audio with fifty fixed text descriptions. I keep training, validation and testing separate. The custom STFT agrees with SciPy and with corresponding frames from the actual course MATLAB code. CLAP has a large external pretraining advantage.", "The proposal measured both component powers on the target activity mask. Sparse distractors can have almost no energy there, causing extreme gains. The implementation uses each source’s own activity as the default and also evaluates the original target-mask convention in full. Common attenuation avoids hard clipping; a gain-matched clean control measures its effect.", "At plus five decibels, white noise reduces SVM macro-F1 to 2.03 percent. CLAP reaches 71.49 under DEMAND and 48.86 under same-category competition. The new table uses errors as its denominator: same-category capture is 18.53 percent for SVM, 31.26 for CNN and 66.95 for CLAP. These are not the all-trial percentages in the listening examples. On high-overlap targets, the original CNN loses 3.28 and 2.94 accuracy points relative to matched DEMAND; neither contrast passes the declared Holm correction.", "Across three CNN seeds, white-noise training raises white-noise macro-F1 from 7.48 to 43.95 percent, but same-category performance changes little. Semantic training reaches 36.05 percent while clean performance falls to 61.70. Direct semantic-over-white augmentation accuracy gains remain inconclusive after correction. Unlike between-class learning, these mixtures keep the target label. The result tests target retention, not proportional multi-event prediction.", "Argmax forces one label. At own power and plus five decibels, 236 of 632 same-category captures retain the target second, or 37.34 percent. Different-category retention is 203 of 612, or 33.17 percent. The chart instead uses all trials as its denominator. Top-two ranking partially recovers the target; it is not a calibrated event detector. Fixed prompt ensembling also changes readout without retraining CLAP.", "Raw white-noise Mel correlation changes from minus 0.728 to minus 0.008 after bandwidth normalisation. Acoustic descriptors therefore need care. On subsets close in spectral similarity, masking fraction and temporal overlap, CLAP remains sixteen to twenty-six accuracy points below DEMAND across both power conventions. CNN intervals include zero. Only about nine to ten percent of targets remain; this is conditional association, not semantic causation.", "I now switch to the local code demonstration and select Run. The program reads the source recordings, rebuilds the plus-five-decibel mixture, checks SIR and peak level, and runs all three saved models. It then compares fresh outputs with the original evaluation. The clean controls identify can opening. In the mixture, CNN and CLAP choose clock alarm; CLAP ranks the target third in this example. I play the control and mixture at the same volume. This is an illustrative failure case, not an aggregate result.", "Representation, training and readout answer different questions. For CLAP under same-category interference at plus five decibels, the median target-minus-distractor margin is 0.0873; 38.05 percent of margins are negative. Median representation displacement is 0.1905 on correct trials and 0.4087 on errors. These associations do not establish a mechanism. About one-third of captured targets remain second. Acoustic restriction retains a CLAP gap, while CNN intervals include zero in those smaller subsets. Limited backgrounds, shared sources and unknown pretraining overlap constrain generalisation.", "The eleven-page report contains the main argument, numerical literature comparison and five core figures. Complete diagnostics remain in the code repository. The website links the code, attributed audio and execution clip. I used OpenAI Codex with GPT-6 Astra and incorporated Anthropic Claude code and review material, whose model version was not recorded. Assistance included implementation, analyses and writing, beyond language editing. I remain responsible for the work. The execution clip is a component of the video; the final submission also needs my own explanation."];
const segments = [];
const slideTitles = [];
function text(s,t,x,y,w,h,size=20,opts={}) {s.addText(t,{x,y,w,h,fontFace:'Arial',fontSize:size,color:C.ink,margin:0,breakLine:false,...opts});}
function rect(s,x,y,w,h,fill,line=fill) {s.addShape(p.ShapeType.rect,{x,y,w,h,fill:{color:fill},line:{color:line,width:.5}});}
function base(title,subtitle='',dark=false) {
 const s=p.addSlide();s.background={color:dark?C.ink:C.white};count++;slideTitles.push(title||'When Background Noise Has Meaning');
 text(s,String(count).padStart(2,'0'),12.0,6.92,.65,.24,11,{color:dark?'B8C9C7':C.muted,align:'right'});
 text(s,'ELEC5305  /  QIHAN ZHU  /  PROJECT PRESENTATION',.6,6.92,10,.24,10,{color:dark?'B8C9C7':C.muted,charSpacing:1});
 if(title)text(s,title,.6,.48,12.1,.9,36,{bold:true,color:dark?C.white:C.ink});
 if(subtitle)text(s,subtitle,.62,1.43,12.0,.53,17,{color:dark?'C9D9D6':C.muted});
 return s;
}
function note(s,t,source='') {t=briefNotes[count-1]||t;const n=t+(source?'\n\nEvidence: '+source:'');s.addNotes(n);notes.push(n);segments.push({slide:count,title:slideTitles[count-1],narration:t,evidence:source});}
function label(s,title,body,x,y,w,colour=C.ink) {const two=title.includes('\n');text(s,title,x,y,w,two ? 0.9 : 0.45,23,{bold:true,color:colour});text(s,body,x,y+(two?1.1:.62),w,1.45,20);}
function foot(s,t) {text(s,t,.62,6.35,12.0,.38,12,{color:C.muted});}
function chart(s,data,x,y,w,h,opts={}) {
 s.addChart(p.ChartType.bar,data,{x,y,w,h,barDir:'col',barGrouping:'clustered',
  showLegend:data.length>1,legendPos:'b',legendFontFace:'Arial',legendFontSize:14,
  showValue:true,dataLabelPosition:'outEnd',dataLabelFormatCode:'0.0',dataLabelColor:C.ink,dataLabelFontSize:13,
  catAxisLabelFontFace:'Arial',catAxisLabelFontSize:15,catAxisLabelColor:C.ink,
  valAxisLabelFontFace:'Arial',valAxisLabelFontSize:13,valAxisLabelColor:C.muted,
  valAxisMinVal:0,valAxisMaxVal:100,valAxisMajorUnit:20,catAxisLineShow:false,valAxisLineShow:false,
  catGridLine:{style:'none'},valGridLine:{color:C.grid,width:.6},showBorder:false,showCatName:false,
  showTitle:true,title:'Macro-F1 (%)',titleFontSize:17,titleColor:C.muted,
  chartColors:[C.svm,C.cnn,C.clap],showMarker:false,showShadow:false,showPlotArea:false,
  showChartArea:false,showCatName:false,...opts});
}
function core(system,condition,region='own') {return 100*Number(d.core.find(r=>r.system===system&&r.region===region&&r.condition===condition+'@+5dB').macro_f1_mean);}
const conditionNames=['White noise','DEMAND','Same category','Different category'];
const conditions=['white','demand','semantic_same','semantic_different'];

// 1. Problem-led title with an editable overlapping-signal motif.
{
 const s=base('', '', true);
 text(s,'WHEN BACKGROUND\nNOISE HAS MEANING',.7,.85,11.9,1.7,46,{bold:true,color:C.white});
 text(s,'Environmental sound classification\nunder semantic interference',.74,2.85,10.9,1.1,28,{color:'CCDED9'});
 text(s,'Qihan Zhu  ·  SID 550489593',.74,6.05,9,.3,17,{color:C.white});
 note(s,'A classifier may hear a target sound and a second meaningful event at the same time. My project asks how three different audio representations fail under that competition, compared with ordinary noise. The contribution is controlled signal processing and failure analysis. It is not a contest that gives all models the same training history. AI-assisted implementation and preparation are disclosed in the report.','Project report, research question and scope.');
}
// 2. Experimental matrix.
{
 const s=base('Same target. Same relative level. Different interference.','2,000 ESC-50 clips · official five-fold evaluation · paired mixtures for every model');
 const boxes=[['White noise','Acoustically nonspecific',C.svm],['DEMAND','Traffic · station · metro',C.ink],['Same category','Different fine class',C.cnn],['Different category','Different coarse category',C.clap]];
 boxes.forEach(([a,b,c],i)=>{const x=.65+i*3.15;rect(s,x,2.25,2.95,2.05,C.pale);text(s,a,x+.2,2.56,2.55,.7,23,{bold:true,color:c});text(s,b,x+.2,3.4,2.55,.6,17);});
 text(s,'+10 dB',1.0,4.85,3,.6,34,{bold:true});text(s,'+5 dB',5.0,4.85,3,.6,34,{bold:true});text(s,'0 dB',9.0,4.85,3,.6,34,{bold:true});
 text(s,'Weaker interference',1.0,5.55,3,.35,17);text(s,'Primary comparison',5.0,5.55,3,.35,17);text(s,'Equal-power stress test',9.0,5.55,3.1,.35,17);
 foot(s,'24,000 own-activity mixtures; 23,991 target-activity mixtures after three undefined pairs × three levels are excluded.');
 note(s,'The targets are all two thousand ESC-50 clips, evaluated with the official five folds. Each gets white Gaussian noise, a real DEMAND background, a same-category event and a different-category event. The exact pair or noise segment stays fixed as its level changes. Every model receives the same mixtures. Plus five decibels is the primary comparison. At zero decibels the two meaningful events have equal power, so a single designated correct label becomes an intentionally difficult convention.','Final manifest audits; report methods.');
}
// Literature and course foundations.
{
 const s=base('Literature and course foundations','Each method addresses a different part of the research question');
 const rows=[['ESC-50 baselines','Historical SVM / CNN: 39.6% / 64.5% accuracy','Ours: 48.9% / 67.9%; different pipelines, not a controlled gain'],['Between-class learning','Mixture proportions become soft training targets','Our semantic augmentation keeps one designated target label'],['CLAP / PAT','Paper: 88–91% with overlap exclusions; ours: 81.75%','Different checkpoint/protocol; fixed ensemble is not full PAT'],['Course DSP / research gap','STFT, features, SNR/SIR and known distractor identity','Measure where the decision moves under identical mixtures']];
 rows.forEach(([a,b,c],i)=>{const y=2.05+i*1.02;text(s,a,.75,y,3.0,.45,21,{bold:true});text(s,b,4.25,y,8.25,.42,20);text(s,c,4.25,y+.44,8.25,.42,17,{color:C.muted});});
 foot(s,'Report Section 2 compares strengths, limitations and the reason for selecting each representation.');
 note(s,'','Course Lectures 3/4/6; ESC-50 https://github.com/karolpiczak/ESC-50; BC learning https://arxiv.org/abs/1711.10282; CLAP revised Table 4 and Appendix H https://arxiv.org/abs/2211.06687v4; PAT https://aclanthology.org/2025.naacl-long.616/');
}
// 3. Three representation paradigms.
{
 const s=base('Three representations, one controlled experiment','Deployment behaviour is comparable; total pretraining history is not.');
 const models=[['A','Handcrafted + SVM','STFT → spectral descriptors\nMFCCs + deltas → 92 features',C.svm,'Training-fold standardisation'],['B','Compact log-Mel CNN','STFT → Mel filters → log power\n36,258 trainable parameters',C.cnn,'Three optimisation seeds'],['C','Frozen LAION-CLAP','Audio / text embedding similarity\n50 fixed class descriptions',C.clap,'No training or fine-tuning']];
 models.forEach(([id,title,body,c,tag],i)=>{const y=2.05+i*1.35;s.addShape(p.ShapeType.ellipse,{x:.72,y:y+.06,w:.62,h:.62,line:{color:c},fill:{color:c}});text(s,id,.72,y+.15,.62,.35,19,{color:C.white,align:'center',bold:true});text(s,title,1.6,y,4.25,.5,24,{bold:true,color:c});text(s,body,5.7,y,6.7,.75,19);text(s,tag,1.6,y+.62,4.15,.35,16,{color:C.muted});});
 foot(s,'Custom STFT checked against SciPy and actual course MATLAB code; maximum absolute discrepancy 2.93 × 10⁻¹⁴.');
 note(s,'The SVM aggregates interpretable DSP descriptors into ninety-two features. The compact CNN learns from log-Mel spectrograms and has just over thirty-six thousand parameters. CLAP is a frozen audio-language model with substantial external pretraining. It compares an audio vector with textual class references. I preserve separate training, validation and test handling, and test the custom STFT against both SciPy and the actual course MATLAB implementation after reconciling padding and frame alignment.','Course STFT audit; clean model definitions; CLAP provenance.');
}
// 4. Power convention explanation.
{
 const s=base('Power matching depends on where power is measured','The original target-mask definition is retained as a full sensitivity analysis.');
 text(s,'Target activity Mₓ',.7,2.1,3.2,.4,22,{bold:true});text(s,'Distractor activity Mᵈ',.7,3.0,3.2,.4,22,{bold:true});
 rect(s,4.1,2.12,8.2,.4,'EDF1F1');rect(s,4.1,3.02,8.2,.4,'EDF1F1');rect(s,5.0,2.12,3.0,.4,C.cnn);rect(s,7.0,3.02,4.2,.4,C.clap);
 label(s,'Own-activity power','Measure each source over\nits own active interval.',.75,4.05,5.55,C.ink);
 label(s,'Target-activity power','Measure both sources over\nthe target’s active interval.',6.9,4.05,5.55,C.ink);
 foot(s,'Illustrative masks. Common peak attenuation prevents clipping; a gain-matched clean control isolates its effect.');
 note(s,'The proposal measured both powers on the target activity mask. A sparse distractor may have almost no energy in that interval but a large event elsewhere. Scaling that tiny overlap can produce extreme gains. The implementation therefore adopted each source’s own activity mask as its default, while also running the original target-mask protocol in full. These are different measurement questions. Neither definition removes temporal-overlap differences. Common attenuation avoids clipping, and a clean control receives exactly the same attenuation.','Report equations and documented proposal deviation.');
}
// 5. Main model-specific failure result.
{
 const s=base('Interference power alone does not predict failure','Own-activity protocol · +5 dB · five-fold mean macro-F1');
 chart(s,['svm','cnn','clap'].map(system=>({name:system==='clap'?'CLAP · single prompt':system.toUpperCase(),labels:conditionNames,values:conditions.map(c=>core(system,c))})),.65,2.0,8.2,4.15,{catAxisLabelFontSize:12,dataLabelFontSize:11,legendFontSize:12});
 text(s,'Where do errors go?',9.1,2.15,3.5,.4,21,{bold:true});
 text(s,'Capture rate (%) · errors only',9.1,2.65,3.5,.4,14,{color:C.muted});
 const captures=[['System','Same','Different'].map(t=>({text:t,options:{bold:true,color:C.white,fill:C.ink}}))];
 for(const system of ['svm','cnn','clap'])captures.push([system.toUpperCase(),...['semantic_same','semantic_different'].map(condition=>(100*Number(d.editorial.capture.find(r=>r.system===system&&r.condition===condition).capture_rate)).toFixed(2))]);
 s.addTable(captures,{x:9.1,y:3.2,w:3.5,colW:[1.05,1.05,1.4],rowH:.55,fontFace:'Arial',fontSize:16,color:C.ink,margin:.08,border:{type:'solid',pt:.5,color:C.grid},align:'center',valign:'mid'});
 text(s,'Captures ÷ errors, pooled over folds.\nNot captures ÷ 2,000 trials.',9.1,5.58,3.5,.5,12,{color:C.muted});
 foot(s,'High-overlap CNN − DEMAND accuracy: −3.28 / −2.94 pp (n = 1,464 / 1,465); both Holm p = 0.0604. Original seed; same / different.');
 note(s,'At the same nominal relative level, the patterns depend strongly on representation. White noise almost destroys the handcrafted SVM’s macro-F1. CLAP is comparatively strong on white noise and DEMAND, but drops much further on meaningful event mixtures. The compact CNN lies between these behaviours. These results reject a universal ordering in which meaningful interference is always worse for every model. The report retains both power conventions and all levels rather than selecting a favourable condition.','all_conditions_verified.csv; fig2_capture_rate_vs_sir.csv; cnn_overlap_replication.csv, own +5 dB, seed 5305.');
}
// 8. Training intervention results appear only after independent verification.
{
 const s=base('Does robustness training transfer to competing events?','Same augmentation probability and epoch cap · clean validation · three CNN seeds');
 if(d.augmentation_complete){
  const format=(rr,system)=>{const vv=rr.map(r=>100*Number(r.macro_f1)), mean=vv.reduce((a,b)=>a+b,0)/vv.length;const sd=system==='cnn'?Math.sqrt(vv.reduce((a,b)=>a+(b-mean)**2,0)/(vv.length-1)):100*Number(rr[0].macro_f1_fold_sd);return mean.toFixed(1)+' ± '+sd.toFixed(1)};
  const table=[['Model / training','Clean','White','DEMAND','Same','Different'].map(t=>({text:t,options:{bold:true,color:C.white,fill:C.ink}}))];
  for(const system of ['svm','cnn'])for(const regime of ['clean','wgn','semantic']){
   const cells=[{text:system.toUpperCase()+' / '+{clean:'clean',wgn:'WGN',semantic:'semantic'}[regime],options:{bold:true,color:C[system]}}];
   cells.push(format(d.clean_variants.filter(r=>r.system===system&&r.regime===regime),system));
   for(const condition of conditions)cells.push(format(d.interventions.filter(r=>r.system===system&&r.regime===regime&&r.region==='own'&&r.condition===condition&&Number(r.level_db)===5),system));
   table.push(cells);
  }
  s.addTable(table,{x:.65,y:2.18,w:12.0,colW:[2.3,1.94,1.94,1.94,1.94,1.94],rowH:.49,fontFace:'Arial',fontSize:16,color:C.ink,margin:.08,border:{type:'solid',pt:.5,color:C.grid},align:'center',valign:'mid'});
  text(s,'Own activity · macro-F1 (%) · interference conditions at +5 dB',.7,5.85,11.9,.3,16,{color:C.muted});
  foot(s,'CNN: mean ± seed SD; SVM: mean ± fold SD. Direct CNN semantic-over-WGN accuracy gains do not pass the 156-test Holm correction.');
 } else {
  ['Clean training','WGN augmentation','Semantic augmentation'].forEach((t,i)=>{rect(s,.7+i*4.2,2.35,3.9,1.6,C.pale);text(s,t,.95+i*4.2,2.7,3.4,.7,25,{bold:true});});
  text(s,'Training and evaluation are still running.',.8,4.65,11.7,.7,30,{bold:true});
  text(s,'This page is filled only after all declared variants pass independent checks.',.8,5.5,11.7,.5,20,{color:C.muted});
 }
 note(s,d.augmentation_complete?'Across all three CNN seeds, WGN augmentation raises white-noise macro-F1 from 7.48 to 43.95 percent, but same-category interference changes only from 32.99 to 33.46 percent. Semantic training reaches 36.05 percent for same-category and 33.73 for different-category interference, while clean macro-F1 falls from 65.78 to 61.70 percent. These are measured trade-offs. Direct semantic-versus-WGN paired accuracy advantages for the CNN do not pass the common 156-test Holm correction, so modest positive point estimates are not an established superiority claim. All seeds are retained. Augmentation probability and maximum epoch budget are shared; actual refit durations are selected separately by clean validation.':'This is a preparation page. The intervention results are not yet complete and must be inserted only after independent verification. Do not narrate expected improvements as measured results.','Extension protocol and independently audited intervention tables.');
}
// 10. Exact ranked readout, with all four states and explicit denominators.
{
 const s=base('A captured decision can retain the target at rank two','Single-template CLAP · +5 dB · own activity · 2,000 targets per category type');
 const rr=['semantic_same','semantic_different'].map(c=>d.closing.readout.find(r=>r.variant==='clap_baseline'&&r.region==='own'&&r.condition===c&&r.k==='2'&&Number(r.level_db)===5));
 const states=['target_only','both','distractor_only','neither'];
 chart(s,states.map((state,i)=>({name:['Target only','Both','Distractor only','Neither'][i],labels:['Same category','Different category'],values:rr.map(r=>100*Number(r['mixture_'+state]))})),.65,2.05,7.6,4.15,{title:'Top-two states (% of all semantic trials)',barGrouping:'stacked',dataLabelPosition:'ctr',chartColors:['E5F1EB','87CDB8','4F9E85','D5DADA'],legendFontSize:12,catAxisLabelFontSize:15});
 text(s,rr.map(r=>(100*Number(r.argmax_captured_target_rank_le_k)).toFixed(2)+'%').join(' / '),8.6,2.27,4.05,.7,30,{bold:true,color:C.clap});
 text(s,'of original captures retain\nthe target at rank two\n(same / different category)',8.6,3.08,4.05,1.05,19);
 text(s,'Top-2 helps some cases.\nIt does not prove that the\nmodel heard both sources.',8.6,4.55,4.05,1.15,21,{bold:true});
 foot(s,'Clean-control joint inclusion: 5.40% / 1.70%. Target-power results and k = 1, 3, 5 sensitivities are retained.');
 note(s,`Argmax forces one label. I therefore re-read the saved fifty-class scores with an exact top-two rule, keeping the original results unchanged. At plus five decibels under own power, the target remains second in ${rr[0].captured_target_in_topk_count} of ${rr[0].argmax_captured_count} same-category captures, or ${(100*Number(rr[0].argmax_captured_target_rank_le_k)).toFixed(2)} percent, and ${rr[1].captured_target_in_topk_count} of ${rr[1].argmax_captured_count} different-category captures, or ${(100*Number(rr[1].argmax_captured_target_rank_le_k)).toFixed(2)} percent. Among all mixtures, both labels appear in the top two in 26.05 and 21.00 percent. However, 23.35 and 24.05 percent include only the distractor among the two designated labels. The target-power analysis shows the same partial-recovery pattern. This is ranking, not a calibrated detector or evidence of subjective hearing.`, 'closing_20261110/readout_outcomes.csv; Complete mixture and control ranks; source-cluster intervals in readout_uncertainty.csv.');
}
// 12. Conditional acoustic comparison. Native shapes retain editable intervals.
{
 const s=base('CLAP retains a gap in acoustically close subsets','Same-target comparison · +5 dB · own activity · primary calipers');
 const primary=d.closing.acoustic_primary.filter(r=>r.region==='own'&&r.comparison.endsWith('__demand'));
 const x0=3.0,x1=8.0,x=v=>x0+(v+40)/60*(x1-x0);
 [-40,-20,0,20].forEach(v=>{s.addShape(p.ShapeType.line,{x:x(v),y:2.1,w:0,h:3.64,line:{color:v===0?C.muted:C.grid,width:v===0?1.2:.6}});text(s,String(v),x(v)-.3,5.85,.6,.3,15,{align:'center',color:C.muted});});
 ['svm','cnn','clap'].forEach((system,i)=>['semantic_same','semantic_different'].forEach((condition,j)=>{
  const row=primary.find(r=>r.system===system&&r.a_condition===condition),y=2.42+(i*2+j)*.59;
  const mid=100*Number(row.accuracy_difference),lo=100*Number(row.ci_low),hi=100*Number(row.ci_high);
  text(s,system.toUpperCase()+' / '+(j?'different':'same'),.72,y-.17,2.1,.35,17,{color:C[system],bold:true});
  s.addShape(p.ShapeType.line,{x:x(lo),y,w:x(hi)-x(lo),h:0,line:{color:C[system],width:2}});
  [lo,hi].forEach(v=>s.addShape(p.ShapeType.line,{x:x(v),y:y-.07,w:0,h:.14,line:{color:C[system],width:1.4}}));
  s.addShape(p.ShapeType.ellipse,{x:x(mid)-.055,y:y-.055,w:.11,h:.11,line:{color:C[system]},fill:{color:C[system]}});
 }));
 text(s,'Semantic − DEMAND accuracy (percentage points)',2.35,6.13,6.2,.33,15,{align:'center',color:C.muted});
 text(s,'CNN intervals include zero.\nCLAP remains lower.',8.55,2.3,4.05,1.05,24,{bold:true});
 text(s,'210 / 178 targets retained\n(same / different category)\n≈ 9–10.5% of the benchmark',8.55,3.8,4.05,1.05,19);
 text(s,'Selected-population association;\nnot proof of semantic causation.',8.55,5.15,4.05,.9,19,{color:C.muted});
 foot(s,'Pointwise 95% target-source bootstrap intervals, 2,000 resamples. Both protocols: CLAP −16 to −26 pp. Subsets differ from high-overlap groups.');
 note(s,'I next compare the same target under semantic interference and DEMAND, retaining pairs close on all three acoustic descriptors. Own-power matching retains 210 same-category and 178 different-category targets; target power retains 205 and 187. These are only about nine to ten and a half percent of the benchmark. Across the two power definitions, CLAP remains sixteen to twenty-six accuracy points below DEMAND, while all four CNN intervals include zero. The intervals resample original target-source recordings and are conditional on fitted models. This does not overturn the larger high-overlap comparison: the selected targets differ. Unmeasured acoustics and salience remain, and regression coefficient shrinkage is not a percentage explained. The finding is a residual association, not proof that meaning itself caused the gap.', 'closing_20261110/acoustic_matched_comparisons.csv; Calipers: 0.10 target-active spectral cosine, 0.05 masked-cell fraction, 0.10 temporal overlap. Core CNN seed 5305, single-template CLAP.');
}

// 13. Real recorded failure example, embedded audio.
{
 const s=base('Real code execution and sound examples','Can opening + clock alarm · +5 dB SIR · full target-activity overlap');
 const ex=d.listening;
 [['Gain-matched clean control','clean_matched',.8],['Target + weaker clock alarm','mixture',7.0]].forEach(([label,key,x])=>{
  rect(s,x,2.2,5.5,1.65,C.pale);text(s,label,x+.25,2.46,4.95,.65,23,{bold:true});
  s.addMedia({type:'audio',path:path.join(root,'examples/listening_own_20261109',ex.audio[key].file),x:x+.25,y:3.22,w:.6,h:.4});
  text(s,'Play 5-second audio',x+1.0,3.25,4,.3,15,{color:C.muted,hyperlink:{url:'https://raw.githubusercontent.com/Charlie-Uni/elec5305-project-550489593/main/examples/listening_own_20261109/'+ex.audio[key].file}});
 });
 ['svm','cnn','clap'].forEach((system,i)=>{const x=.8+i*4.17;const pp=ex.predictions[system];text(s,system.toUpperCase(),x,4.45,3.85,.4,22,{bold:true,color:C[system]});text(s,d.labels[String(pp.control_prediction)].replaceAll('_',' ')+'\n↓\n'+d.labels[String(pp.prediction)].replaceAll('_',' '),x,5.0,3.85,1.1,20);});
 text(s,'Open the live code demonstration',.8,6.27,8.0,.35,17,{color:C.clap,hyperlink:{url:'http://127.0.0.1:8766/'}});
 text(s,'Recorded execution video',9.1,6.27,3.4,.35,15,{color:C.clap,hyperlink:{url:'https://charlie-uni.github.io/elec5305-project-550489593/results/final_submission_20261110/code_demo.mp4'}});
 note(s,'First play the gain-matched clean control, then the mixture, without speaking over either clip. All three core controls classify the target as can opening. With a weaker clock-alarm distractor, the CNN and single-prompt CLAP classify the clock alarm; the SVM produces another wrong class. This is an illustration selected from observed results, not proof that every can-opening and clock-alarm pair behaves this way. Its ratio, overlap, source recordings and audio-export hashes are recorded in the listening manifest.','examples/listening_own_20261109/examples.json, row '+ex.row+'. Can Open.WAV by LG, CC BY: https://freesound.org/people/LG/sounds/101404/ . alarm clock.aif by hookhead, CC0: https://freesound.org/people/hookhead/sounds/13613/ . ESC-50 compilation by Karol J. Piczak, CC BY-NC 3.0: https://creativecommons.org/licenses/by-nc/3.0/ . Changes: ESC-50 cropping, controlled mixing, common attenuation and PCM16 export.');
}
// 15. Takeaways and explicit limits.
{
 const s=base('Discussion and conclusion','Representation, training and readout explain different parts of the evidence',true);
 [['01','Representation matters','CLAP displacement: correct 0.1905 / error 0.4087.\nOwn / +5 dB / same category; association, not cause.'],['02','Readout changes the answer','Some argmax captures retain the target\nat rank two; many do not.'],['03','Acoustic controls narrow the claim','CLAP retains a gap in selected subsets;\nthis does not establish semantic causation.']].forEach(([n,a,b],i)=>{const y=2.15+i*1.27;text(s,n,.75,y,.7,.55,31,{bold:true,color:'73BAA1'});text(s,a,1.75,y,10.8,.5,26,{bold:true,color:C.white});text(s,b,1.75,y+.55,10.8,.7,19,{color:'CCDCD8'});});
 text(s,'Limits: three backgrounds · one pair per category type · pretraining overlap · 0-dB label ambiguity',.75,6.2,11.8,.4,13,{color:'CCDCD8'});
 note(s,'Interference power alone does not explain these failures. The analyses distinguish representation, readout and measured acoustic overlap. Top-two ranking recovers some original CLAP captures, while close-descriptor subsets retain a CLAP disadvantage and uncertain CNN contrasts. The project’s strength is the controlled DSP protocol and the ability to trace a wrong decision to a meaningful distractor. Its limits remain important: only three DEMAND backgrounds, one pair of each type per target, some official source reuse, possible CLAP pretraining overlap and ambiguity at equal power. The results support measured behavioural differences, not an isolated causal claim that semantics alone determines robustness.','Research report conclusions and limitations.');
}
// 16. Sources and reproducibility.
{
 const s=base('Report and reproducible code','11-page report · five core figures · working code · execution video');
 const refs=[['ESC-50 / Piczak','Official folds, dataset caveats and classic spectrogram CNN'],['BC learning / Tokozume et al.','Mixtures with proportional labels, ICLR 2018'],['LAION-CLAP / Wu et al.','ICASSP 2023; numerical context from revised arXiv v4'],['PAT / Seth et al.','Released prompt-list ablation; full PAT is not reproduced'],['Mix-CLAP / Kohno et al.','Noise-aware audio-language model context, ICASSP 2026'],['RSA-Bench / Zhang et al.','Structured-background speech/reasoning context, ACL Findings 2026']];
 refs.forEach(([a,b],i)=>{text(s,a,.75,2.0+i*.53,4.15,.45,20,{bold:true});text(s,b,5.1,2.0+i*.53,7.45,.45,18);});
 text(s,'github.com/Charlie-Uni/elec5305-project-550489593',.75,5.42,11.85,.5,22,{color:C.clap,hyperlink:{url:'https://github.com/Charlie-Uni/elec5305-project-550489593'}});
 text(s,'Audio: LG, Can Open.WAV (CC BY); hookhead, alarm clock.aif (CC0). ESC-50 compilation: CC BY-NC 3.0.',.75,6.03,11.85,.22,11,{color:C.muted});
 foot(s,'AI: OpenAI Codex / GPT-6 Astra; Anthropic Claude (version not recorded). Substantial code, analysis and drafting assistance; see report.');
 note(s,'Full references, licences, code ownership and AI-assistance disclosure are in the report and repository. The project distinguishes original DSP and evaluation code from external CLAP code and the PAT prompt list. The research manuscript, this editable deck and the script are preparation materials; they do not certify that the author’s final video or course submission has been completed.','https://github.com/karolpiczak/ESC-50 ; https://arxiv.org/abs/2211.06687v4 ; https://arxiv.org/abs/1711.10282 ; https://aclanthology.org/2025.naacl-long.616/ ; https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=4206 ; https://aclanthology.org/2026.findings-acl.913/ ; DEMAND https://doi.org/10.5281/zenodo.1227121');
}
fs.writeFileSync(path.join(dest,'speaker_notes.txt'),notes.map((t,i)=>`SLIDE ${i+1}\n${t}`).join('\n\n'));
fs.writeFileSync(path.join(dest,'narration_segments.json'),JSON.stringify(segments,null,2));
const digest = file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
p.writeFile({fileName:path.join(dest,'presentation.pptx')}).then(async()=>{
 // Remove generator-only content-type declarations for masters that do not exist.
 // Media actions do not need an empty relationship ID; keep the action and audio links.
 const file=path.join(dest,'presentation.pptx');
 const zip=await JSZip.loadAsync(fs.readFileSync(file));
 const types=await zip.file('[Content_Types].xml').async('string');
 zip.file('[Content_Types].xml',types.replace(/<Override\b[^>]*PartName="([^"]+)"[^>]*\/>/g,(tag,name)=>zip.file(name.replace(/^\//,''))?tag:''));
 for(const name of Object.keys(zip.files).filter(n=>/^ppt\/slides\/slide\d+\.xml$/.test(n))){
  const xml=await zip.file(name).async('string');zip.file(name,xml.replace(/ r:id=""/g,''));
 }
 // Every category array in this deck is one-dimensional. Use its equivalent
 // single-level reference cache while retaining the original workbook formula.
 for(const name of Object.keys(zip.files).filter(n=>/^ppt\/charts\/chart\d+\.xml$/.test(n))){
  const xml=await zip.file(name).async('string');
  zip.file(name,xml.replace(/<c:multiLvlStrRef>([\s\S]*?)<\/c:multiLvlStrRef>/g,(tag,body)=>{
   if((body.match(/<c:lvl>/g)||[]).length!==1)throw new Error('Unexpected hierarchical chart category');
   return '<c:strRef>'+body.replaceAll('multiLvlStrCache','strCache').replace(/<\/?c:lvl>/g,'')+'</c:strRef>';
  }));
 }
 fs.writeFileSync(file,await zip.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
 const library=JSON.parse(fs.readFileSync(path.join(path.dirname(require.resolve('pptxgenjs')),'../package.json'),'utf8'));
 fs.writeFileSync(path.join(dest,'build_status.json'),JSON.stringify({slides:count,concise_report_pages:11,real_code_demo:true,closing_analyses_complete:true,editable_charts:true,embedded_audio_clips:2,augmentation_complete:d.augmentation_complete,cost_panel_complete:d.cost_panel_complete,draft:false,pptxgenjs:library.version,node:process.version,source_sha256:digest(__filename),data_sha256:digest(path.join(dest,'presentation_data.json')),pptx_sha256:digest(path.join(dest,'presentation.pptx'))},null,2));
}).catch(error=>{console.error(error);process.exitCode=1;});

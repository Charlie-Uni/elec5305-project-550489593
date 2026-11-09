/* Editable, native-chart research deck. Result values come from audited tables. */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const PptxGenJS = require('pptxgenjs');
const root = path.resolve(__dirname, '..');
const dest = path.join(root, 'results/presentation_20261109_draft');
const d = JSON.parse(fs.readFileSync(path.join(dest, 'presentation_data.json'), 'utf8'));
const p = new PptxGenJS();
p.layout = 'LAYOUT_WIDE';
p.author = 'Qihan Zhu';
p.subject = 'ELEC5305 research presentation draft; verified results with explicit limitations';
p.title = 'When Background Noise Has Meaning';
p.company = 'ELEC5305 · University of Sydney';
p.lang = 'en-AU';
p.theme = {headFontFace:'Arial', bodyFontFace:'Arial', lang:'en-AU'};
const C = {ink:'172A2D',muted:'536366',pale:'F0F5F4',white:'FFFFFF',svm:'0072B2',cnn:'D55E00',clap:'009E73',grid:'DCE4E3',grey:'BAC5C5'};
let count = 0;
const notes = [];
function text(s,t,x,y,w,h,size=20,opts={}) {s.addText(t,{x,y,w,h,fontFace:'Arial',fontSize:size,color:C.ink,margin:0,breakLine:false,...opts});}
function rect(s,x,y,w,h,fill,line=fill) {s.addShape(p.ShapeType.rect,{x,y,w,h,fill:{color:fill},line:{color:line,width:.5}});}
function base(title,subtitle='',dark=false) {
 const s=p.addSlide();s.background={color:dark?C.ink:C.white};count++;
 text(s,String(count).padStart(2,'0'),12.0,6.92,.65,.24,11,{color:dark?'B8C9C7':C.muted,align:'right'});
 text(s,'ELEC5305  /  QIHAN ZHU  /  RESEARCH DRAFT',.6,6.92,10,.24,10,{color:dark?'B8C9C7':C.muted,charSpacing:1});
 if(title)text(s,title,.6,.48,12.1,.9,36,{bold:true,color:dark?C.white:C.ink});
 if(subtitle)text(s,subtitle,.62,1.43,12.0,.53,17,{color:dark?'C9D9D6':C.muted});
 return s;
}
function note(s,t,source='') {const n=t+(source?'\n\nEvidence: '+source:'');s.addNotes(n);notes.push(n);}
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
 for(let lane=0;lane<2;lane++)for(let i=0;i<62;i++){
  const x=.8+i*.19, a=.10+.35*Math.abs(Math.sin(i*.39+lane*2))*Math.exp(-Math.pow((i-(lane?37:25))/23,2));
  s.addShape(p.ShapeType.line,{x,y:5.28+lane*.44-a,w:0,h:2*a,line:{color:lane?'90BCAD':C.clap,width:2}});
 }
 text(s,'Qihan Zhu  ·  SID 550489593',.74,6.05,9,.3,17,{color:C.white});
 note(s,'A classifier may hear a target sound and a second meaningful event at the same time. My project asks how three different audio representations fail under that competition, compared with ordinary noise. The contribution is controlled signal processing and failure analysis. It is not a contest that gives all models the same training history. This presentation is a research draft prepared with AI assistance for author review.','Project report, research question and scope.');
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
 chart(s,['svm','cnn','clap'].map(system=>({name:system==='clap'?'CLAP · single prompt':system.toUpperCase(),labels:conditionNames,values:conditions.map(c=>core(system,c))})),.65,2.0,12.0,4.15);
 foot(s,'The report gives fold SD, all three levels, retained-target comparisons and the complete target-activity counterpart.');
 note(s,'At the same nominal relative level, the patterns depend strongly on representation. White noise almost destroys the handcrafted SVM’s macro-F1. CLAP is comparatively strong on white noise and DEMAND, but drops much further on meaningful event mixtures. The compact CNN lies between these behaviours. These results reject a universal ordering in which meaningful interference is always worse for every model. The report retains both power conventions and all levels rather than selecting a favourable condition.','all_conditions_verified.csv, own +5 dB.');
}
// 6. Replication without pseudo-replication.
{
 const s=base('The CNN overlap contrast replicates across seeds','Semantic minus matched DEMAND accuracy at +5 dB, overlap ≥ 0.8');
 const rs=d.replication.filter(r=>r.region==='own');
 chart(s,[5305,5306,5307].map((seed,i)=>({name:'Seed '+seed,labels:['Same category','Different category'],values:['semantic_same','semantic_different'].map(c=>100*Number(rs.find(r=>Number(r.seed)===seed&&r.condition===c).accuracy_difference))})),.65,2.05,7.2,4.05,{title:'Accuracy difference (percentage points)',valAxisMinVal:-10,valAxisMaxVal:0,valAxisMajorUnit:2,dataLabelPosition:'outEnd',chartColors:['F7BC9C',C.cnn,'8D3E16']});
 label(s,'Direction repeats','All three seeds favour DEMAND\non these matched targets.',8.3,2.35,4.2,C.cnn);
 text(s,'The two new seeds pass the declared\nHolm family. Seed 5305 does not.',8.3,4.55,4.3,.9,20);
 foot(s,'Report: per-seed paired 95% intervals and all 12 corrected tests. Repeated predictions of one target are not independent samples.');
 note(s,'Low-overlap semantic pairs can appear easier because the interferer falls mainly in the target’s silence. On high-overlap pairs, I compare semantic interference and DEMAND on the same target clips. All three clean-trained CNN seeds show a negative semantic-minus-DEMAND accuracy contrast. The two additional seeds produce larger differences and pass the declared correction family. The original seed remains inconclusive after correction. Seed variability and clip uncertainty are reported separately; three runs do not create three independent copies of the dataset.','clean_replication/cnn_overlap_replication.csv; own panel shown, target also verified.');
}
// 7. Frozen prompt ensemble and error composition.
{
 const s=base('Better class readout still leaves distractor competition','Frozen audio encoder · all 300 released prompt entries · no test-label prompt selection');
 text(s,'81.75% → 92.90%',.7,2.0,6.9,.8,42,{bold:true,color:C.clap});
 text(s,'Clean accuracy: single prompt → fixed ensemble',.75,2.88,10.9,.4,20);
 const cap=(v,c)=>d.captures.find(r=>r.variant===v&&r.region==='own'&&r.condition===c&&Number(r.level_db)===5&&r.stratum==='all');
 const items=[['clap_baseline','semantic_same'],['clap_prompt_ensemble','semantic_same'],['clap_baseline','semantic_different'],['clap_prompt_ensemble','semantic_different']].map(([v,c])=>cap(v,c));
 chart(s,[{name:'Distractor label',labels:['Same / single','Same / ensemble','Different / single','Different / ensemble'],values:items.map(r=>Number(r.distractor_captures))},{name:'Other wrong label',labels:['Same / single','Same / ensemble','Different / single','Different / ensemble'],values:items.map(r=>Number(r.other_wrong_labels))}],.7,3.42,11.95,2.8,{title:'Wrong predictions out of 2,000 trials at +5 dB',titleFontSize:15,barGrouping:'stacked',dataLabelPosition:'ctr',dataLabelFormatCode:'0',valAxisMinVal:0,valAxisMaxVal:1100,valAxisMajorUnit:250,chartColors:[C.clap,'CFDAD7'],catAxisLabelFontSize:12});
 foot(s,'This is a prompt-ensemble ablation, not full PAT. Deduplicating the 275 unique templates changes F1 by at most 0.47 pp.');
 note(s,'A fixed ensemble of the complete released PAT prompt list raises CLAP clean accuracy from 81.75 to 92.90 percent, using exactly the same saved audio embeddings. This is not full PAT and does not select prompts on test labels. Under semantic interference, other wrong labels become less common, but the distractor-capture counts are slightly higher in these own-protocol examples. Consequently, a higher fraction of remaining errors can name the distractor even while overall accuracy improves. I report both denominators and distinguish newly induced captures from baseline confusion.','Prompt audit; capture_transitions_core; prompt_multiplicity sensitivity.');
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
 note(s,d.augmentation_complete?'Across all three CNN seeds, WGN augmentation raises white-noise macro-F1 from 7.48 to 43.95 percent, but same-category interference changes only from 33.00 to 33.46 percent. Semantic training reaches 36.05 percent for same-category and 33.73 for different-category interference, while clean macro-F1 falls from 65.78 to 61.70 percent. These are measured trade-offs. Direct semantic-versus-WGN paired accuracy advantages for the CNN do not pass the common 156-test Holm correction, so modest positive point estimates are not an established superiority claim. All seeds are retained. Augmentation probability and maximum epoch budget are shared; actual refit durations are selected separately by clean validation.':'This is a preparation page. The intervention results are not yet complete and must be inserted only after independent verification. Do not narrate expected improvements as measured results.','Extension protocol and independently audited intervention tables.');
}
// 9. Interpretable feature movement.
{
 const s=base('The SVM weakness is visible in its DSP features','Standardised within-model displacement from the gain-matched clean control');
 const groups=['zcr','spectral_centroid','spectral_spread','spectral_rolloff'];
 chart(s,['white','demand','semantic_same'].map((condition,i)=>({name:['White noise','DEMAND','Same-category event'][i],labels:['ZCR','Centroid','Spread','Roll-off'],values:groups.map(g=>Number(d.feature_groups.find(r=>r.region==='own'&&r.condition===condition&&Number(r.level_db)===5&&r.feature_group===g&&r.outcome==='all').median))})),.65,2.0,8.1,4.1,{title:'Median standardised displacement',valAxisMaxVal:5,valAxisMajorUnit:1,chartColors:['003B5C',C.svm,'96C5DD'],dataLabelFormatCode:'0.00'});
 label(s,'White noise shifts\nspectral shape','Movement alone is insufficient:\nits direction relative to the\nclassifier also matters.',9.1,2.3,3.65,C.svm);
 foot(s,'Training-fold scales; each group divided by √dimension. CNN and CLAP cosine distances are interpreted in separate spaces.');
 note(s,'The handcrafted representation makes one failure mode interpretable. At plus five decibels, white noise shifts zero-crossing rate and spectral-shape features much more than the other conditions. Distances are standardised using training-fold scales. The CNN presents a complementary pattern: high-overlap semantic mixtures can yield smaller embedding movement than DEMAND but worse accuracy. Magnitude is therefore not a complete explanation. I do not compare absolute distances across the SVM, CNN and CLAP spaces, and I do not claim this isolates the causal effect of meaning.','diagnostics/feature_group_displacement.csv; representation analysis in report.');
}
// 10. Real recorded failure example, embedded audio.
{
 const s=base('Listen to a decision being captured','Can opening + clock alarm · +5 dB SIR · full target-activity overlap');
 const ex=d.listening;
 [['Gain-matched clean control','clean_matched',.8],['Target + weaker clock alarm','mixture',7.0]].forEach(([label,key,x])=>{
  rect(s,x,2.2,5.5,1.65,C.pale);text(s,label,x+.25,2.46,4.95,.65,23,{bold:true});
  s.addMedia({type:'audio',path:path.join(root,'examples/listening_own_20261109',ex.audio[key].file),x:x+.25,y:3.22,w:.6,h:.4});
  text(s,'Play 5-second audio',x+1.0,3.25,4,.3,15,{color:C.muted,hyperlink:{url:'https://raw.githubusercontent.com/Charlie-Uni/elec5305-project-550489593/feature/interference/examples/listening_own_20261109/'+ex.audio[key].file}});
 });
 ['svm','cnn','clap'].forEach((system,i)=>{const x=.8+i*4.17;const pp=ex.predictions[system];text(s,system.toUpperCase(),x,4.45,3.85,.4,22,{bold:true,color:C[system]});text(s,d.labels[String(pp.control_prediction)].replaceAll('_',' ')+'\n↓\n'+d.labels[String(pp.prediction)].replaceAll('_',' '),x,5.0,3.85,1.1,20);});
 foot(s,'PPTX: embedded audio; PDF: linked audio. Illustrative case using core CNN seed 5305 and single-prompt CLAP; provenance in notes.');
 note(s,'First play the gain-matched clean control, then the mixture, without speaking over either clip. All three core controls classify the target as can opening. With a weaker clock-alarm distractor, the CNN and single-prompt CLAP classify the clock alarm; the SVM produces another wrong class. This is an illustration selected from observed results, not proof that every can-opening and clock-alarm pair behaves this way. Its ratio, overlap, source recordings and audio-export hashes are recorded in the listening manifest.','examples/listening_own_20261109/examples.json, row '+ex.row+'. Can Open.WAV by LG, CC BY: https://freesound.org/people/LG/sounds/101404/ . alarm clock.aif by hookhead, CC0: https://freesound.org/people/hookhead/sounds/13613/ . ESC-50 compilation by Karol J. Piczak, CC BY-NC 3.0: https://creativecommons.org/licenses/by-nc/3.0/ . Changes: ESC-50 cropping, controlled mixing, common attenuation and PCM16 export.');
}
// 11. CPU deployment cost; distinguish microbenchmark from panel.
{
 const s=base('Cost ordering survives timing drift','Same machine · CPU · batch size one · preprocessing and class scoring included');
 const names=['svm_clean','cnn_clean','clap_baseline'];
 const times=d.cost_panel_complete?names.map(n=>Number(d.cost_panel.find(r=>r.variant===n).total_median_ms)):d.core_cost_stage_median_sum;
 const timingData=[{name:'First pass',labels:['Handcrafted SVM','Compact CNN','Frozen CLAP'],values:times}];
 if(d.cost_panel_complete)timingData.push({name:'Reversed order',labels:['Handcrafted SVM','Compact CNN','Frozen CLAP'],values:names.map(n=>Number(d.cost_panel_reverse.find(r=>r.variant===n).total_median_ms))});
 chart(s,timingData,.65,2.1,8.0,4.1,{title:d.cost_panel_complete?'Median CPU query time (ms)':'Sum of stage medians (ms)',valAxisMaxVal:Math.ceil(Math.max(...times)/25)*25,valAxisMajorUnit:25,chartColors:[C.ink,'A5B4B3'],showLegend:d.cost_panel_complete,legendFontSize:12});
 label(s,'36,258','CNN parameters',9.1,2.25,3.7,C.cnn);label(s,'158.3 million','CLAP parameters',9.1,4.2,3.7,C.clap);
 foot(s,d.cost_panel_complete?'250 clips × three repetitions per pass. The reversed-order pass checks observed drift; both are retained. Report includes IQR and p95.':'Core single-clip microbenchmark, 100 repetitions after warm-up. The balanced 250-clip extension is pending.');
 note(s,'This measures local query cost, not lifetime model-training cost. After observing timing differences for CNNs with identical preprocessing, I repeated the seven-model benchmark in reverse order. Model, audio and software hashes are unchanged, and all timed predictions match the clean evaluation. Both passes are retained. Absolute times change, but CNN, SVM and CLAP retain the same speed ordering. This does not establish a thermal cause or an augmentation-induced change in operation count. The measurement includes frontend and class scoring and excludes loading, disk access and one-off prompt preparation.','Audited cost_panel, cost_panel_reverse and cost_repeatability.');
}
// 12. Takeaways and explicit limits.
{
 const s=base('Power is only part of the story.','Controlled mixtures reveal where the classification decision moves.',true);
 [['01','Representation matters','White noise and meaningful events\nproduce different failure patterns.'],['02','Controls change interpretation','Activity masks, temporal overlap and\nmatched clean controls must be explicit.'],['03','Robustness needs diagnosis','Report error destinations, transfer\nand uncertainty alongside accuracy.']].forEach(([n,a,b],i)=>{const y=2.15+i*1.27;text(s,n,.75,y,.7,.55,31,{bold:true,color:'73BAA1'});text(s,a,1.75,y,10.8,.5,26,{bold:true,color:C.white});text(s,b,1.75,y+.55,10.8,.7,19,{color:'CCDCD8'});});
 text(s,'Limits: three backgrounds · one pair per category type · pretraining overlap · 0-dB label ambiguity',.75,6.2,11.8,.4,13,{color:'CCDCD8'});
 note(s,'Interference power alone does not explain these failures. Representation type, overlap and the competing concept all affect what the classifier predicts. The project’s strength is the controlled DSP protocol and the ability to trace a wrong decision to a meaningful distractor. Its limits remain important: only three DEMAND backgrounds, one pair of each type per target, some official source reuse, possible CLAP pretraining overlap and ambiguity at equal power. The results support measured behavioural differences, not an isolated causal claim that semantics alone determines robustness.','Research report conclusions and limitations.');
}
// 13. Sources and reproducibility.
{
 const s=base('Evidence, code and reproducible materials','Editable figures · numerical tables · manifests · model hashes · attributed listening examples');
 const refs=[['ESC-50 / Piczak','Official folds, dataset caveats and classic spectrogram CNN'],['LAION-CLAP / Wu et al.','Frozen audio-language reference, ICASSP 2023'],['PAT / Seth et al.','Released prompt-list ablation; full PAT is not reproduced'],['Mix-CLAP / Kohno et al.','Noise-aware audio-language model context, ICASSP 2026'],['RSA-Bench / Zhang et al.','Structured-background speech/reasoning context, ACL Findings 2026']];
 refs.forEach(([a,b],i)=>{text(s,a,.75,2.0+i*.68,4.15,.45,20,{bold:true});text(s,b,5.1,2.0+i*.68,7.45,.45,18);});
 text(s,'github.com/Charlie-Uni/elec5305-project-550489593',.75,5.42,11.85,.5,22,{color:C.clap,hyperlink:{url:'https://github.com/Charlie-Uni/elec5305-project-550489593/tree/feature/interference'}});
 text(s,'Audio: LG, Can Open.WAV (CC BY); hookhead, alarm clock.aif (CC0). ESC-50 compilation: CC BY-NC 3.0.',.75,6.03,11.85,.22,11,{color:C.muted});
 foot(s,'Draft for author review. AI assistance was used for implementation, checking and preparation; confirm course disclosure and video format.');
 note(s,'Full references, licences, code ownership and AI-assistance disclosure are in the report and repository. The project distinguishes original DSP and evaluation code from external CLAP code and the PAT prompt list. The research manuscript, this editable deck and the script are preparation materials; they do not certify that the author’s final video or course submission has been completed.','https://github.com/karolpiczak/ESC-50 ; https://arxiv.org/abs/2211.06687 ; https://aclanthology.org/2025.naacl-long.616/ ; https://www.cmsworkshops.com/ICASSP2026/view_paper.php?PaperNum=4206 ; https://aclanthology.org/2026.findings-acl.913/ ; DEMAND https://doi.org/10.5281/zenodo.1227121');
}
fs.writeFileSync(path.join(dest,'speaker_notes.txt'),notes.map((t,i)=>`SLIDE ${i+1}\n${t}`).join('\n\n'));
const digest = file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
p.writeFile({fileName:path.join(dest,'research_presentation.pptx')}).then(()=>{
 const library=JSON.parse(fs.readFileSync(path.join(path.dirname(require.resolve('pptxgenjs')),'../package.json'),'utf8'));
 fs.writeFileSync(path.join(dest,'build_status.json'),JSON.stringify({slides:count,editable_charts:true,embedded_audio_clips:2,augmentation_complete:d.augmentation_complete,cost_panel_complete:d.cost_panel_complete,draft:true,pptxgenjs:library.version,node:process.version,source_sha256:digest(__filename),data_sha256:digest(path.join(dest,'presentation_data.json')),pptx_sha256:digest(path.join(dest,'research_presentation.pptx'))},null,2));
}).catch(error=>{console.error(error);process.exitCode=1;});

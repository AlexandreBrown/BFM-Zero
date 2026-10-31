const sourceLinks = () => {
  document.querySelectorAll('[data-code]').forEach(a => {
    const ref = a.dataset.code, pos = ref.indexOf(':'), path = pos < 0 ? ref : ref.slice(0,pos), symbol = pos < 0 ? '' : ref.slice(pos+1);
    const line = window.BFM_SOURCE_INDEX?.symbols?.[`${path}:${symbol}`] || 1;
    a.href = `source.html#${encodeURIComponent(path)}:L${line}`;
    a.title = `${path}:${line}`;
  });
};
if (window.BFM_SOURCE_INDEX) document.getElementById('source-stamp').textContent = `Source snapshot · ${window.BFM_SOURCE_INDEX.commit.slice(0,10)} · ${window.BFM_SOURCE_INDEX.built}`;
const steps = [
 ['ROLLOUT CONTEXT','Choose a latent for each environment.','Refresh prompts on reset or at the configured interval. Some environments follow motion-derived prompt sequences; others draw from the mixed latent buffer or a random sphere.','z = agent.maybe_update_rollout_context(z, step_count, replay_buffer)\n# z: [1024, 256]','humanoidverse/agents/fb/agent.py:FBAgent.maybe_update_rollout_context'],
 ['ENVIRONMENT INTERACTION','Act from proprioception and history.','Before num_seed_steps the environment action space supplies random actions. Afterward the latent-conditioned actor samples an action with exploration noise. The simulator advances four physics steps per control step.','action = agent.act(obs=obs, z=context, mean=False)\nnext_obs, reward, terminated, truncated, info = train_env.step(action)\n# action: [1024, 29] · control dt: 0.02 s','humanoidverse/train.py:Workspace.train_online'],
 ['PHYSICAL SUCCESSOR','Store what happened before any reset.','make_transition replaces autoreset observations with their physical final observations for replay. Store the action, behavior z, raw auxiliary rewards, qpos/qvel and explicit successor. The trajectory buffer keeps episode boundaries.','data = make_transition(...)\ndata["z"] = context\nreplay_buffer["train"].extend(data)\n# online storage layout: [time, env, feature]','humanoidverse/train.py:make_transition'],
 ['OFF-POLICY MINIBATCHES','Sample online transitions and expert sequences.','Each agent update samples 1024 online transitions and 128 expert sequences of length 8, flattened to 1024 expert frames. Normalize using online observation statistics. Expert B embeddings are averaged and detached to form expert prompts.','expert_batch = expert_slicer.sample(batch_size)\ntrain_batch = train_buffer.sample(batch_size)\nexpert_z = encode_expert(expert_next_obs)\n# 128 × 8 frames → 1024 expert state/prompt pairs','humanoidverse/agents/fb_cpr_aux/agent.py:FBcprAuxAgent.update'],
 ['FIVE OPTIMIZER STAGES','Update D, F/B, QD, QR, then the actor.','D first sees expert prompts and original rollout prompts. The update then constructs mixed z values and relabels 80% of online samples. It updates representations, style value, auxiliary value and actor in that order. Finally, softly update target networks.','update_discriminator(...)\nz = sample_mixed_z(...)  # relabel online samples\nupdate_fb(...); update_critic(...); update_aux_critic(...)\nupdate_actor(...)\n# repeat this agent update 16 times per reference vector step','humanoidverse/agents/fb_cpr_aux/agent.py:FBcprAuxAgent.update'],
 ['EVALUATION & CHECKPOINTS','Measure behavior, save state, collect again.','Configured intervals count aggregate transitions. Evaluation shares the simulator with training and updates motion priorities, then training resets. Checkpoint publication writes a staged complete agent/replay and replaces the previous checkpoint. Final completed steps are saved.','if eval_time_checker.check(t):\n    eval_metrics = self.eval(t, replay_buffer)\n    # update motion priorities, then reset training env\nself.save(completed_steps, replay_buffer)','humanoidverse/train.py:Workspace.train_online']
];
let currentStep = 0;
function showStep(index) {
  currentStep = index;
  document.querySelectorAll('[data-step]').forEach(b => {const selected=+b.dataset.step===index;b.classList.toggle('active',selected);b.setAttribute('aria-pressed',String(selected));});
  const s=steps[index];
  ['step-kicker','step-title','step-text','step-code'].forEach((id,i)=>document.getElementById(id).textContent=s[i]);
  document.getElementById('step-link').dataset.code=s[4];
  document.querySelectorAll('[data-step-notes]').forEach(panel=>panel.hidden=+panel.dataset.stepNotes!==index);
  updateNotesButton(); sourceLinks();
}
document.querySelectorAll('[data-step]').forEach(button=>button.addEventListener('click',()=>showStep(+button.dataset.step)));
function selectLoss(button){
  document.querySelectorAll('[data-loss]').forEach(b=>{const selected=b===button;b.setAttribute('aria-selected',String(selected));b.tabIndex=selected?0:-1;document.getElementById(`loss-${b.dataset.loss}`).hidden=!selected;});
}
document.querySelectorAll('[data-loss]').forEach(button=>{
  button.tabIndex=button.getAttribute('aria-selected')==='true'?0:-1;
  button.addEventListener('click',()=>selectLoss(button));
  button.addEventListener('keydown',event=>{if(!['ArrowRight','ArrowLeft','Home','End'].includes(event.key))return;event.preventDefault();const all=[...document.querySelectorAll('[data-loss]')];let n=all.indexOf(button)+(event.key==='ArrowLeft'?-1:1);if(event.key==='Home')n=0;if(event.key==='End')n=all.length-1;const next=all[(n+all.length)%all.length];selectLoss(next);next.focus();});
});

// Native details elements keep the main reading path short and work with keyboards.
const escapeCode = value => value.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function lessonElement(note) {
  const details=document.createElement('details'); details.className='lesson'; details.id=note.id;
  if(note.trace){
    details.innerHTML=`<summary><span class="lesson-summary">${note.title}</span><span class="lesson-count">interactive</span></summary><div class="lesson-body"><p>Use four environments and the real 100-step / 250-step branch order. The sample IDs below stand in for whole 256-number vectors. Group choices are fixed for this illustration; no policy is being simulated.</p><div class="trace-controls" role="group" aria-label="Episode step in the four-environment example">${[0,1,99,100,249,250,300,500].map(t=>`<button type="button" data-trace-step="${t}" aria-pressed="${t===100}">${t}</button>`).join('')}</div><label class="trace-mix"><input type="checkbox" id="trace-mix" checked> Use saved z pool for general refresh</label><p class="trace-assumption">Clocks are synchronized; assume the task pool is populated by step 100. All other settings match the reference scheduling logic.</p><div class="trace-stages"><div><span>GENERAL REFRESH MASK</span><b id="trace-mask"></b></div><div><span>MOTION-SELECTED GROUP</span><b id="trace-group"></b></div></div><div class="table-wrap"><table><thead><tr><th>Environment</th><th>Final prompt ID</th><th>Why this row has that prompt</th></tr></thead><tbody id="trace-rows"></tbody></table></div><div class="trace-explanation" aria-live="polite" id="trace-explanation"></div><p class="trace-assumption">A/B are the first two motion sequences, C/D the next pair, and E/F the third pair. “A[100]” means the precomputed prompt at index 100, which already summarizes its short future window.</p></div>`;
    return details;
  }
  details.innerHTML=`<summary><span class="lesson-summary">${note.title}</span><span class="lesson-count">${note.blocks.length} blocks</span></summary><div class="lesson-body">${note.lead}<div class="annotated-blocks">${note.blocks.map((b,i)=>`<div class="code-block"><div class="block-heading"><span>${String(i+1).padStart(2,'0')}</span><h4>${b.title.replace(/^\d+\. /,'')}</h4></div>${b.code?`<pre><code>${escapeCode(b.code)}</code></pre>`:''}<div class="code-meaning">${b.explanation}</div></div>`).join('')}</div><details class="self-check"><summary>Check your understanding: ${note.check}</summary><p>${note.answer}</p></details><p class="lesson-source">Source excerpts, wrapped for reading; <code>...</code> marks omitted surrounding lines. <a data-code="${note.source}">View the complete function ↗</a></p></div>`;
  return details;
}
function appendLessons(host, notes){notes.forEach(note=>host.append(lessonElement(note)));}
const lessonData=window.BFM_LESSONS;
if(lessonData){
  lessonData.steps.forEach((notes,index)=>{const panel=document.createElement('div');panel.dataset.stepNotes=String(index);panel.hidden=index!==0;appendLessons(panel,notes);document.getElementById('step-lessons').append(panel);});
  for(const [key,notes] of Object.entries(lessonData.losses)){
    const host=document.createElement('div');host.className='loss-code-notes';appendLessons(host,notes);document.getElementById(`loss-${key}`).append(host);
  }
  appendLessons(document.getElementById('observation-lessons'),lessonData.observations);
  appendLessons(document.getElementById('inference-lessons'),lessonData.inference);
}
function activeStageNotes(){return [...document.querySelectorAll(`[data-step-notes="${currentStep}"] > details.lesson`)];}
function updateNotesButton(){
  const notes=activeStageNotes(),allOpen=notes.length>0&&notes.every(note=>note.open);
  document.getElementById('toggle-stage-notes').textContent=allOpen?'Close all notes':'Open all notes';
}
document.getElementById('toggle-stage-notes').addEventListener('click',()=>{
  const notes=activeStageNotes(),shouldOpen=notes.some(note=>!note.open);notes.forEach(note=>note.open=shouldOpen);updateNotesButton();
});
document.querySelectorAll('#step-lessons > div > details').forEach(note=>note.addEventListener('toggle',updateNotesButton));

function rolloutTrace(step, mix){
  let tasks=[], selected=[], letters=[];
  for(let t=0;t<=step;t++){
    if(t===0)tasks=Array.from({length:4},(_,e)=>`random@0/e${e}`);
    else if(t%100===0)tasks=Array.from({length:4},(_,e)=>`${mix?'pool':'random'}@${t}/e${e}`);
    if(t%250===0){const segment=Math.floor(t/250);selected=[[0,2],[1,3],[0,3]][segment%3];letters=[String.fromCharCode(65+segment*2),String.fromCharCode(66+segment*2)];}
    selected.forEach((env,i)=>tasks[env]=`${letters[i]}[${t%250}]`);
  }
  return {tasks,selected};
}
// Exposed for a small consistency check against the native scheduler.
window.BFM_ROLLOUT_TRACE=rolloutTrace;
let traceStep=100;
const traceText={
  0:'First call: everyone gets a random prompt. Environments 0 and 2 are then selected for motion sequences A and B, so their random prompts are overwritten.',
  1:'No general refresh. Environments 1 and 3 keep their initial prompts. The motion group advances to A[1] and B[1].',
  99:'The general mask is still False. The motion group keeps advancing each step; the other environments keep their initial prompts.',
  100:'General refresh fires for every row. The motion write then wins for environments 0 and 2. Only environments 1 and 3 actually use the newly sampled general prompts.',
  249:'The non-motion environments last refreshed at step 200. The motion group is using the final prompts of its current 250-step segments.',
  250:'A useful subtlety: 250 is not a multiple of 100, so no general refresh occurs. The new motion group is 1 and 3. Environments 0 and 2 keep A[249] and B[249] until a later task change; they are not automatically given random prompts when leaving the group.',
  300:'The general refresh replaces every candidate row. The current motion group (1 and 3) then overrides its candidates with C[50] and D[50]. Environments 0 and 2 now use fresh general prompts.',
  500:'Both boundaries occur. First refresh all general prompts, then choose the next motion group (0 and 3) and overwrite those rows with E[0] and F[0].'
};
function drawTrace(){
  const mix=document.getElementById('trace-mix')?.checked;if(mix===undefined)return;
  const state=rolloutTrace(traceStep,mix);
  document.getElementById('trace-mask').textContent=traceStep===0?'first-call branch':`[${Array(4).fill(traceStep%100===0?'True':'False').join(', ')}]`;
  document.getElementById('trace-group').textContent=`[${state.selected.join(', ')}]`;
  document.getElementById('trace-rows').innerHTML=state.tasks.map((task,env)=>{
    const motion=state.selected.includes(env),changed=traceStep===0||traceStep%100===0;
    const reason=motion?'Motion write overrides this row':changed?(traceStep===0?'Initial random assignment':mix?'New candidate from z_buffer':'Fresh random direction'):'Keep previous z';
    return `<tr class="${motion?'trace-motion':''}"><td>${env}</td><td><code>${task}</code></td><td>${reason}</td></tr>`;
  }).join('');
  document.getElementById('trace-explanation').textContent=traceText[traceStep];
  document.querySelectorAll('[data-trace-step]').forEach(b=>b.setAttribute('aria-pressed',String(+b.dataset.traceStep===traceStep)));
}
document.querySelectorAll('[data-trace-step]').forEach(b=>b.addEventListener('click',()=>{traceStep=+b.dataset.traceStep;drawTrace();}));
document.getElementById('trace-mix')?.addEventListener('change',drawTrace);drawTrace();

function openLessonHash(){
  const target=document.getElementById(location.hash.slice(1));if(!target?.classList.contains('lesson'))return;
  const stage=target.closest('[data-step-notes]');if(stage)showStep(+stage.dataset.stepNotes);
  const loss=target.closest('.loss-panel');if(loss)selectLoss(document.querySelector(`[aria-controls="${loss.id}"]`));
  target.open=true;requestAnimationFrame(()=>target.scrollIntoView({block:'start'}));
}
window.addEventListener('hashchange',openLessonHash);
sourceLinks();showStep(0);openLessonHash();
function calculate(){
  const [envs,every,updates,batch]=['envs','every','updates','batch'].map(id=>Number(document.getElementById(`calc-${id}`).value));
  const valid=[envs,every,updates,batch].every(n=>Number.isFinite(n)&&n>0&&Number.isInteger(n))&&batch>=2;
  const pretty=n=>Number(n.toPrecision(7)).toLocaleString('en-US',{maximumFractionDigits:7});
  document.getElementById('calc-utd').textContent=valid?pretty(updates/every):'—';
  document.getElementById('calc-reuse').textContent=valid?pretty(updates*batch/every):'—';
  document.getElementById('calc-vector').textContent=valid?pretty(updates*envs/every):'—';
  document.getElementById('calc-note').textContent=!valid?'Enter positive integer values.':every%envs?'The launcher requires update cadence to be a multiple of the environment count; this combination would be rejected.':`One update trigger every ${pretty(every/envs)} vector step(s). These are theoretical steady-state ratios after warm-up.`;
}
document.querySelectorAll('.calc-inputs input').forEach(input=>input.addEventListener('input',calculate));calculate();
const observer=new IntersectionObserver(entries=>{entries.forEach(entry=>{if(entry.isIntersecting)document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.hash===`#${entry.target.id}`));});},{rootMargin:'-10% 0px -70% 0px'});
document.querySelectorAll('main section[id]').forEach(section=>observer.observe(section));

/* Teaching notes for the local code. Excerpts are selected lines, not replacements. */
(() => {
  const block = (title, code, explanation) => ({ title, code, explanation });
  const lesson = (id, title, lead, blocks, check, answer, source) => ({ id, title, lead, blocks, check, answer, source });
  const fb = 'humanoidverse/agents/fb/agent.py:';
  const model = 'humanoidverse/agents/fb/model.py:';
  const cpr = 'humanoidverse/agents/fb_cpr/agent.py:';
  const aux = 'humanoidverse/agents/fb_cpr_aux/agent.py:';
  window.BFM_LESSONS = {
    steps: [
      [
        lesson('rollout-branches', 'Read maybe_update_rollout_context(): what each branch decides',
          '<p><b>Job:</b> return the task vector to use for each environment’s next action. It schedules tasks; it does not train a network.</p><p><code>z</code> is the previous batch of prompts, shaped <code>[N, 256]</code>, or <code>None</code> on the first call. <code>step_count</code> is <code>[N, 1]</code>: each environment’s own episode clock. In the trainer the returned value is called <code>context</code>.</p>', [
          block('1. Have we already assigned prompts?',
`if z is not None:
    ...
else:
    z = self._model.sample_z(step_count.shape[0], device=self._model.device)`,
            '<p>On the first call, there is no previous task to keep. The <code>else</code> branch gives every environment a random prompt. It can then replace some rows with motion prompts, as described below. On later calls, the first branch starts from the prompts already in use.</p><p>This test concerns the whole batch. An individual environment resetting later does not make the whole <code>z</code> tensor <code>None</code>; its zero episode clock handles that case.</p>'),
          block('2. Which environments are due for a general task change?',
`mask_reset_z = step_count % self.cfg.train.update_z_every_step == 0`,
            '<p>The default interval is <b>100 control steps</b>, or 2 seconds at 50 Hz. Remainder zero means “change task now”: episode steps 0, 100, 200, and so on. The result is one True/False value per environment.</p><p>For clocks <code>[99, 100, 0, 47]</code>, the mask is <code>[False, True, True, False]</code>. Environments 1 and 2 need replacements. Holding a task for several steps gives the robot time to act on it.</p>'),
          block('3. Where do replacement candidates come from?',
`if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
    new_z = self.z_buffer.sample(z.shape[0], device=self._model.device)
else:
    new_z = self._model.sample_z(z.shape[0], device=self._model.device)`,
            '<p>Both conditions must hold to use the saved task pool: mixed rollout is enabled, and learning has already put vectors in <code>z_buffer</code>. Otherwise, draw fresh random directions. This fallback lets collection begin before the first learning update.</p><p><code>z.shape[0]</code> is the number of environments. The code draws a candidate for <em>every</em> row, including rows that will keep their current prompt. The next line decides which candidates actually get used.</p>'),
          block('4. Keep old tasks until their clocks say to change',
`z = torch.where(mask_reset_z, new_z, z.to(self._model.device))`,
            '<p>Read this as: <b>“if this row is due, take its candidate; otherwise keep its previous vector.”</b> The <code>[N, 1]</code> mask broadcasts across all 256 coordinates, so a whole prompt changes together.</p><p>This is a selection, not an average of old and new prompts. Candidate vectors for False rows are discarded.</p>'),
          block('5. Is the motion-prompt override enabled?',
`if self.cfg.train.rollout_expert_trajectories:
    idxs = step_count % self.cfg.train.rollout_expert_trajectories_length`,
            '<p>This is a second schedule layered on top of the general task schedule. The default motion segment is 250 steps, or 5 seconds. <code>idxs</code> is each environment’s position within that segment: 0 through 249.</p><p>These environments will get a changing sequence of prompts derived from an expert motion. The actor still produces every action itself.</p>'),
          block('6. At a segment boundary, choose a group and prepare its sequences',
`if torch.any(idxs == 0):
    n_elem = int(
        self.cfg.train.rollout_expert_trajectories_percentage
        * step_count.shape[0]
    )
    self.env_idx_with_expert_rollout = torch.randperm(
        step_count.shape[0], device=self._model.device
    )[:n_elem]
    self.tracking_z = self._sample_tracking_z(
        replay_buffer, n_elem,
        self.cfg.train.rollout_expert_trajectories_length
    )`,
            '<p>At the default 50% and 1024 environments, select 512 distinct environment indices. <code>randperm</code> shuffles indices, so this group has no duplicates. Prepare one sequence per selected environment: <code>tracking_z</code> has shape <code>[512, 250, 256]</code>.</p><p><b>Read <code>torch.any</code> literally:</b> if any environment has segment index zero, the implementation redraws the entire selected group and its sequences. With synchronized episode clocks, this happens together. With different clocks, this is a shared group refresh, not an independent refresh just for the environment at zero. A newly selected environment whose own index is 83 starts at frame 83 of its new sequence.</p>'),
          block('7. Override the selected group with the current motion prompts',
`mod_time = idxs[self.env_idx_with_expert_rollout].ravel()
z[self.env_idx_with_expert_rollout] = self.tracking_z[
    torch.arange(
        len(self.env_idx_with_expert_rollout),
        device=self._model.device
    ),
    mod_time
]`,
            '<p><code>mod_time</code> picks the current frame index for each selected environment. The advanced indexing pairs sequence row 0 with its frame, row 1 with its frame, and so on. One 256-number vector is copied into each selected environment’s <code>z</code> row.</p><p><b>This write happens last, so it wins.</b> At step 100, a selected environment can receive a saved-pool candidate in block 4, then immediately have it replaced by its motion prompt in this block. Selected environments follow the motion sequence every control step, not just every 100 steps.</p>'),
          block('8. The first-call branch also initializes this motion group',
`z[self.env_idx_with_expert_rollout] = self.tracking_z[:, 0]
return z`,
            '<p>The <code>z is None</code> branch creates the same kind of selected group and sequences, then takes frame zero for that first action. Finally, the function returns one prompt per environment. The trainer passes those prompts to the actor and records them with the resulting transitions.</p><p>There is no reward evaluation or “best task” search inside this function. The random group choice and the saved task pool steer what data the policy collects.</p>')
        ], 'At step 100, does a motion-selected environment necessarily use a vector sampled from z_buffer?',
          'No. The general refresh runs first, but the motion-sequence assignment runs afterward and overwrites that row. The last write determines the prompt used by the actor.', fb + 'FBAgent.maybe_update_rollout_context'),
        lesson('z-samplers', 'Why save z? Compare the two samplers and the three buffers',
          '<p><b>The main reason is task selection:</b> use recent training-task vectors to collect new experience. Reusing vectors also avoids encoding those tasks again during rollout, but this buffer is not just a speed optimization.</p>', [
          block('Fresh directions: sample_z()',
`z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
return self.project_z(z)`,
            '<p>Draw Gaussian noise, then normalize each vector to length √256 = 16. Its direction is random. No motion, goal, replay state, or neural-network encoder is consulted. This explores directions across the task space; there is no guarantee that a particular draw produces a useful skill.</p>'),
          block('Populate the pool during learning',
`z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
self.z_buffer.add(z)`,
            '<p>These lines are in <code>FBcprAuxAgent.update()</code>. The new batch contains, in expectation, 60% expert-sequence prompts, 20% online-goal prompts, and 20% random prompts. Saving this batch makes that training-task mixture available for later rollout choices.</p><p>The pool is not ranked by return, and vectors are not accepted only after a successful rollout. “Saved in z_buffer” means “recently constructed for training,” not “known to be a good behavior.”</p>'),
          block('Reuse stored directions: ZBuffer.sample()',
`idx = np.random.randint(0, len(self), size=num)
return self._storage[idx].clone().to(device if device is not None else self.device)`,
            '<p>Choose stored row indices uniformly <em>with replacement</em>, copy those vectors, and put them on the requested device. The same vector may be drawn more than once. There is no call to B here.</p><p>The default pool holds 8192 vectors and overwrites old rows as it fills. B changes during learning, so these vectors can come from slightly older encodings; they are not refreshed from their original states. The small rolling pool keeps them recent. “Saved” here means kept in memory: this pool itself is not part of the agent checkpoint.</p>'),
          block('Three containers, three different jobs', '',
            '<div class="table-wrap"><table><thead><tr><th>Container</th><th>What is stored?</th><th>Why?</th></tr></thead><tbody><tr><td><code>z_buffer</code></td><td>A small pool of task vectors</td><td>Choose future rollout tasks from recent training tasks.</td></tr><tr><td><code>replay_buffer["train"]</code></td><td>Online transitions, including the z used when collecting each one</td><td>Learn from physical experience and retain its task context.</td></tr><tr><td><code>replay_buffer["expert_slicer"]</code></td><td>Expert state trajectories</td><td>Draw motion sequences for the prior and for task encoding.</td></tr></tbody></table></div><p>Turning <code>use_mix_rollout</code> off changes the general rollout sampler to fresh random directions. It does not turn off expert-sequence rollouts or change <code>sample_mixed_z()</code> inside learning.</p>')
        ], 'Would replacing z_buffer.sample() with sample_z() only change execution speed?',
          'It changes which tasks are attempted during data collection. The saved pool includes motion- and goal-derived directions; fresh sample_z() draws have no such data-dependent selection.', 'humanoidverse/agents/misc/zbuffer.py:ZBuffer.sample'),
        lesson('tracking-prompts', 'Read _sample_tracking_z(): turn motion frames into a task sequence',
          '<p><b>Job:</b> prepare prompts for the motion-selected group. It reads expert states, encodes them with B, and returns a sequence of task vectors. It never supplies expert joint-control actions.</p>', [
          block('1. Handle an empty selected group',
`if batch_dim == 0:
    return torch.empty((0, traj_length, self.cfg.model.archi.z_dim), device=self._model.device)`,
            '<p>If the selected percentage gives zero environments, return an empty tensor with the expected shape. This avoids trying to sample a nonexistent motion batch.</p>'),
          block('2. Request whole sequences, then encode each frame',
`batch = replay_buffer["expert_slicer"].sample(batch_dim * traj_length, seq_length=traj_length)
next_obs = tree_map(lambda x: x.to(self._model.device), batch["next"]["observation"])
z = self._model.backward_map(next_obs)
z = z.view(batch_dim, traj_length, z.shape[-1])`,
            '<p>For 512 selected environments and 250 steps, request 128,000 frames arranged as 512 contiguous expert sequences. Replay can live on CPU; <code>tree_map</code> moves each observation tensor to the encoder’s device before B runs. This changes where the data lives, not the prompt calculation. B maps each successor state to a 256-number feature vector. <code>view</code> restores the sequence layout: <code>[512, 250, 256]</code>.</p><p><code>next</code> expresses the state the actor should try to reach next. The helper’s explicit <code>seq_length=250</code> controls this sample length; it is distinct from the model’s default 8-frame averaging window.</p>'),
          block('3. Use a short window of future frames',
`for step in range(traj_length):
    end_idx = min(step + self.cfg.model.seq_length, traj_length)
    z[:, step] = z[:, step:end_idx].mean(dim=1)`,
            '<p>At frame t, average B-vectors from t through t+7. This gives a short upcoming-motion target instead of only one pose. At the end of the segment, <code>min</code> shortens the window so it stays inside the sampled data.</p><p>Each selected environment receives a different vector as its segment clock advances. This is how a fixed actor can follow a moving reference.</p>'),
          block('4. Restore the common prompt length',
`return self._model.project_z(z)`,
            '<p>Averaging can shorten a vector. Projection restores the agreed task scale while retaining its direction. These 8-frame rollout prompts are separate from the released tracking evaluator’s one-frame prompts.</p>')
        ], 'Does the actor observe the full future motion sequence?',
          'The helper compresses the upcoming states into z for the current step. The actor gets that vector plus its own current observation/history, rather than the whole raw reference sequence.', fb + 'FBAgent._sample_tracking_z'),
        { id: 'rollout-example', title: 'Try it: watch four environments keep, replace, and override z', trace: true }
      ],
      [lesson('act-code', 'Read the action branch: task choice and action choice are separate',
        '<p>Choosing z answers “which task?” The next branch answers “which joint targets shall we send now?” During initial exploration, these two choices are deliberately disconnected.</p>', [
        block('1. During warm-up, collect random actions',
`if t < self.cfg.num_seed_steps:
    action = train_env.action_space.sample().astype(np.float32)`,
          '<p><code>t</code> counts transitions across all environments. Before 10,240 transitions in the reference recipe, actions come from the action space. A z has still been assigned, but these random warm-up actions do not follow its task. The replay needs initial experience before value-based learning can become useful.</p>'),
        block('2. After warm-up, let the actor follow z',
`action = self.agent.act(obs=obs, z=context, mean=False)`,
          '<p>Here <code>context</code> is the z batch returned by the prompt scheduler. For each environment the actor combines its measurements/history with that task vector and produces an action. Changing z can change the action even when the measured robot state is identical.</p><p><code>mean=False</code> requests a sampled action, adding exploration. Evaluation normally uses <code>mean=True</code> to use the distribution’s mean. The same policy weights support both.</p>'),
        block('3. Execute the action and observe its physical result',
`new_td, new_reward, new_terminated, new_truncated, new_info = train_env.step(action)`,
          '<p>The environment turns the 29 action values into PD targets, advances physics, and returns new measurements and boundary flags. <code>new_reward</code> is the environment’s scalar aggregate; the auxiliary agent later assembles its own penalty reward from raw terms in <code>new_info</code>.</p><p>The surrounding <code>torch.no_grad()</code> means collection does not build a gradient graph through the actor or simulator. Learning happens later from sampled replay.</p>')
      ], 'Does mean=False update the actor?', 'No. It changes action sampling. The actor’s optimizer updates its weights later in update_actor().', 'humanoidverse/train.py:Workspace.train_online')],
      [lesson('replay-code', 'Read make_transition(): save the experience and its task context',
        '<p>A replay row must answer: “from this observation, this action led to this physical successor.” The stored z also tells us which task the collecting policy was trying to perform.</p>', [
        block('1. Copy observations so later simulator writes cannot change history',
`return torch.as_tensor(value).detach().clone()`,
          '<p>The small <code>tensor()</code> helper detaches values from any gradient graph and copies their storage. Simulator observations are reused and mutated on later steps; replay needs a stable record of this step.</p>'),
        block('2. Replace reset observations with the actual final state',
`final_mask = next_info.get("_final_observation")
if final_mask is not None:
    ...`,
          '<p>An environment can finish an episode and reset inside <code>step()</code>. Its returned observation may then be the new episode’s start. For replay, the successor of the old action is still the final physical state. The masked replacement preserves that cause-and-effect pair. qpos/qvel get the same treatment.</p>'),
        block('3. Keep “stops the sequence” separate from “stops bootstrap”',
`data["done"] = data["next"]["terminated"] | data["next"]["truncated"]`,
          '<p>Either flag marks a sequence boundary. A true termination also means there is no future task value to bootstrap. A time limit merely stopped the rollout: the agent still estimates future value from its physical successor. This is why its discount uses <code>~terminated</code>.</p>'),
        block('4. Record the prompt used for collection',
`if context is not None:
    data["z"] = context`,
          '<p>This <code>z</code> belongs to <em>this transition</em>. The discriminator needs online state/task pairs that reflect the collecting policy. The learning update also keeps this original z for samples it does not relabel.</p><p>The helper does not copy this field itself; the replay buffer’s insertion converts/copies the data into its storage. This per-transition z has a different role from the standalone z-buffer’s pool of future task candidates.</p>'),
        block('5. Add one time slice for all parallel environments',
`data = tree_map(lambda value: value[None, ...], data)
replay_buffer["train"].extend(data)`,
          '<p>The trajectory buffer expects time and environment axes. <code>[None, ...]</code> turns a <code>[1024, feature]</code> batch into <code>[1, 1024, feature]</code>: one new time slice. Sampling later draws stored experience rather than taking another simulator step.</p>')
      ], 'Is a transition invalid for FB learning if we later assign it a different z?', 'Off-policy learning can use the same transition for another task: the physical transition remains valid. A task-dependent target or reward must be evaluated for the new z. The discriminator first uses the original collection pairing.', 'humanoidverse/train.py:make_transition')],
      [
        lesson('expert-encoding', 'Read encode_expert(): eight frames become one shared task',
          '<p>The expert dataset contains motion states. This helper asks: “what single task direction describes this short sequence?” It produces task targets for the discriminator and the task mixture.</p>', [
          block('1. Encode each frame with B',
`B_expert = self._model._backward_map(next_obs).detach()`,
            '<p>Each expert successor observation becomes a feature vector. <code>detach()</code> makes this encoding a fixed target for this update. Training D should not also move B to make D’s classification task easier.</p>'),
          block('2. Recover the sequence axis',
`B_expert = B_expert.view(
    self.cfg.train.batch_size // self.cfg.model.seq_length,
    self.cfg.model.seq_length,
    B_expert.shape[-1],
)`,
            '<p>A batch of 1024 expert frames is 128 sampled sequences of 8 frames. This reshape changes <code>[1024, 256]</code> into <code>[128, 8, 256]</code>. It relies on the expert sampler keeping each sequence’s frames together.</p>'),
          block('3. Average, normalize, and give all frames the same task',
`z_expert = B_expert.mean(dim=1)
z_expert = self._model.project_z(z_expert)
z_expert = torch.repeat_interleave(z_expert, self.cfg.model.seq_length, dim=0)`,
            '<p>Averaging gives 128 sequence descriptions. Projection fixes their scale. Repeating each 8 times restores <code>[1024, 256]</code>, so every expert frame can be paired with its sequence’s task in D’s batch.</p><p>This is different from <code>_sample_tracking_z()</code>: here each sampled 8-frame sequence has one shared prompt; that rollout helper builds a changing 250-step prompt sequence.</p>')
        ], 'Why repeat the sequence prompt instead of encoding every expert frame as its own task?', 'D should recognize states that belong to the same short behavior under one task direction. Repeating pairs each frame with that shared behavior description; it also matches the discriminator batch shape.', cpr + 'FBcprAgent.encode_expert'),
        lesson('normalization-code', 'Read the normalizer calls: use a common ruler for online and expert states',
          '<p>If the same physical quantity has different scales in the two data streams, D can learn that mismatch instead of a motion prior.</p>', [
          block('1. Update statistics using online observations',
`self._model._obs_normalizer(train_obs)
self._model._obs_normalizer(train_next_obs)`,
            '<p>These calls update running normalization statistics while the module is in training mode. Their returned normalized values are unused here; updating the statistics is the purpose of these first two calls.</p>'),
          block('2. Normalize both streams without updating statistics again',
`with torch.no_grad(), eval_mode(self._model._obs_normalizer):
    ...`,
            '<p>Inside this block, current online states, successors, and expert states all use the same current statistics. <code>eval_mode</code> freezes the statistic updates; <code>no_grad</code> avoids a gradient graph for this preprocessing.</p><p>The later underscored calls, such as <code>_backward_map</code>, receive already-normalized inputs. Public <code>backward_map()</code> is the inference wrapper that performs normalization for its caller.</p>')
        ], 'Why not call the public backward_map() on these already-normalized expert states?', 'The public wrapper normalizes its inputs. Doing so here would normalize the same values twice.', aux + 'FBcprAuxAgent.update')
      ],
      [
        lesson('update-pairing', 'Follow train_z through update(): why the order matters',
          '<p>The same variable name changes meaning midway through the update. Follow the pairing of a state with its task, rather than treating every z as interchangeable.</p>', [
          block('1. Read the collection task and train D on that pairing',
`train_z = train_batch["z"].to(self.device)
...
metrics = self.update_discriminator(
    expert_obs=expert_obs,
    expert_z=expert_z,
    train_obs=train_obs,
    train_z=train_z,
    grad_penalty=grad_penalty,
)`,
            '<p>The negative online examples are paired with the task used when collecting them. The positive expert examples are paired with their sequence encoding. At this point, the online prompts have not been relabeled.</p>'),
          block('2. Make alternate tasks, then share them with future rollouts',
`z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
self.z_buffer.add(z)`,
            '<p>Construct a new task batch from expert encodings, online goals, and random directions. Add it to the small rollout task pool. <code>clone()</code> gives the batch its own storage; it does not train the vectors or improve their quality.</p>'),
          block('3. Relabel most online examples for value/policy learning',
`if self.cfg.train.relabel_ratio is not None:
    mask = torch.rand(
        (self.cfg.train.batch_size, 1), device=self.device
    ) <= self.cfg.train.relabel_ratio
    train_z = torch.where(mask, z, train_z)`,
            '<p>With probability 0.8 per row, use the newly sampled task instead of the collection task. The action and physical successor do not change. The update now asks about this same experience under another task.</p><p><code>None</code> skips this step. A ratio of 1 relabels all rows; a ratio of 0 effectively keeps the original tasks. This is a probability per sample, not a promise of exactly 80% of every batch.</p>'),
          block('4. Learn values for those tasks, then improve the actor',
`self.update_fb(...)
self.update_critic(...)
self.update_aux_critic(...)
self.update_actor(...)`,
            '<p>These four calls all use the resulting <code>train_z</code>. F/B learn future features, QD learns future style reward, and QR learns future penalties. The actor then chooses actions those value estimates favor. Each call steps its own optimizer; the actor is not learning from a single summed critic-training loss.</p>')
        ], 'Does D train on the newly relabeled online prompts in this update?', 'No. D is updated before relabeling, with the original collection prompts. The subsequent value and actor updates use the partly relabeled task batch.', aux + 'FBcprAuxAgent.update'),
        lesson('mixed-z-code', 'Read sample_mixed_z(): construct the task distribution',
          '<p><b>Three ways to pose a task:</b> a random direction, an encoded online state, or an encoded expert sequence. The helper chooses one source per row.</p>', [
          block('1. Begin with random tasks and draw source labels',
`z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
...
mix_idxs = torch.multinomial(prob, num_samples=self.cfg.train.batch_size, replacement=True).reshape(-1, 1)`,
            '<p>The probability vector is ordered as goal, expert, random. With defaults it is <code>[0.2, 0.6, 0.2]</code>. A row labeled random keeps its initial random vector. The <code>[batch, 1]</code> labels later select whole task vectors.</p>'),
          block('2. Replace goal-labeled rows with encoded online states',
`perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
train_goal = tree_map(lambda x: x[perm], train_goal)
goals = self._model._backward_map(train_goal)
goals = self._model.project_z(goals)
z = torch.where(mix_idxs == 0, goals, z)`,
            '<p>Shuffle online successor states, encode them with B, and use those directions for goal-labeled rows. The shuffle breaks the automatic pairing between a transition and its own successor as the chosen goal. It does not create new robot states.</p>'),
          block('3. Replace expert-labeled rows with sequence encodings',
`perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
z = torch.where(mix_idxs == 1, expert_encodings[perm], z)`,
            '<p>Shuffle the expert-task vectors and select them for expert-labeled rows. Rows labeled random remain as they were. This helper and <code>encode_expert()</code> run without gradients: they choose training tasks, while <code>update_fb()</code> is where B’s optimizer learns its representation.</p>')
        ], 'Is 60/20/20 a mixture of state vectors or a mixture of task vectors?', 'Task vectors. Every result row has the same 256-number format, even though its source can be an expert sequence, an online state, or a random direction.', cpr + 'FBcprAgent.sample_mixed_z')
      ],
      [lesson('schedules-code', 'Read the outer schedules: collection, learning, and evaluation use different clocks',
        '<p>It helps to ask “which clock is this condition checking?” The episode clock in the prompt function and the total training budget are not the same counter.</p>', [
        block('1. Decide when learning may begin',
`if len(replay_buffer["train"]) > 0 and t > self.cfg.num_seed_steps and update_agent_time_checker.check(t):
    update_agent_time_checker.update_last_step(t)
    for _ in range(self.cfg.num_agent_updates):
        metrics = self.agent.update(replay_buffer, t)`,
            '<p>The replay must contain data, warm-up must be over, and the global update interval must have elapsed. Then perform 16 agent updates in the reference recipe. These updates reuse replay; they do not each advance physics.</p><p>The <code>update_last_step</code> call records that the scheduled work ran, preventing it from firing on every subsequent loop iteration.</p>'),
        block('2. Evaluate behavior and feed motion difficulty back into sampling',
`priorities = 2**priorities`,
            '<p>The tracking evaluation produces a per-motion EMD. The surrounding code clamps/scales it, then this exponential gives harder clips more weight. Both reset-motion sampling and expert-sequence sampling receive those priorities. Evaluation cadence therefore influences the data curriculum as well as chart frequency.</p><p>Evaluation shares the simulator, so training resets afterward. That reset refreshes each episode clock; it does not set the global training counter back to zero.</p>'),
        block('3. Save a completed training state',
`if completed_steps > self._checkpoint_time:
    self.save(completed_steps, replay_buffer)`,
            '<p>The final save uses the number of transitions actually completed. The checkpoint stores networks, optimizer state and optionally replay. Replay supports future reward relabeling as well as continued off-policy learning.</p><p>Full simulator state, rollout task schedules and complete RNG state are not saved here. A resumed job restores learning state, then starts a new physical rollout; it is not an exact replay of an uninterrupted run.</p>')
      ], 'Why can changing evaluation frequency change the learned policy?', 'Evaluation changes motion priorities and interrupts/resets the shared simulator. It affects which experience is collected, not just how often metrics are written.', 'humanoidverse/train.py:Workspace.train_online')]
    ],
    losses: {
      d: [lesson('disc-code', 'Read the D update: learn a state/task pairing test',
        '<p>D receives a state and a task vector. Its question is: “does this state look like an expert example of this task?” This conditioning is why the pairing of states with z matters.</p>', [
        block('Two batches, two labels',
`expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)`,
          '<p>The first term encourages a high expert logit; the second encourages a low online logit. These stable formulas compute binary classification losses without manually taking logs of probabilities near zero.</p>'),
        block('Regularize steep shortcuts, then update D',
`loss += grad_penalty * wgan_gp
...
self.discriminator_optimizer.zero_grad(set_to_none=True)
loss.backward()
self.discriminator_optimizer.step()`,
          '<p>The gradient penalty discourages an overly sharp classifier in joint state/task space. Clearing gradients prevents accumulation from the previous update; backward computes this loss’s parameter gradients; step applies them to D.</p><p>D’s output later becomes a reward signal. It does not directly supply a demonstrated action for the actor to copy.</p>')
      ], 'If D’s loss falls, must tracking have improved?', 'No. D may simply be getting better at distinguishing weak online behavior from expert states. Inspect behavior evaluation and both classification terms.', cpr + 'FBcprAgent.update_discriminator')],
      fb: [lesson('fb-matrix-code', 'Read the FB matrix: what one row, one column, and the diagonal mean',
        '<p>Use a batch of just three transitions to picture this. Row i is a starting state/action/task. Column j is a candidate future state, taken from transition j’s successor. Every row is compared with every column.</p>', [
        block('1. Make one future-state question per matrix entry',
`Fs = self._model._forward_map(obs, z, action)
B = self._model._backward_map(goal)
Ms = torch.matmul(Fs, B.T)`,
          '<p><code>Fs[k,i]</code> describes future features under task i, from recorded action i, for forward head k. <code>B[j]</code> describes candidate state j. Their dot product asks how strongly that future points toward candidate j. With two forward heads and three examples, <code>Ms</code> is <code>[2, 3, 3]</code>.</p><p>The argument named <code>goal</code> is the batch of successor observations here. It does not mean all these states are the task goals used to choose z.</p>'),
        block('2. Ask the same question one step later',
`next_action = self.sample_action_from_norm_obs(next_obs, z)
target_Fs = self._model._target_forward_map(next_obs, z, next_action)
target_B = self._model._target_backward_map(goal)`,
          '<p>For the target, move to the observed successor, keep the same z, and let the current actor choose a next action. Use slowly moving target networks for F/B. This whole target block has no gradient graph: it supplies a reference for today’s prediction.</p>'),
        block('3. Separate future propagation from the observed one-step successor',
`diff = Ms - discount * target_M
fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]`,
          '<p>Off-diagonal entries learn the discounted future relation. On the diagonal, column i is the actual immediate successor of row i: this is where the observed one-step connection enters as a linear attraction term.</p><p>For a fixed target, minimizing <code>−diff[i,i]</code> pushes the current immediate-successor score upward. Squaring every entry instead would remove this special immediate-successor teaching signal.</p>'),
        block('4. Keep backward directions informative, then move both networks',
`Cov = torch.matmul(B, B.T)
...
fb_loss += self.cfg.train.ortho_coef * orth_loss
...
self.forward_optimizer.step()
self.backward_optimizer.step()`,
          '<p><code>Cov[i,j]</code> is a dot product between two state embeddings. Penalizing large off-diagonal similarities discourages mapping every state to the same direction. This is a sample Gram matrix, despite the variable name <code>Cov</code>.</p><p>F and B are jointly trained by this objective. B is not a fixed encoder loaded before RL. Its normalized vector length also explains why a healthy norm alone does not prove a useful representation.</p>')
      ], 'In Ms[i,j], does j choose the task?', 'No. The task is z_i, attached to the row. Column j chooses the candidate future state whose occupancy is being scored.', fb + 'FBAgent.update_fb')],
      qd: [lesson('style-target-code', 'Read the style TD target: why a second critic is needed',
        '<p>D judges the current state/task pair. The actor needs to know whether an action will lead to good style over time. QD bridges that gap.</p>', [
        block('A current style score becomes a reward',
`reward = self._model._discriminator.compute_reward(obs=obs, z=z)`,
          '<p>The clamped discriminator probability is converted to log-odds. This is computed without gradients for the critic target. D is not updated by the QD regression loss.</p>'),
        block('Add the estimated style after the next action',
`target_Q = reward + discount * next_V
expanded_targets = target_Q.expand(num_parallel, -1, -1)`,
          '<p><code>next_V</code> combines target QD heads evaluated at the successor and a next action from the actor. The same one-step-plus-future target is fitted by both current QD heads. A true termination makes the discounted future term zero.</p><p>The actor later differentiates QD with respect to a newly chosen action. This is how a state-based style signal can guide action choice.</p>')
      ], 'Why not replace QD with D directly in the action objective?', 'D in this implementation takes the state and z, not the proposed action. QD learns how actions change future style and gives the actor an action-dependent learning signal.', cpr + 'FBcprAgent.update_critic')],
      qr: [lesson('penalty-code', 'Read auxiliary reward assembly: follow the raw term to the actor',
        '<p>There are two weighting layers in the repository. For this agent, follow <code>aux_rewards</code> from environment info into the agent’s own weighted sum.</p>', [
        block('Accumulate the configured raw penalties',
`for aux_reward_name in self.cfg.aux_rewards:
    metrics[f"aux_rew/{aux_reward_name}"] = train_batch["aux_rewards"][aux_reward_name].mean()
    aux_reward += self.cfg.aux_rewards_scaling[aux_reward_name] * train_batch["aux_rewards"][aux_reward_name].to(self.device)`,
          '<p>The metric is logged before the agent weight is applied. The weighted sum is what trains QR. A zero torque weight means that term contributes nothing to QR’s reward, even if the raw torque term is computed and logged.</p>'),
        block('Normalize the scale, then fit future penalty value',
`aux_reward = self._model._aux_reward_normalizer(aux_reward)
...
target_Q = aux_reward + discount * next_V`,
          '<p>The normalizer rescales the weighted reward by running variation, keeping its mean untranslated in this recipe. QR then predicts the sum of present and future normalized penalty rewards under z.</p><p>The actor maximizes this value. With negative penalty rewards, a larger value usually means a less negative predicted total: fewer future costs. The actor coefficient controls how strongly it trades that objective against FB and style value.</p>')
      ], 'If reg_coeff_aux is zero, does the auxiliary critic stop training?', 'No. QR still receives updates. The actor simply stops using its value term in its own loss.', aux + 'FBcprAuxAgent.update')],
      pi: [lesson('actor-gradient-code', 'Read the actor update: imagine trying a different action in the same state',
        '<p>The physical transition used to learn values came from replay. For the actor update, propose a fresh action and ask the learned values whether it is promising.</p>', [
        block('Replace the recorded action with the current actor’s proposal',
`dist = self._model._actor(obs, z, self._model.cfg.actor_std)
action = dist.sample(clip=self.cfg.train.stddev_clip)`,
          '<p>The argument called <code>action</code> is overwritten here. This loss does not fit the actor to the replayed action. The sampled proposal retains a gradient path back to the actor parameters through this distribution implementation.</p>'),
        block('Score the proposal under the same task',
`Fs = self._model._forward_map(obs, z, action)
Qs_fb = (Fs * z).sum(-1)`,
          '<p>The elementwise multiply and sum is the dot product Fᵀz. F predicts future features, and z selects which combination is valuable. QD and QR score that same proposal for future style and penalties. Twin heads are combined conservatively.</p>'),
        block('Change the actor, using those values as a differentiable guide',
`weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
...
self.actor_optimizer.zero_grad(set_to_none=True)
actor_loss.backward()
self.actor_optimizer.step()`,
          '<p>The negative signs turn value maximization into loss minimization. <code>detach()</code> makes the regularizer scale a fixed number for this update, rather than another route for actor gradients.</p><p>Gradients flow through the critics’ action inputs into the actor. Their optimizers are not stepped in this method. Physics is not differentiated: the critics supply the learned guide to how an action affects the future.</p>')
      ], 'Which measured action should the actor copy in this update?', 'There is no action-copying target in this loss. The actor changes its own proposal to increase the combined learned values.', aux + 'FBcprAuxAgent.update_actor')]
    },
    observations: [lesson('input-filters', 'Read the input filters: find what the deployed actor can actually use',
      '<p>Start with a dictionary of robot measurements. Different components read different keys from that dictionary.</p>', [
      block('Actor filter: measurements it can use at deployment',
`key=['state', 'last_action', 'history_actor']`,
        '<p>The actor receives current proprioception, its previous action, and a short history. History helps it infer things such as motion trends or changed dynamics from recent measurements. z supplies the task separately.</p>'),
      block('F and value critics: extra information while learning',
`key=['state', 'privileged_state', 'last_action', 'history_actor']`,
        '<p>In simulation, the critics can inspect full body motion and height. This richer view can make their predictions more accurate while the actor still learns to act from its smaller input. This is the asymmetry in “asymmetric training.”</p>'),
      block('B and D: a state description without action history',
`key=['state', 'privileged_state']`,
        '<p>Both online and expert data can provide these keys. Expert motions do not need to supply the control actions that originally produced them. The model uses B to describe desired states in task space, and D to compare state/task pairs.</p>')
    ], 'If privileged_state is present in replay, does the actor necessarily read it?', 'Only keys selected by the actor filter enter the actor. Replay can store more information than that one component uses.', 'humanoidverse/train.py:default_train_config')],
    inference: [lesson('inference-code', 'Read inference as three ways to construct the same actor input',
      '<p>The actor’s interface stays <code>π(observation history, z)</code>. Inference changes how the task vector is constructed; it does not fit new policy weights.</p>', [
      block('A goal: describe one target state',
`z = self.backward_map(next_obs)
return self.project_z(z)`,
        '<p>Here the caller supplies the desired goal observation. B turns it into a direction in the task space learned during training. A fixed goal gives a fixed z while the robot’s current observations continue changing.</p>'),
      block('A motion: move the target forward with time',
`z[step] = z[step:end_idx].mean(dim=0)`,
        '<p>Average encoded reference frames over a lookahead window, project, and provide the current prompt to the actor. The changing z sequence tells the same policy to pursue a changing objective. Check which script supplies the window: the generic method uses 8, while the active evaluator and video path use 1.</p>'),
      block('A reward: combine features of states that score well',
`z += torch.matmul(wr[start_idx:end_idx].to(self.device).T, B)`,
        '<p>Score sampled replay states with the chosen task reward. Accumulate their B-vectors weighted by those scores, then project the result. The released script additionally gives high-reward samples extra softmax weight. The physical replay helps estimate this task direction; no actor gradient step occurs.</p><p>Weak zero-shot behavior can come from a poor value/policy representation, poor state coverage, or an inaccurate inferred z. Keeping the weights fixed while repeating reward inference helps distinguish these possibilities.</p>')
    ], 'Do goal, motion, and reward prompting require three separate actors?', 'They construct different inputs to the same z-conditioned actor. The observation/history interface and policy weights remain the same.', model + 'FBModel.reward_inference')]
  };
})();

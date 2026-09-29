<script>
  let { boards = [], providers = [], onstart } = $props()

  let idea = $state('')
  let board = $state('startup')
  let provider = $state('')
  let model = $state('')
  let baseUrl = $state('')
  let title = $state('')
  let phases = $state({ hostile: true, deliberation: true, coaching: true, synthesis: true })
  let busy = $state(false)
  let iterations = $state(1)
  let target = $state(7)

  let selected = $derived(providers.find((p) => p.name === provider))
  let ready = $derived(idea.trim().length >= 10 && Object.values(phases).some(Boolean))

  async function submit() {
    busy = true
    const llm = {}
    if (provider) llm.provider = provider
    if (model) llm.model = model
    if (baseUrl) llm.base_url = baseUrl
    await onstart({
      idea: idea.trim(),
      board,
      title: title || null,
      llm: Object.keys(llm).length ? llm : null,
      phases: Object.entries(phases).filter(([, v]) => v).map(([k]) => k),
      refine: iterations > 1 ? { max_iterations: iterations, target_score: target } : null,
    })
    busy = false
  }
</script>

<section class="panel">
  <p class="brand">Put your idea in front of the board</p>
  <p class="muted">Hostile round, chaired debate, coaching round, refined pitch. Nothing leaves your machine when you use a local model.</p>

  <label for="idea">Idea</label>
  <textarea id="idea" bind:value={idea} placeholder="A two-pass legal document comparison system that runs locally..."></textarea>

  <div class="row" style="margin-top:12px">
    <div>
      <label for="board">Board</label>
      <select id="board" bind:value={board}>
        {#each boards as b}<option value={b.name}>{b.name} ({b.agents.length} agents)</option>{/each}
      </select>
    </div>
    <div>
      <label for="provider">Provider</label>
      <select id="provider" bind:value={provider}>
        <option value="">auto (server default)</option>
        {#each providers as p}
          <option value={p.name}>{p.name}{p.local ? ' (local)' : p.key_present ? '' : ' (no key)'}</option>
        {/each}
      </select>
    </div>
    <div>
      <label for="model">Model</label>
      <input id="model" bind:value={model} placeholder={selected?.default_model ?? 'provider default'} />
    </div>
    {#if selected?.local || provider === 'openai-compatible'}
      <div>
        <label for="base">Base URL</label>
        <input id="base" bind:value={baseUrl} placeholder={selected?.base_url ?? 'http://host:port/v1'} />
      </div>
    {/if}
  </div>

  <div class="row" style="margin-top:12px">
    <div style="flex:2 1 240px">
      <label for="title">Title (optional)</label>
      <input id="title" bind:value={title} placeholder="Shown in the run list and report" />
    </div>
    <div>
      <p class="muted" style="margin:0 0 4px;font-size:13px">Phases</p>
      <div style="display:flex;gap:12px">
        {#each Object.keys(phases) as p}
          <label style="display:flex;gap:4px;align-items:center;color:inherit"><input type="checkbox" bind:checked={phases[p]} /> {p}</label>
        {/each}
      </div>
    </div>
    <div style="flex:0 0 120px">
      <label for="iter">Refine loop</label>
      <select id="iter" bind:value={iterations}>
        <option value={1}>single pass</option>
        {#each [2, 3, 4, 5] as n}<option value={n}>up to {n} revisions</option>{/each}
      </select>
    </div>
    {#if iterations > 1}
      <div style="flex:0 0 90px">
        <label for="target">Target score</label>
        <input id="target" type="number" min="0" max="10" step="0.5" bind:value={target} />
      </div>
    {/if}
    <div style="flex:0 0 auto">
      <button class="primary" disabled={!ready || busy} onclick={submit}>{busy ? 'Starting...' : 'Convene the board'}</button>
    </div>
  </div>
</section>

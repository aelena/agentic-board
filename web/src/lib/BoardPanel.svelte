<script>
  let { boards = [] } = $props()
  let open = $state('')
  let board = $derived(boards.find((b) => b.name === open))
</script>

<section class="panel" style="margin-top:16px">
  <div class="row" style="align-items:center">
    <div><p class="brand" style="margin:0">Boards</p><p class="muted" style="margin:0">Declared in YAML. Add your own in <code>./boards</code> or <code>~/.idea-refiner/boards</code>.</p></div>
    <div style="flex:0 0 220px">
      <select bind:value={open}>
        <option value="">inspect a board...</option>
        {#each boards as b}<option value={b.name}>{b.name}</option>{/each}
      </select>
    </div>
  </div>
  {#if board}
    <p style="margin:12px 0 0">{board.description}</p>
    <div class="agents">
      {#each board.agents as a}
        <div class="agent">
          <b>{a.hostile?.role || a.role}</b>
          <span class="muted">has seen projects fail from {a.hostile?.focus || a.focus}</span><br />
          <span class="muted">coach: {a.coach?.role || a.role}{a.coach?.focus ? `, ${a.coach.focus}` : ''}</span>
          {#if a.llm}<br /><code>{a.llm.provider ?? ''}{a.llm.provider && a.llm.model ? '/' : ''}{a.llm.model ?? ''}</code>{/if}
        </div>
      {/each}
      <div class="agent" style="border-style:dashed">
        <b>{board.synthesizer.role}</b>
        <span class="muted">{board.synthesizer.goal}</span>
      </div>
    </div>
    <p class="muted" style="margin:10px 0 0">phases: {board.phases.join(', ')}{board.source ? ` | ${board.source}` : ''}</p>
  {/if}
</section>

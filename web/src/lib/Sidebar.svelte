<script>
  let { runs = [], health = null, activeId = null, onopen, onremove, onnew } = $props()
  const when = (iso) => new Date(iso).toLocaleString(undefined, { dateStyle: 'short', timeStyle: 'short' })
</script>

<aside class="sidebar">
  <p class="brand">Idea Refiner</p>
  <p class="muted">{health ? `API v${health.version} | default ${health.provider}` : 'connecting...'}</p>
  <button class="primary" style="width:100%;margin:8px 0 12px" onclick={onnew}>+ New run</button>

  <p class="muted" style="margin:0 0 4px">Runs</p>
  {#if !runs.length}<p class="muted">No runs yet.</p>{/if}
  <ul class="runs">
    {#each runs as r (r.id)}
      <li class:active={r.id === activeId} onclick={() => onopen(r.id)} onkeydown={(e) => e.key === 'Enter' && onopen(r.id)} role="button" tabindex="0">
        <div class="t"><b>{r.title || r.idea}</b></div>
        <div class="muted" style="display:flex;justify-content:space-between;gap:6px">
          <span>{r.project ? `${r.project} | ` : ''}{r.board} | {when(r.created_at)}</span>
          <span class="badge {r.status}">{r.status}</span>
        </div>
        {#if r.status !== 'running'}
          <button style="padding:2px 8px;font-size:12px;margin-top:4px" onclick={(e) => { e.stopPropagation(); onremove(r.id) }}>delete</button>
        {/if}
      </li>
    {/each}
  </ul>
</aside>

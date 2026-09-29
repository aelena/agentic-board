<script>
  import { marked } from 'marked'
  import { api } from './api.js'

  let { run, onback } = $props()

  const PHASES = ['hostile', 'coaching', 'synthesis']
  const TITLES = { hostile: 'Hostile feedback', coaching: 'Coaching advice', synthesis: 'Refined pitch' }

  // Build a phase -> cards view from either the live event stream or a finished result.
  let view = $derived.by(() => {
    const out = {}
    if (run.result) {
      for (const p of run.result.phases) out[p.phase] = { done: true, tally: p.tally, cards: p.outputs.map((o) => ({ ...o, done: true })) }
      return out
    }
    for (const e of run.events) {
      if (e.type === 'phase_start') out[e.phase] = { done: false, cards: [] }
      if (e.type === 'agent_start') out[e.phase]?.cards.push({ agent_id: e.agent_id, role: e.role, text: '', done: false })
      if (e.type === 'agent_done') {
        const c = out[e.phase]?.cards.find((c) => c.agent_id === e.agent_id && !c.done)
        const verdict = e.data?.verdict ?? null
        if (c) { c.text = e.text; c.verdict = verdict; c.done = true } else out[e.phase]?.cards.push({ agent_id: e.agent_id, role: e.role, text: e.text, verdict, done: true })
      }
      if (e.type === 'phase_done' && out[e.phase]) { out[e.phase].done = true; out[e.phase].tally = e.data?.tally ?? null }
    }
    return out
  })

  let model = $derived(run.result?.model ?? run.events.find((e) => e.type === 'run_start')?.text ?? '')
  let idea = $derived(run.result?.idea ?? run.request?.idea ?? run.idea ?? '')
  let live = $derived(run.status === 'running' || run.status === 'queued')
  let errorText = $derived(run.error ?? run.events.find((e) => e.type === 'error')?.text)
  const md = (t) => marked.parse(t || '')
  const tallyText = (t) => Object.entries(t.votes).map(([d, n]) => `${n} ${d}`).join(' · ') + (t.mean_score != null ? ` · mean ${t.mean_score}/10` : '')
</script>

<div class="row" style="align-items:center;margin-bottom:12px">
  <div style="flex:0 0 auto"><button onclick={onback}>&larr; New run</button></div>
  <div>
    <p class="brand" style="margin:0">{run.result?.title ?? run.title ?? 'Board session'}
      {#if live}<span class="badge running"><span class="spinner"></span> running</span>{:else if run.status === 'error'}<span class="badge error">error</span>{/if}
    </p>
    <p class="muted" style="margin:0">run {run.id}{model ? ` | ${model}` : ''}{run.result?.seconds ? ` | ${Math.round(run.result.seconds)}s` : ''}</p>
  </div>
  {#if run.result}
    <div style="flex:0 0 auto"><a href={api.reportUrl(run.id)} download="report-{run.id}.md"><button>Download report.md</button></a></div>
  {/if}
</div>

{#if idea}
  <div class="panel"><p class="muted" style="margin:0 0 4px">Idea</p><div class="md">{@html md(idea)}</div></div>
{/if}

{#if errorText}<p class="err">{errorText}</p>{/if}

{#each PHASES as p}
  {#if view[p]}
    <section class="phase {p}">
      <h2>{TITLES[p]} {#if !view[p].done}<span class="spinner"></span>{/if}</h2>
      {#if view[p].tally?.decision}
        <p class="tally"><span class="verdict v-{view[p].tally.decision}">{view[p].tally.decision}{view[p].tally.unanimous ? ' (unanimous)' : ''}</span> {tallyText(view[p].tally)}{#if view[p].tally.dissent.length} · dissent: {view[p].tally.dissent.join(', ')}{/if}</p>
      {/if}
      {#each view[p].cards as c (c.agent_id + c.role)}
        <article class="card" class:pending={!c.done} class:pitch={p === 'synthesis'}>
          <h3>{c.role} {#if !c.done}<span class="spinner"></span>{/if}
            {#if c.verdict}<span class="verdict v-{c.verdict.decision}">{c.verdict.decision} {c.verdict.score}/10</span>{/if}</h3>
          {#if c.done}<div class="md">{@html md(c.text)}</div>{:else}<p class="muted">thinking...</p>{/if}
          {#if c.verdict?.issues?.length}<ul class="issues">{#each c.verdict.issues as i}<li>{i}</li>{/each}</ul>{/if}
        </article>
      {/each}
    </section>
  {/if}
{/each}

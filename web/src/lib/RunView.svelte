<script>
  import { marked } from 'marked'
  import { api } from './api.js'

  let { run, onback } = $props()

  const TITLES = { hostile: 'Hostile feedback', deliberation: 'Deliberation', coaching: 'Coaching advice', synthesis: 'Refined pitch' }
  const keyOf = (x) => `${x.iteration ?? 1}:${x.phase}:${x.round ?? ''}`

  // Ordered sections (one per phase, per deliberation round, per iteration), built from either the live
  // event stream or a finished result. Decisions (chair, loop control) attach to the section they close.
  let sections = $derived.by(() => {
    const out = []
    const find = (x) => out.find((s) => s.key === keyOf(x))
    if (run.result) {
      const its = run.result.iterations ?? []
      run.result.phases.forEach((p, i) => {
        out.push({
          key: keyOf(p), phase: p.phase, round: p.round, iteration: p.iteration ?? 1, done: true, tally: p.tally,
          chair: p.chair, closed: p.closed, cards: p.outputs.map((o) => ({ ...o, done: true })),
        })
        const n = p.iteration ?? 1
        const lastOfIteration = run.result.phases[i + 1]?.iteration !== p.iteration
        if (its.length > 1 && lastOfIteration) {
          const it = its.find((x) => x.n === n)
          out.push({ key: `loop:${n}`, loop: true, iteration: n, action: it?.outcome ? 'stop' : 'iterate', text: it?.outcome ?? 'sent back to the board' })
        }
      })
      return out
    }
    for (const e of run.events) {
      if (e.type === 'decision' && !e.phase) {
        out.push({ key: `loop:${e.iteration}`, loop: true, iteration: e.iteration, action: e.data?.action, text: e.text })
        continue
      }
      if (e.type === 'phase_start') out.push({ key: keyOf(e), phase: e.phase, round: e.round, iteration: e.iteration ?? 1, done: false, cards: [] })
      const s = e.phase ? find(e) : null
      if (!s) continue
      if (e.type === 'agent_start' && e.agent_id !== 'chair') s.cards.push({ agent_id: e.agent_id, role: e.role, text: '', done: false })
      if (e.type === 'agent_start' && e.agent_id === 'chair') s.chairPending = true
      if (e.type === 'agent_done' && e.agent_id === 'chair') { s.chairPending = false; s.chair = { summary: e.text } }
      else if (e.type === 'agent_done') {
        const verdict = e.data?.verdict ?? null
        const c = s.cards.find((c) => c.agent_id === e.agent_id && !c.done)
        if (c) { c.text = e.text; c.verdict = verdict; c.done = true } else s.cards.push({ agent_id: e.agent_id, role: e.role, text: e.text, verdict, done: true })
      }
      if (e.type === 'phase_done') { s.done = true; s.tally = e.data?.tally ?? null }
      if (e.type === 'decision') {
        if (e.data?.action === 'close') s.closed = e.text
        else s.next = e.text
      }
    }
    return out
  })

  let model = $derived(run.result?.model ?? run.events.find((e) => e.type === 'run_start')?.text ?? '')
  let idea = $derived(run.result?.idea ?? run.request?.idea ?? run.idea ?? '')
  let live = $derived(run.status === 'running' || run.status === 'queued')
  let errorText = $derived(run.error ?? run.events.find((e) => e.type === 'error')?.text)
  let final = $derived(run.result?.verdict)
  const md = (t) => marked.parse(t || '')
  const tallyText = (t) => Object.entries(t.votes).map(([d, n]) => `${n} ${d}`).join(' · ') + (t.mean_score != null ? ` · mean ${t.mean_score}/10` : '')
  const title = (s) => TITLES[s.phase] + (s.round ? `, round ${s.round}` : '')
</script>

<div class="row" style="align-items:center;margin-bottom:12px">
  <div style="flex:0 0 auto"><button onclick={onback}>&larr; New run</button></div>
  <div>
    <p class="brand" style="margin:0">{run.result?.title ?? run.title ?? 'Board session'}
      {#if live}<span class="badge running"><span class="spinner"></span> running</span>{:else if run.status === 'error'}<span class="badge error">error</span>{/if}
      {#if final?.decision}<span class="verdict v-{final.decision}">{final.decision}</span>{/if}
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

{#each sections as s, i (s.key)}
  {#if s.iteration > 1 && sections[i - 1]?.iteration !== s.iteration}
    <h2 class="revision">Revision {s.iteration}</h2>
  {/if}
  {#if s.loop}
    <p class="loop {s.action}">{s.action === 'iterate' ? 'Back to the board' : 'Loop stopped'}: {s.text}</p>
  {:else}
  <section class="phase {s.phase}">
    <h2>{title(s)} {#if !s.done}<span class="spinner"></span>{/if}</h2>
    {#if s.tally?.decision}
      <p class="tally"><span class="verdict v-{s.tally.decision}">{s.tally.decision}{s.tally.unanimous ? ' (unanimous)' : ''}</span> {tallyText(s.tally)}{#if s.tally.dissent.length} · dissent: {s.tally.dissent.join(', ')}{/if}</p>
    {/if}
    {#each s.cards as c (c.agent_id + c.role)}
      <article class="card" class:pending={!c.done} class:pitch={s.phase === 'synthesis'}>
        <h3>{c.role} {#if !c.done}<span class="spinner"></span>{/if}
          {#if c.verdict}<span class="verdict v-{c.verdict.decision}">{c.verdict.decision} {c.verdict.score}/10</span>{/if}</h3>
        {#if c.done}<div class="md">{@html md(c.text)}</div>{:else}<p class="muted">thinking...</p>{/if}
        {#if c.verdict?.issues?.length}<ul class="issues">{#each c.verdict.issues as i}<li>{i}</li>{/each}</ul>{/if}
      </article>
    {/each}
    {#if s.chairPending}<article class="card chair pending"><h3>Chair <span class="spinner"></span></h3><p class="muted">weighing the round...</p></article>{/if}
    {#if s.chair?.summary}
      <article class="card chair">
        <h3>Chair</h3>
        <div class="md">{@html md(s.chair.summary)}</div>
        {#if s.chair.questions && Object.keys(s.chair.questions).length}
          <ul class="issues">{#each Object.entries(s.chair.questions) as [who, q]}<li><b>{who}:</b> {q}</li>{/each}</ul>
        {/if}
      </article>
    {/if}
    {#if s.closed}<p class="decision">Closed: {s.closed}</p>{:else if s.next}<p class="decision">Another round: {s.next}</p>{/if}
  </section>
  {/if}
{/each}

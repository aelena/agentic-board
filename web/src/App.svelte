<script>
  import { onMount } from 'svelte'
  import { api } from './lib/api.js'
  import Sidebar from './lib/Sidebar.svelte'
  import RunForm from './lib/RunForm.svelte'
  import RunView from './lib/RunView.svelte'
  import BoardPanel from './lib/BoardPanel.svelte'

  let health = $state(null)
  let boards = $state([])
  let providers = $state([])
  let runs = $state([])
  let error = $state('')

  // The run currently shown: either live (events accumulate) or loaded from history (result only).
  let current = $state(null) // { id, status, events: [], result: null, request: {} }
  let stop = null

  async function refresh() {
    try {
      ;[health, boards, providers, runs] = await Promise.all([api.health(), api.boards(), api.providers(), api.runs()])
      error = ''
    } catch (e) {
      error = `API unreachable: ${e.message}`
    }
  }

  async function start(body) {
    error = ''
    try {
      const { id } = await api.startRun(body)
      follow({ id, status: 'running', events: [], result: null, request: body })
      runs = await api.runs()
    } catch (e) {
      error = e.message
    }
  }

  function follow(run) {
    stop?.()
    current = run
    stop = api.events(run.id, (e) => {
      current.events = [...current.events, e]
      if (e.type === 'error') current.status = 'error'
    }, async () => {
      const data = await api.run(run.id).catch(() => null)
      if (data) { current.status = data.status; current.result = data.result; current.error = data.error }
      runs = await api.runs()
    })
  }

  async function open(id) {
    stop?.()
    const data = await api.run(id)
    if (data.status === 'running' || data.status === 'queued') return follow({ ...data, events: [], result: null, request: {} })
    current = { ...data, events: [], request: { board: data.board } }
  }

  async function remove(id) {
    await api.deleteRun(id)
    if (current?.id === id) current = null
    runs = await api.runs()
  }

  onMount(refresh)
</script>

<div class="layout">
  <Sidebar {runs} {health} activeId={current?.id} onopen={open} onremove={remove} onnew={() => (current = null)} />
  <main class="main">
    {#if error}<p class="err">{error}</p>{/if}
    {#if !current}
      <RunForm {boards} {providers} onstart={start} />
      <BoardPanel {boards} />
    {:else}
      <RunView run={current} onback={() => (current = null)} />
    {/if}
  </main>
</div>

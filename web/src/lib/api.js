// Thin client over the Sanic API. Relative URLs: Vite proxies /api in dev, Sanic serves both in prod.

async function req(path, opts = {}) {
  const r = await fetch(path, { headers: { 'content-type': 'application/json' }, ...opts })
  const body = r.headers.get('content-type')?.includes('json') ? await r.json() : await r.text()
  if (!r.ok) throw new Error(typeof body === 'string' ? body : body.error ? JSON.stringify(body.error) : r.statusText)
  return body
}

export const api = {
  health: () => req('/api/health'),
  providers: () => req('/api/providers'),
  boards: () => req('/api/boards'),
  board: (name) => req(`/api/boards/${encodeURIComponent(name)}`),
  validateBoard: (yaml, name) => req('/api/boards/validate', { method: 'POST', body: JSON.stringify({ yaml, name }) }),
  runs: () => req('/api/runs'),
  run: (id) => req(`/api/runs/${id}`),
  startRun: (body) => req('/api/runs', { method: 'POST', body: JSON.stringify(body) }),
  deleteRun: (id) => req(`/api/runs/${id}`, { method: 'DELETE' }),
  reportUrl: (id) => `/api/runs/${id}/report.md`,

  /** Subscribe to a run's SSE stream. Returns a stop function. */
  events(id, onEvent, onEnd) {
    const es = new EventSource(`/api/runs/${id}/events`)
    const types = ['run_start', 'phase_start', 'agent_start', 'agent_done', 'phase_done', 'decision', 'run_done', 'error']
    for (const t of types) {
      es.addEventListener(t, (m) => {
        const e = JSON.parse(m.data)
        onEvent(e)
        if (t === 'run_done' || t === 'error') { es.close(); onEnd?.(e) }
      })
    }
    es.onerror = () => { es.close(); onEnd?.(null) }
    return () => es.close()
  },
}

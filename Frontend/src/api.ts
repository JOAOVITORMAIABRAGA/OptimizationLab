export type Variable = {
  name: string
  variable_type: string
  lower_bound: number | null
  upper_bound: number | null
}

export type ProblemForm = {
  name: string
  description: string
  problem_family: string
  mathematical_properties: string[]
  variables: Variable[]
  objective_kind: string
  objective_sense: string
  expression: string
  representation: string
}

export type AIProblem = Omit<ProblemForm, 'representation'> & {
  representation: string | null
}

export type Analysis = {
  problem: Record<string, unknown>
  validation: { valid: boolean; blocking: boolean; errors: string[]; warnings: string[] }
  compatibility: { algorithm_id: string; algorithm_name: string; status: string; reasons: string[]; warnings: string[]; required_adapters: string[]; required_operators: string[] }[]
  recommendations: { algorithm_id: string; algorithm_name: string; score: number; rank: number; rationale: string; strengths: string[]; weaknesses: string[]; evidence: string[] }[]
  excluded_algorithms: { algorithm_id: string; reason: string; compatibility_status: string | null; evidence: string[] }[]
}

export type AIModel = {
  problem: AIProblem
  explanation: string
  assumptions: string[]
  dataset: { filename: string; row_count: number; column_count: number; columns: string[]; sample_rows: Record<string, string>[] }
  incomplete: boolean
}

const API_URL = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8001'

async function parseError(response: Response, fallback: string) {
  const data = await response.json().catch(() => ({}))
  return data.detail ?? fallback
}

export async function modelProblem(description: string, file: File): Promise<AIModel> {
  const body = new FormData()
  body.append('description', description)
  body.append('file', file)
  const response = await fetch(`${API_URL}/api/model`, { method: 'POST', body })
  if (!response.ok) throw new Error(await parseError(response, 'Could not model the problem with AI.'))
  return response.json()
}

export async function analyze(problem: ProblemForm): Promise<Analysis> {
  const response = await fetch(`${API_URL}/api/analyze`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(problem) })
  if (!response.ok) throw new Error(await parseError(response, 'Could not analyze problem.'))
  return response.json()
}

export async function metadata() {
  const response = await fetch(`${API_URL}/api/metadata`)
  if (!response.ok) throw new Error('Could not load backend metadata.')
  return response.json()
}

export type SolveResult = {
  algorithm_id: string
  solution: number[]
  variable_values: Record<string, number>
  objective_value: number
  parameters: Record<string, unknown>
}

export async function solve(problem: ProblemForm, algorithm_id: string): Promise<SolveResult> {
  const response = await fetch(`${API_URL}/api/solve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...problem, algorithm_id })
  })
  if (!response.ok) throw new Error(await parseError(response, 'Could not solve the problem.'))
  return response.json()
}

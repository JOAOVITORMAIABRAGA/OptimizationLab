import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, ArrowRight, Check, FileText, Settings2, Sparkles, Upload, X } from 'lucide-react'
import { analyze, metadata, modelProblem, solve, type AIModel, type Analysis, type ProblemForm, type SolveResult, type Variable } from './api'

const fallback = { problem_families: ['continuous_optimization','routing','scheduling','assignment','feature_selection','generic'], mathematical_properties: ['continuous','discrete','binary','constrained','unconstrained','linear','nonlinear','integer','mixed_integer','black_box','multiobjective'], variable_types: ['continuous','integer','binary','categorical','discrete'], representations: ['vector','permutation','graph','set','sequence','mixed','matrix'], objective_kinds: ['single','multi'], objective_senses: ['minimize','maximize'] }
const emptyForm: ProblemForm = { name: '', description: '', problem_family: 'generic', mathematical_properties: [], variables: [{name:'x1',variable_type:'continuous',lower_bound:null,upper_bound:null}], objective_kind:'single', objective_sense:'minimize', expression:'x1', representation:'vector' }

type Mode = 'ai' | 'advanced'

function renderValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value)
  }

  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function App() {
  const [mode, setMode] = useState<Mode>('ai')
  const [description, setDescription] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [aiModel, setAiModel] = useState<AIModel | null>(null)
  const [form, setForm] = useState<ProblemForm>(emptyForm)
  const [meta, setMeta] = useState(fallback)
  const [result, setResult] = useState<Analysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [solution, setSolution] = useState<SolveResult | null>(null)
  const [selectedAlgorithm, setSelectedAlgorithm] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => { metadata().then(setMeta).catch(() => {}) }, [])

  const updateVar = (i:number, patch:Partial<Variable>) => setForm(f=>({...f,variables:f.variables.map((v,j)=>j===i?{...v,...patch}:v)}))
  const compatibleCount = useMemo(() => result?.compatibility.filter(a=>a.status!=='incompatible').length ?? 0,[result])

  const runAI = async () => {
    if (!description.trim()) return setError('Describe the optimization problem first.')
    if (files.length === 0) return setError('Upload at least one dataset first.')
    setLoading(true); setError(''); setAiModel(null); setResult(null); setSolution(null)
    try { setAiModel(await modelProblem(description, files)) } catch(e) { setError(e instanceof Error?e.message:'AI modeling failed.') } finally { setLoading(false) }
  }

  const validateProblem = async (problem: ProblemForm) => {
    setLoading(true); setError(''); setSolution(null)
    try {
      const normalized = { ...problem, representation: problem.representation || 'vector' }
      const analysis = await analyze(normalized)
      setForm(normalized)
      setResult(analysis)
      setSelectedAlgorithm(analysis.recommendations[0]?.algorithm_id ?? analysis.candidates[0]?.algorithm_id ?? '')
      return analysis
    } catch(e) {
      setError(e instanceof Error?e.message:'Request failed.')
      return null
    } finally { setLoading(false) }
  }

  const validate = async () => validateProblem(form)

  const acceptAI = async () => {
    if (!aiModel || aiModel.incomplete) return
    const nextForm: ProblemForm = {
      ...aiModel.problem,
      representation: aiModel.problem.representation || 'vector',
    }
    setForm(nextForm)
    setAiModel(null)
    setMode('ai')
    await validateProblem(nextForm)
  }

  const solveSelected = async () => {
    if (!result?.validation.valid || !selectedAlgorithm) return
    const algorithm = selectedAlgorithm
    setLoading(true); setError('')
    try {
      setSolution(await solve(form, algorithm))
    } catch(e) {
      setError(e instanceof Error?e.message:'Could not solve the problem.')
    } finally { setLoading(false) }
  }

  const toggle = (p:string) => setForm(f=>({...f,mathematical_properties:f.mathematical_properties.includes(p)?f.mathematical_properties.filter(x=>x!==p):[...f.mathematical_properties,p]}))
  const addFiles = (incoming: File[]) => {
    const allowed = ['.csv', '.txt', '.xlsx']
    const invalid = incoming.find(f => !allowed.some(ext => f.name.toLowerCase().endsWith(ext)))
    if (invalid) return setError(`Unsupported dataset format: ${invalid.name}. Use CSV, TXT or XLSX.`)
    setFiles(current => {
      const merged = [...current, ...incoming]
      const unique = merged.filter((file, index, all) => all.findIndex(other => other.name === file.name && other.size === file.size && other.lastModified === file.lastModified) === index)
      return unique.slice(0, 10)
    })
    setError('')
  }

  return <div className="shell">
    <header><div><div className="eyebrow">AI MODEL · VALIDATE · RECOMMEND</div><h1>Optimization <span>Lab</span></h1><p>Turn a real-world problem into a validated optimization model.</p></div><div className="status"><i/> Backend-driven</div></header>
    <main>
      <section className="hero card">
        <div className="hero-copy"><span className="step">01</span><div><h2>What do you want to optimize?</h2><p>The default flow uses the AI modeler to translate your description and dataset into an OptimizationProblem.</p></div></div>
        <label>Problem description<textarea rows={5} placeholder="Example: I need to optimize delivery routes for 120 customers. Each vehicle has a capacity limit and I want to minimize total distance..." value={description} onChange={e=>setDescription(e.target.value)}/></label>
        <div className="upload" onClick={()=>fileRef.current?.click()} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();addFiles(Array.from(e.dataTransfer.files))}}>
          <input ref={fileRef} type="file" multiple accept=".csv,.txt,.xlsx,text/csv,text/plain,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" hidden onChange={e=>addFiles(Array.from(e.target.files ?? []))}/>
          {files.length > 0 ? <><FileText size={26}/><div><strong>{files.length} dataset{files.length===1?'':'s'} selected</strong><span>CSV, TXT or XLSX · up to 10 files</span></div><button className="ghost" type="button" onClick={e=>{e.stopPropagation();setFiles([])}}><X size={16}/></button></> : <><Upload size={26}/><div><strong>Drop your datasets here</strong><span>CSV, TXT or XLSX · up to 10 files · 10 MB each</span></div></>}
        </div>
        {files.length > 0 && <div className="dataset-list">{files.map((item,index)=><div className="dataset-card" key={`${item.name}-${item.size}-${item.lastModified}`}><FileText size={16}/><strong>{item.name}</strong><span>{(item.size/1024/1024).toFixed(2)} MB</span><button className="icon" type="button" onClick={()=>setFiles(current=>current.filter((_,i)=>i!==index))}>×</button></div>)}</div>}
        {error && <div className="alert error"><X size={18}/>{error}</div>}
        <button className="primary" onClick={runAI} disabled={loading}>{loading?'Modeling problem…':'Analyze with AI'}<Sparkles size={18}/></button>
        <button className="advanced-link" onClick={()=>{setMode('advanced');setError('')}}><Settings2 size={15}/> Configure manually instead</button>
      </section>

      {aiModel && <section className="card ai-review">
        <div className="section-head"><span className="step">02</span><div><h2>AI-generated problem model</h2><p>Review how the AI interpreted your description and dataset before validation.</p></div></div>
        <div className="ai-summary"><Sparkles size={18}/><p>{aiModel.explanation}</p></div>
        <div className="grid two"><div className="info"><span>Problem family</span><strong>{aiModel.problem.problem_family}</strong></div><div className="info"><span>Objective</span><strong>{aiModel.problem.objective_sense}{aiModel.problem.expression ? ` · ${aiModel.problem.expression}` : aiModel.problem.objective_metric ? ` · ${aiModel.problem.objective_metric}` : ' · Not yet determined'}</strong></div></div>
        <div className="grid two"><div className="info"><span>Decision variables</span><strong>{aiModel.problem.variables.length > 0 ? aiModel.problem.variables.map(v=>v.name).join(', ') : 'Not determined'}</strong></div><div className="info"><span>Model status</span><strong>{aiModel.incomplete ? 'Incomplete — needs review' : 'Complete proposal'}</strong></div></div>
        <div className="info"><span>Mathematical properties</span><div className="chips readonly">{aiModel.problem.mathematical_properties.map(p=><span className="chip active" key={p}>{p}</span>)}</div></div>
        <div className="dataset-card"><div><FileText size={17}/><strong>{aiModel.datasets.length} data source{aiModel.datasets.length===1?'':'s'} analyzed</strong><span>{aiModel.datasets.map(source => source.source_name || source.filename).join(' · ')}</span></div></div>
        <div className="dataset-list">{aiModel.datasets.map(source=><div className="dataset-card" key={source.source_name || source.filename}><div><FileText size={15}/><strong>{source.source_name || source.filename}</strong><span>{source.row_count.toLocaleString()} rows · {source.column_count} columns · {source.format.toUpperCase()}</span></div><small>{source.columns.join(' · ')}</small></div>)}</div>
        {aiModel.assumptions.length>0 && <div className="assumptions"><AlertTriangle size={16}/><div><strong>AI assumptions / uncertainties</strong>{aiModel.assumptions.map(a=><div key={a}>• {a}</div>)}</div></div>}
        <div className="actions"><button className="ghost" onClick={()=>{setForm({...aiModel.problem, representation: aiModel.problem.representation || 'vector'});setAiModel(null);setMode('advanced')}}><Settings2 size={16}/> Edit model</button><button className="primary compact" disabled={aiModel.incomplete || loading} onClick={acceptAI}>{aiModel.incomplete ? 'Edit model to continue' : loading ? 'Validating…' : 'Accept & validate'} {!aiModel.incomplete && !loading && <ArrowRight size={17}/>}</button></div>
      </section>}

      {mode==='advanced' && <section className="card configuration">
        <div className="section-head"><span className="step">02A</span><div><h2>Advanced model editor</h2><p>For expert users who need fine-grained control over the OptimizationProblem.</p></div></div>
        <div className="grid two"><label>Name<input value={form.name} onChange={e=>setForm({...form,name:e.target.value})}/></label><label>Problem family<select value={form.problem_family} onChange={e=>setForm({...form,problem_family:e.target.value})}>{meta.problem_families.map(x=><option key={x}>{x}</option>)}</select></label></div>
        <label>Description <span>optional</span><textarea rows={2} value={form.description} onChange={e=>setForm({...form,description:e.target.value})}/></label>
        <div className="field-title">Mathematical properties</div><div className="chips">{meta.mathematical_properties.map(p=><button type="button" key={p} className={form.mathematical_properties.includes(p)?'chip active':'chip'} onClick={()=>toggle(p)}>{form.mathematical_properties.includes(p)?'✓ ':''}{p}</button>)}</div>
        <div className="subhead"><h3>Variables</h3><button className="ghost" onClick={()=>setForm(f=>({...f,variables:[...f.variables,{name:`x${f.variables.length+1}`,variable_type:'continuous',lower_bound:null,upper_bound:null}]}))}>+ Add variable</button></div>
        <div className="variable-list">{form.variables.map((v,i)=><div className="variable" key={i}><input aria-label="Variable name" value={v.name} onChange={e=>updateVar(i,{name:e.target.value})}/><select value={v.variable_type} onChange={e=>updateVar(i,{variable_type:e.target.value})}>{meta.variable_types.map(x=><option key={x}>{x}</option>)}</select><input type="number" aria-label="Lower bound" value={v.lower_bound ?? ''} onChange={e=>updateVar(i,{lower_bound:e.target.value===''?null:Number(e.target.value)})}/><input type="number" aria-label="Upper bound" value={v.upper_bound ?? ''} onChange={e=>updateVar(i,{upper_bound:e.target.value===''?null:Number(e.target.value)})}/><button className="icon" disabled={form.variables.length===1} onClick={()=>setForm(f=>({...f,variables:f.variables.filter((_,j)=>j!==i)}))}>×</button></div>)}</div>
        <div className="subhead"><h3>Objective</h3></div><div className="grid three"><label>Kind<select value={form.objective_kind} onChange={e=>setForm({...form,objective_kind:e.target.value})}>{meta.objective_kinds.map(x=><option key={x}>{x}</option>)}</select></label><label>Sense<select value={form.objective_sense} onChange={e=>setForm({...form,objective_sense:e.target.value})}>{meta.objective_senses.map(x=><option key={x}>{x}</option>)}</select></label><label>Representation<select value={form.representation} onChange={e=>setForm({...form,representation:e.target.value})}><option value="" disabled>Select representation</option>{meta.representations.map(x=><option key={x}>{x}</option>)}</select></label></div>
        <label>Expression <span>safe structured parser · no eval/exec</span><input className="expression" value={form.expression} onChange={e=>setForm({...form,expression:e.target.value})}/></label>
        <button className="primary" onClick={validate} disabled={loading}>{loading?'Validating…':'Validate model'}<ArrowRight size={18}/></button>
      </section>}

      {result&&<>
        <section className="card"><div className="section-head"><span className="step">03</span><div><h2>Validation</h2><p>Returned by the existing ValidationEngine.</p></div></div><div className={result.validation.valid?'result success':'result failure'}>{result.validation.valid?<Check/>:<X/>}<div><strong>{result.validation.valid?'Problem is valid':'Problem has validation errors'}</strong>{result.validation.errors.map(e=><div className="message" key={e}>• {e}</div>)}{result.validation.warnings.map(w=><div className="message warning" key={w}><AlertTriangle size={14}/> {w}</div>)}</div></div></section>
        <section className="card"><div className="section-head"><span className="step">04</span><div><h2>Structured problem</h2><p>What the user/AI proposed → how the domain understood it.</p></div></div><pre className="json">{JSON.stringify(result.problem,null,2)}</pre></section>
        <section className="card"><div className="section-head"><span className="step">05</span><div><h2>Compatible algorithms</h2><p>{result.validation.valid ? `${compatibleCount} executable candidate${compatibleCount===1?'':'s'} passed capability matching. Compatibility is independent from recommendation.` : `${compatibleCount} algorithm${compatibleCount===1?'':'s'} match the declared capabilities, but execution is blocked until the validation errors are resolved.`}</p></div></div><div className="algorithm-grid">{result.compatibility.map(a=><article className={`algorithm ${a.status==='incompatible'?'bad':''}`} key={a.algorithm_id}><div className="algo-top"><strong>{a.algorithm_name}</strong>{a.status==='incompatible'?<X/>:<Check/>}</div><span className="badge">{a.status.replaceAll('_',' ')}</span>{a.target_representation&&<div className="message">Adapted target: {a.target_representation}</div>}{a.reasons.length>0&&<ul>{a.reasons.slice(0,3).map(r=><li key={r}>{r}</li>)}</ul>}</article>)}</div></section>
        <section className="card recommendations"><div className="section-head"><span className="step">06</span><div><h2>Choose an algorithm</h2><p><Sparkles size={15}/> The first candidate is recommended by structural fit, but the final choice is yours.</p></div></div>{result.candidates.length===0?<div className="empty">No executable compatible algorithms are available.</div>:<div className="algorithm-grid">{result.candidates.map(c=><article className={`recommendation ${selectedAlgorithm===c.algorithm_id?'selected':''}`} key={c.algorithm_id} onClick={()=>{setSelectedAlgorithm(c.algorithm_id);setSolution(null)}} role="button" tabIndex={0} onKeyDown={e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();setSelectedAlgorithm(c.algorithm_id);setSolution(null)}}}><div className="rank">{c.recommended?'★':`#${result.recommendations.find(r=>r.algorithm_id===c.algorithm_id)?.rank ?? '–'}`}</div><div className="rec-body"><div className="rec-title"><strong>{c.algorithm_name}</strong><b>{c.recommendation_score.toFixed(2)}</b></div><p>{c.recommended?'Recommended by structural fit.':'Compatible candidate — available for manual selection.'}</p><div className="evidence"><span>{c.compatibility.replaceAll('_',' ')}</span><span>{c.algorithm_type}</span><span>cost: {c.estimated_cost}</span>{c.adaptation.map(a=><span key={a}>adapter: {a}</span>)}</div></div></article>)}</div>}</section>
        {result.validation.valid && result.candidates.length > 0 && !solution && <section className="card solve-card"><div className="section-head"><span className="step">07</span><div><h2>Solution</h2><p>Selected algorithm: <strong>{result.candidates.find(c=>c.algorithm_id===selectedAlgorithm)?.algorithm_name ?? selectedAlgorithm}</strong>. You can change the selection above at any time.</p></div></div><div className="result success"><Check/><div><strong>{selectedAlgorithm ? 'Ready to solve' : 'Select an algorithm'}</strong><div className="message">No automatic execution will occur unless you choose a candidate.</div></div></div><button className="primary" onClick={solveSelected} disabled={loading || !selectedAlgorithm}>{loading?'Solving…':'Solve selected algorithm'}<ArrowRight size={18}/></button></section>}
        {solution && <section className="card solve-card"><div className="section-head"><span className="step">07</span><div><h2>Solution</h2><p>Returned by the selected executable solver.</p></div></div><div className="result success"><Check/><div><strong>Solution found</strong><div className="message">Algorithm: {solution.algorithm_id}</div><div className="message">Objective value: {solution.objective_value}</div></div></div><div className="variable-list">{Object.entries(solution.variable_values).map(([name,value])=><div className="variable" key={name}><strong>{name}</strong><span>{renderValue(value)}</span></div>)}</div><pre className="json">{JSON.stringify(solution.parameters,null,2)}</pre></section>}
      </>}
    </main><footer>Optimization Lab · AI model · validate · recommend · solve</footer>
  </div>
}
export default App
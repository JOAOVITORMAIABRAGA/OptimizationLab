import { useEffect, useMemo, useRef, useState } from 'react'
import { AlertTriangle, ArrowRight, Check, FileText, Settings2, Sparkles, Upload, X } from 'lucide-react'
import { analyze, metadata, modelProblem, type AIModel, type Analysis, type ProblemForm, type Variable } from './api'

const fallback = { problem_families: ['continuous_optimization','routing','scheduling','assignment','feature_selection','generic'], mathematical_properties: ['continuous','discrete','binary','constrained','unconstrained','linear','nonlinear','integer','mixed_integer','black_box','multiobjective'], variable_types: ['continuous','integer','binary','categorical','discrete'], representations: ['vector','permutation','graph','set','sequence','mixed','matrix'], objective_kinds: ['single','multi'], objective_senses: ['minimize','maximize'] }
const emptyForm: ProblemForm = { name: '', description: '', problem_family: 'generic', mathematical_properties: [], variables: [{name:'x1',variable_type:'continuous',lower_bound:null,upper_bound:null}], objective_kind:'single', objective_sense:'minimize', expression:'x1', representation:'vector' }

type Mode = 'ai' | 'advanced'

function App() {
  const [mode, setMode] = useState<Mode>('ai')
  const [description, setDescription] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [aiModel, setAiModel] = useState<AIModel | null>(null)
  const [form, setForm] = useState<ProblemForm>(emptyForm)
  const [meta, setMeta] = useState(fallback)
  const [result, setResult] = useState<Analysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => { metadata().then(setMeta).catch(() => {}) }, [])

  const updateVar = (i:number, patch:Partial<Variable>) => setForm(f=>({...f,variables:f.variables.map((v,j)=>j===i?{...v,...patch}:v)}))
  const compatibleCount = useMemo(() => result?.compatibility.filter(x=>x.status!=='incompatible').length ?? 0,[result])

  const runAI = async () => {
    if (!description.trim()) return setError('Describe the optimization problem first.')
    if (!file) return setError('Upload a CSV dataset first.')
    setLoading(true); setError(''); setAiModel(null); setResult(null)
    try { setAiModel(await modelProblem(description, file)) } catch(e) { setError(e instanceof Error?e.message:'AI modeling failed.') } finally { setLoading(false) }
  }

  const validate = async () => {
    setLoading(true); setError('')
    try { setResult(await analyze(form)) } catch(e) { setError(e instanceof Error?e.message:'Request failed.') } finally { setLoading(false) }
  }

  const acceptAI = () => { if (!aiModel) return; setForm({...aiModel.problem, representation: aiModel.problem.representation ?? ''}); setMode('advanced'); setAiModel(null) }
  const toggle = (p:string) => setForm(f=>({...f,mathematical_properties:f.mathematical_properties.includes(p)?f.mathematical_properties.filter(x=>x!==p):[...f.mathematical_properties,p]}))
  const selectFile = (f: File | undefined) => { if (!f) return; if (!f.name.toLowerCase().endsWith('.csv')) return setError('Only CSV files are supported in this MVP.'); setFile(f); setError('') }

  return <div className="shell">
    <header><div><div className="eyebrow">AI MODEL · VALIDATE · RECOMMEND</div><h1>Optimization <span>Lab</span></h1><p>Turn a real-world problem into a validated optimization model.</p></div><div className="status"><i/> Backend-driven</div></header>
    <main>
      <section className="hero card">
        <div className="hero-copy"><span className="step">01</span><div><h2>What do you want to optimize?</h2><p>The default flow uses the AI modeler to translate your description and dataset into an OptimizationProblem.</p></div></div>
        <label>Problem description<textarea rows={5} placeholder="Example: I need to optimize delivery routes for 120 customers. Each vehicle has a capacity limit and I want to minimize total distance..." value={description} onChange={e=>setDescription(e.target.value)}/></label>
        <div className="upload" onClick={()=>fileRef.current?.click()} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();selectFile(e.dataTransfer.files[0])}}>
          <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={e=>selectFile(e.target.files?.[0])}/>
          {file ? <><FileText size={26}/><div><strong>{file.name}</strong><span>Dataset ready for analysis</span></div><button className="ghost" type="button" onClick={e=>{e.stopPropagation();setFile(null)}}><X size={16}/></button></> : <><Upload size={26}/><div><strong>Drop your CSV here</strong><span>or click to browse · max 10 MB</span></div></>}
        </div>
        {error && <div className="alert error"><X size={18}/>{error}</div>}
        <button className="primary" onClick={runAI} disabled={loading}>{loading?'Modeling problem…':'Analyze with AI'}<Sparkles size={18}/></button>
        <button className="advanced-link" onClick={()=>{setMode('advanced');setError('')}}><Settings2 size={15}/> Configure manually instead</button>
      </section>

      {aiModel && <section className="card ai-review">
        <div className="section-head"><span className="step">02</span><div><h2>AI-generated problem model</h2><p>Review how the AI interpreted your description and dataset before validation.</p></div></div>
        <div className="ai-summary"><Sparkles size={18}/><p>{aiModel.explanation}</p></div>
        <div className="grid two"><div className="info"><span>Problem family</span><strong>{aiModel.problem.problem_family}</strong></div><div className="info"><span>Objective</span><strong>{aiModel.problem.objective_sense}{aiModel.problem.expression ? ` · ${aiModel.problem.expression}` : ' · Not yet determined'}</strong></div></div>
        <div className="grid two"><div className="info"><span>Decision variables</span><strong>{aiModel.problem.variables.length > 0 ? aiModel.problem.variables.map(v=>v.name).join(', ') : 'Not determined'}</strong></div><div className="info"><span>Model status</span><strong>{aiModel.incomplete ? 'Incomplete — needs review' : 'Complete proposal'}</strong></div></div>
        <div className="info"><span>Mathematical properties</span><div className="chips readonly">{aiModel.problem.mathematical_properties.map(p=><span className="chip active" key={p}>{p}</span>)}</div></div>
        <div className="dataset-card"><div><FileText size={17}/><strong>{aiModel.dataset.filename}</strong><span>{aiModel.dataset.row_count.toLocaleString()} rows · {aiModel.dataset.column_count} columns</span></div><small>{aiModel.dataset.columns.join(' · ')}</small></div>
        {aiModel.assumptions.length>0 && <div className="assumptions"><AlertTriangle size={16}/><div><strong>AI assumptions / uncertainties</strong>{aiModel.assumptions.map(a=><div key={a}>• {a}</div>)}</div></div>}
        <div className="actions"><button className="ghost" onClick={()=>{setForm({...aiModel.problem, representation: aiModel.problem.representation ?? ''});setAiModel(null);setMode('advanced')}}><Settings2 size={16}/> Edit model</button><button className="primary compact" disabled={aiModel.incomplete} onClick={()=>{setForm({...aiModel.problem, representation: aiModel.problem.representation ?? ''});setAiModel(null);setMode('advanced');setTimeout(validate,0)}}>{aiModel.incomplete ? 'Edit model to continue' : 'Accept & validate'} {!aiModel.incomplete && <ArrowRight size={17}/>}</button></div>
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
        <section className="card"><div className="section-head"><span className="step">05</span><div><h2>Compatibility</h2><p>{compatibleCount} algorithm{compatibleCount===1?'':'s'} can work with this problem according to CompatibilityEngine.</p></div></div><div className="algorithm-grid">{result.compatibility.map(a=><article className={`algorithm ${a.status==='incompatible'?'bad':''}`} key={a.algorithm_id}><div className="algo-top"><strong>{a.algorithm_name}</strong>{a.status==='incompatible'?<X/>:<Check/>}</div><span className="badge">{a.status.replaceAll('_',' ')}</span>{a.reasons.length>0&&<ul>{a.reasons.slice(0,3).map(r=><li key={r}>{r}</li>)}</ul>}</article>)}</div></section>
        <section className="card recommendations"><div className="section-head"><span className="step">06</span><div><h2>Recommendations</h2><p><Sparkles size={15}/> RecommendationEngine scoring — not empirical benchmark results.</p></div></div>{result.recommendations.length===0?<div className="empty">No compatible algorithms were recommended.</div>:result.recommendations.map(r=><article className="recommendation" key={r.algorithm_id}><div className="rank">#{r.rank}</div><div className="rec-body"><div className="rec-title"><strong>{r.algorithm_name}</strong><b>{r.score.toFixed(2)}</b></div><p>{r.rationale}</p><div className="evidence">{r.evidence.map(x=><span key={x}>{x}</span>)}</div></div></article>)}</section>
      </>}
    </main><footer>Optimization Lab · AI-first MVP · Steps 1–5 only · No execution / benchmark yet</footer>
  </div>
}
export default App

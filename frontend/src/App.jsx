import { useState, useRef, useEffect } from 'react'
import { Mic, MicOff, Upload, AlertTriangle, CheckCircle, Clock, TrendingUp, MessageSquare, Brain, Zap, Square, Radio, History, ChevronDown, ChevronUp } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

const EMOTION_COLOR = {
  angry:'#ef4444', frustration:'#f97316', sad:'#6366f1', happy:'#22c55e',
  excited:'#eab308', neutral:'#64748b', calm:'#06b6d4', fearful:'#a855f7',
  disgust:'#84cc16', surprised:'#f43f5e'
}
const EMOTION_SCORE = {
  angry:5, frustration:4, fearful:4, disgust:3, sad:3,
  surprised:2, excited:2, happy:1, calm:0, neutral:0
}
const PRIORITY_COLOR = { high:'#ef4444', medium:'#f97316', low:'#22c55e' }

function Card({ title, icon:Icon, children, accent }) {
  return (
    <div style={{ background:'#0e1520', border:`1px solid ${accent||'#1c2a3a'}`, borderRadius:12, padding:'20px 24px' }}>
      {title && (
        <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:16 }}>
          {Icon && <Icon size={15} color="#64748b" />}
          <span style={{ fontSize:11, fontWeight:600, letterSpacing:'0.1em', color:'#64748b', textTransform:'uppercase' }}>{title}</span>
        </div>
      )}
      {children}
    </div>
  )
}

function ConfidenceBar({ value, color }) {
  return (
    <div style={{ background:'#1c2a3a', borderRadius:4, height:6, overflow:'hidden', marginTop:8 }}>
      <div style={{ width:`${Math.round((value||0)*100)}%`, height:'100%', background:color||'#00d4ff', borderRadius:4, transition:'width 0.6s ease' }} />
    </div>
  )
}

async function sendAudioBlob(blob, sessionId) {
  const file = new File([blob], 'recording.wav', { type:'audio/wav' })
  const form = new FormData()
  form.append('file', file)
  form.append('session_id', sessionId)
  const res = await fetch('/analyze', { method:'POST', body:form })
  return res.json()
}

// ── Recorder ───────────────────────────────────
function Recorder({ onResult, loading, setLoading, sessionId }) {
  const [mode, setMode] = useState(null)
  const [recording, setRecording] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [chunkCount, setChunkCount] = useState(0)
  const mediaRef = useRef(null)
  const chunksRef = useRef([])
  const timerRef = useRef(null)
  const chunkIntRef = useRef(null)
  const streamRef = useRef(null)

  useEffect(() => () => stopAll(), [])

  const stopAll = () => {
    clearInterval(timerRef.current)
    clearInterval(chunkIntRef.current)
    if (mediaRef.current && mediaRef.current.state !== 'inactive') mediaRef.current.stop()
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
  }

  const startRecording = async (selectedMode) => {
    setMode(selectedMode); setSeconds(0); setChunkCount(0); chunksRef.current = []
    const stream = await navigator.mediaDevices.getUserMedia({ audio:true })
    streamRef.current = stream
    const mr = new MediaRecorder(stream, { mimeType:'audio/webm' })
    mediaRef.current = mr
    timerRef.current = setInterval(() => setSeconds(s => s+1), 1000)

    if (selectedMode === 'full') {
      mr.ondataavailable = e => chunksRef.current.push(e.data)
      mr.onstop = async () => {
        setLoading(true)
        try { onResult(await sendAudioBlob(new Blob(chunksRef.current, { type:'audio/webm' }), sessionId)) }
        catch (e) { onResult({ error:e.message }) }
        finally { setLoading(false) }
      }
      mr.start()
    } else {
      const runChunk = () => {
        if (mediaRef.current.state === 'recording') { chunksRef.current = []; mediaRef.current.stop() }
        setTimeout(() => {
          const newMr = new MediaRecorder(stream, { mimeType:'audio/webm' })
          mediaRef.current = newMr
          newMr.ondataavailable = e => chunksRef.current.push(e.data)
          newMr.onstop = async () => {
            setChunkCount(c => c+1); setLoading(true)
            try { onResult(await sendAudioBlob(new Blob(chunksRef.current, { type:'audio/webm' }), sessionId)) }
            catch (e) { onResult({ error:e.message }) }
            finally { setLoading(false) }
          }
          newMr.start()
        }, 100)
      }
      mr.ondataavailable = e => chunksRef.current.push(e.data)
      mr.onstop = async () => {
        setChunkCount(c => c+1); setLoading(true)
        try { onResult(await sendAudioBlob(new Blob(chunksRef.current, { type:'audio/webm' }), sessionId)) }
        catch (e) { onResult({ error:e.message }) }
        finally { setLoading(false) }
      }
      mr.start()
      chunkIntRef.current = setInterval(runChunk, 10000)
    }
    setRecording(true)
  }

  const stopRecording = () => { stopAll(); setRecording(false); setMode(null) }
  const fmt = s => `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`

  return (
    <Card title="Live Recording" icon={Mic}>
      {!recording ? (
        <div style={{ display:'flex', flexDirection:'column', gap:10 }}>
          <button onClick={() => startRecording('full')} style={{ display:'flex', alignItems:'center', gap:10, padding:'12px 16px', background:'#080b10', border:'1px solid #1c2a3a', borderRadius:8, color:'#e2e8f0', cursor:'pointer', fontSize:13, textAlign:'left', width:'100%' }}>
            <Mic size={16} color="#00d4ff" />
            <div><div style={{ fontWeight:600 }}>Record then Analyze</div><div style={{ fontSize:11, color:'#64748b', marginTop:2 }}>Analyze after you stop recording</div></div>
          </button>
          <button onClick={() => startRecording('chunk')} style={{ display:'flex', alignItems:'center', gap:10, padding:'12px 16px', background:'#080b10', border:'1px solid #1c2a3a', borderRadius:8, color:'#e2e8f0', cursor:'pointer', fontSize:13, textAlign:'left', width:'100%' }}>
            <Radio size={16} color="#f97316" />
            <div><div style={{ fontWeight:600 }}>Chunk-based (every 10s)</div><div style={{ fontSize:11, color:'#64748b', marginTop:2 }}>Auto-analyzes while you speak</div></div>
          </button>
        </div>
      ) : (
        <div style={{ textAlign:'center', padding:'8px 0' }}>
          <div style={{ display:'flex', alignItems:'center', justifyContent:'center', gap:8, marginBottom:12 }}>
            <div style={{ width:10, height:10, borderRadius:'50%', background:'#ef4444', animation:'pulse 1s infinite' }} />
            <span style={{ fontSize:13, color:'#ef4444', fontWeight:600 }}>{mode==='chunk' ? 'CHUNK MODE' : 'RECORDING'}</span>
          </div>
          <div style={{ fontFamily:'Space Mono, monospace', fontSize:32, color:'#e2e8f0', marginBottom:8 }}>{fmt(seconds)}</div>
          {mode === 'chunk' && <div style={{ fontSize:12, color:'#64748b', marginBottom:12 }}>Chunks: {chunkCount} · Next in {10-(seconds%10)}s</div>}
          <button onClick={stopRecording} style={{ display:'flex', alignItems:'center', gap:8, margin:'0 auto', padding:'10px 20px', background:'#ef4444', border:'none', borderRadius:8, color:'#fff', fontWeight:700, cursor:'pointer', fontSize:13 }}>
            <Square size={14} /> Stop
          </button>
        </div>
      )}
      <style>{`@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.3)} }`}</style>
    </Card>
  )
}

// ── Uploader ───────────────────────────────────
function Uploader({ onResult, loading, setLoading, sessionId }) {
  const fileRef = useRef()
  const [fileName, setFileName] = useState(null)
  const [file, setFile] = useState(null)
  const [drag, setDrag] = useState(false)
  const handleFile = f => { setFile(f); setFileName(f.name) }
  const handleSubmit = async () => {
    if (!file) return; setLoading(true)
    const form = new FormData(); form.append('file', file); form.append('session_id', sessionId)
    try { onResult(await (await fetch('/analyze', { method:'POST', body:form })).json()) }
    catch (e) { onResult({ error:e.message }) }
    finally { setLoading(false) }
  }
  return (
    <Card title="Upload Audio" icon={Upload}>
      <div onDragOver={e => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
        onDrop={e => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]) }}
        onClick={() => fileRef.current.click()}
        style={{ border:`2px dashed ${drag?'#00d4ff':'#1c2a3a'}`, borderRadius:10, padding:'24px 20px', textAlign:'center', cursor:'pointer', marginBottom:12 }}>
        <Upload size={22} color={drag?'#00d4ff':'#64748b'} style={{ margin:'0 auto 8px' }} />
        <p style={{ color:fileName?'#e2e8f0':'#64748b', fontSize:13 }}>{fileName || 'Drop .wav / .mp3 here or click'}</p>
        <input ref={fileRef} type="file" accept=".wav,.mp3,.m4a,.ogg,.flac" style={{ display:'none' }} onChange={e => handleFile(e.target.files[0])} />
      </div>
      <button onClick={handleSubmit} disabled={!file||loading} style={{
        width:'100%', padding:'11px 0', background:(!file||loading)?'#1c2a3a':'#00d4ff',
        color:(!file||loading)?'#64748b':'#080b10', border:'none', borderRadius:8, fontWeight:700, fontSize:13, cursor:(!file||loading)?'not-allowed':'pointer'
      }}>{loading ? 'Analyzing...' : 'Run Analysis'}</button>
    </Card>
  )
}

// ── Timeline ───────────────────────────────────
function EmotionTimeline({ history }) {
  if (!history || history.length === 0) return null
  const data = history.map((h,i) => ({ turn:`T${i+1}`, score:EMOTION_SCORE[h.emotion]??0, emotion:h.emotion }))
  return (
    <Card title="Emotion Timeline" icon={TrendingUp}>
      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={data}>
          <XAxis dataKey="turn" stroke="#1c2a3a" tick={{ fill:'#64748b', fontSize:11 }} />
          <YAxis hide domain={[0,5]} />
          <Tooltip contentStyle={{ background:'#0e1520', border:'1px solid #1c2a3a', borderRadius:8, fontSize:12 }}
            formatter={(v,n,p) => [p.payload.emotion, 'emotion']} />
          <Line type="monotone" dataKey="score" stroke="#00d4ff" strokeWidth={2} dot={{ fill:'#00d4ff', r:4 }} activeDot={{ r:6 }} />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  )
}

// ── Session History Panel ──────────────────────
function SessionHistory() {
  const [open, setOpen] = useState(false)
  const [sessions, setSessions] = useState([])
  const [selectedHistory, setSelectedHistory] = useState(null)
  const [selectedId, setSelectedId] = useState(null)

  const loadSessions = async () => {
    try { setSessions(await (await fetch('/sessions')).json()) }
    catch { setSessions([]) }
  }

  const loadHistory = async (sid) => {
    setSelectedId(sid)
    try { setSelectedHistory(await (await fetch(`/history/${sid}`)).json()) }
    catch { setSelectedHistory([]) }
  }

  useEffect(() => { if (open) loadSessions() }, [open])

  return (
    <Card title="Session History" icon={History}>
      <button onClick={() => setOpen(!open)} style={{
        display:'flex', alignItems:'center', justifyContent:'space-between', width:'100%',
        padding:'10px 14px', background:'#080b10', border:'1px solid #1c2a3a', borderRadius:8,
        color:'#e2e8f0', cursor:'pointer', fontSize:13
      }}>
        <span>{open ? 'Hide Sessions' : 'View Past Sessions'}</span>
        {open ? <ChevronUp size={14}/> : <ChevronDown size={14}/>}
      </button>

      {open && (
        <div style={{ marginTop:12 }}>
          {sessions.length === 0 && <p style={{ color:'#64748b', fontSize:12 }}>No sessions yet</p>}

          {sessions.map(s => (
            <div key={s.session_id} onClick={() => loadHistory(s.session_id)} style={{
              padding:'10px 14px', margin:'6px 0', background: selectedId===s.session_id ? '#1c2a3a' : '#080b10',
              border:'1px solid #1c2a3a', borderRadius:8, cursor:'pointer', transition:'background 0.2s'
            }}>
              <div style={{ display:'flex', justifyContent:'space-between', fontSize:13 }}>
                <span style={{ color:'#e2e8f0', fontFamily:'Space Mono, monospace' }}>{s.session_id}</span>
                <span style={{ color:'#64748b', fontSize:11 }}>{s.turns} turns</span>
              </div>
              <div style={{ fontSize:11, color:'#64748b', marginTop:4 }}>{s.last_active}</div>
            </div>
          ))}

          {selectedHistory && (
            <div style={{ marginTop:16, borderTop:'1px solid #1c2a3a', paddingTop:12 }}>
              <div style={{ fontSize:11, color:'#64748b', marginBottom:10, textTransform:'uppercase', letterSpacing:'0.1em' }}>
                Conversation Flow
              </div>
              {selectedHistory.map((turn, i) => (
                <div key={i} style={{ display:'flex', gap:12, marginBottom:12 }}>
                  <div style={{ display:'flex', flexDirection:'column', alignItems:'center' }}>
                    <div style={{ width:8, height:8, borderRadius:'50%', background:EMOTION_COLOR[turn.emotion]||'#64748b' }} />
                    {i < selectedHistory.length-1 && <div style={{ width:1, flex:1, background:'#1c2a3a', marginTop:4 }} />}
                  </div>
                  <div style={{ flex:1, paddingBottom:8 }}>
                    <div style={{ display:'flex', justifyContent:'space-between', marginBottom:4 }}>
                      <span style={{ fontSize:12, fontWeight:600, color:EMOTION_COLOR[turn.emotion]||'#64748b', textTransform:'capitalize' }}>
                        {turn.emotion}
                      </span>
                      <span style={{ fontSize:11, color:'#64748b' }}>
                        Risk: {Math.round((turn.escalation_risk||0)*100)}%
                      </span>
                    </div>
                    <p style={{ fontSize:12, color:'#94a3b8', lineHeight:1.5 }}>{turn.transcript}</p>
                    <span style={{ fontSize:10, color:'#1c2a3a' }}>{turn.intent} · {turn.timestamp}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  )
}

// ── Main App ───────────────────────────────────
export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState([])
  const [sessionId] = useState('session_001')

  const handleResult = data => {
    setResult(data)
    if (data.emotion?.emotion) {
      setHistory(prev => [...prev, { emotion:data.emotion.emotion, escalation_risk:data.escalation_risk, intent:data.intent?.intent }])
    }
  }

  const emotion = result?.emotion
  const intent  = result?.intent
  const eColor  = EMOTION_COLOR[emotion?.emotion] || '#64748b'
  const pColor  = PRIORITY_COLOR[result?.priority] || '#64748b'

  return (
    <div style={{ minHeight:'100vh', background:'#080b10' }}>
      <div style={{ borderBottom:'1px solid #1c2a3a', padding:'18px 32px', display:'flex', alignItems:'center', gap:12 }}>
        <Zap size={20} color="#00d4ff" />
        <span style={{ fontFamily:'Space Mono, monospace', fontWeight:700, fontSize:16, color:'#e2e8f0' }}>AURALIS AI</span>
        <span style={{ fontSize:12, color:'#64748b', marginLeft:4 }}>Speech Intelligence System</span>
        <div style={{ marginLeft:'auto', display:'flex', alignItems:'center', gap:6 }}>
          <div style={{ width:7, height:7, borderRadius:'50%', background:'#22c55e' }} />
          <span style={{ fontSize:12, color:'#64748b' }}>Live</span>
        </div>
      </div>

      <div style={{ maxWidth:1100, margin:'0 auto', padding:'32px 24px', display:'grid', gridTemplateColumns:'340px 1fr', gap:20 }}>
        <div style={{ display:'flex', flexDirection:'column', gap:20 }}>
          <Recorder onResult={handleResult} loading={loading} setLoading={setLoading} sessionId={sessionId} />
          <Uploader onResult={handleResult} loading={loading} setLoading={setLoading} sessionId={sessionId} />
          <EmotionTimeline history={history} />
          <SessionHistory />
        </div>

        <div style={{ display:'flex', flexDirection:'column', gap:20 }}>
          {!result && !loading && (
            <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:300, color:'#64748b', fontSize:14, flexDirection:'column', gap:12 }}>
              <Brain size={36} color="#1c2a3a" /><span>Record or upload audio to begin analysis</span>
            </div>
          )}
          {loading && (
            <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:300, color:'#64748b', fontSize:14, flexDirection:'column', gap:12 }}>
              <Clock size={36} color="#00d4ff" /><span>Running pipeline...</span>
              <span style={{ fontSize:12, color:'#1c2a3a' }}>ASR → Emotion + Intent → Memory → Action</span>
            </div>
          )}
          {result && !loading && (
            <>
              <Card title="Transcript" icon={MessageSquare}>
                <p style={{ fontFamily:'Space Mono, monospace', fontSize:13, lineHeight:1.7, color:result.transcript?'#e2e8f0':'#64748b' }}>
                  {result.transcript || 'No transcript — audio confidence too low'}
                </p>
                {result.asr_confidence !== null && (
                  <div style={{ marginTop:12 }}>
                    <div style={{ display:'flex', justifyContent:'space-between', fontSize:12, color:'#64748b' }}>
                      <span>ASR Confidence</span><span style={{ fontFamily:'Space Mono, monospace' }}>{Math.round((result.asr_confidence||0)*100)}%</span>
                    </div>
                    <ConfidenceBar value={result.asr_confidence||0} color="#00d4ff" />
                  </div>
                )}
              </Card>

              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:20 }}>
                <Card title="Emotion" icon={Brain} accent={emotion ? eColor+'44' : undefined}>
                  {emotion ? (<>
                    <div style={{ display:'flex', alignItems:'baseline', gap:10 }}>
                      <span style={{ fontSize:28, fontWeight:700, color:eColor, fontFamily:'Space Mono, monospace', textTransform:'capitalize' }}>{emotion.emotion}</span>
                      <span style={{ fontSize:12, color:'#64748b' }}>intensity {Math.round(emotion.intensity*100)}%</span>
                    </div>
                    <ConfidenceBar value={emotion.confidence} color={eColor} />
                    <div style={{ marginTop:14, display:'flex', flexDirection:'column', gap:6 }}>
                      {[['Pitch',`${emotion.pitch_mean?.toFixed(0)} Hz`],['Energy',emotion.energy_mean?.toFixed(4)],['Speaking Rate',emotion.speaking_rate?.toFixed(2)],['Confidence',`${Math.round(emotion.confidence*100)}%`]].map(([k,v])=>(
                        <div key={k} style={{ display:'flex', justifyContent:'space-between', fontSize:12 }}>
                          <span style={{ color:'#64748b' }}>{k}</span><span style={{ fontFamily:'Space Mono, monospace', color:'#e2e8f0' }}>{v}</span>
                        </div>
                      ))}
                    </div>
                    {result.emotion_retry_count > 0 && <div style={{ marginTop:12, fontSize:11, color:'#f97316' }}>↺ Reflection loop ran {result.emotion_retry_count}x</div>}
                  </>) : <span style={{ color:'#64748b', fontSize:13 }}>Not analyzed</span>}
                </Card>

                <Card title="Intent">
                  {intent ? (<>
                    <span style={{ fontSize:28, fontWeight:700, color:'#00d4ff', fontFamily:'Space Mono, monospace', textTransform:'capitalize' }}>{intent.intent}</span>
                    <ConfidenceBar value={intent.confidence} color="#00d4ff" />
                    <p style={{ marginTop:14, fontSize:13, color:'#94a3b8', lineHeight:1.6 }}>{intent.summary}</p>
                    <div style={{ marginTop:10, fontSize:12, display:'flex', justifyContent:'space-between' }}>
                      <span style={{ color:'#64748b' }}>Confidence</span><span style={{ fontFamily:'Space Mono, monospace' }}>{Math.round(intent.confidence*100)}%</span>
                    </div>
                  </>) : <span style={{ color:'#64748b', fontSize:13 }}>Not analyzed</span>}
                </Card>
              </div>

              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:20 }}>
                <Card title="Recommended Action">
                  <div style={{ display:'flex', alignItems:'center', gap:12 }}>
                    {result.action==='escalate' ? <AlertTriangle size={28} color="#ef4444"/> : <CheckCircle size={28} color="#22c55e"/>}
                    <div>
                      <div style={{ fontSize:20, fontWeight:700, fontFamily:'Space Mono, monospace', color:result.action==='escalate'?'#ef4444':'#22c55e', textTransform:'uppercase' }}>{result.action}</div>
                      <div style={{ fontSize:12, marginTop:4 }}>
                        <span style={{ color:'#64748b' }}>Priority: </span>
                        <span style={{ color:pColor, fontWeight:600, textTransform:'uppercase' }}>{result.priority}</span>
                      </div>
                    </div>
                  </div>
                  {result.response && <p style={{ marginTop:16, fontSize:13, color:'#94a3b8', lineHeight:1.6, borderTop:'1px solid #1c2a3a', paddingTop:14 }}>"{result.response}"</p>}
                </Card>

                <Card title="Escalation Risk">
                  <div style={{ textAlign:'center', padding:'8px 0' }}>
                    <div style={{ fontSize:48, fontWeight:700, fontFamily:'Space Mono, monospace',
                      color:result.escalation_risk>0.7?'#ef4444':result.escalation_risk>0.4?'#f97316':'#22c55e' }}>
                      {Math.round((result.escalation_risk||0)*100)}<span style={{ fontSize:20 }}>%</span>
                    </div>
                    <ConfidenceBar value={result.escalation_risk||0} color={result.escalation_risk>0.7?'#ef4444':result.escalation_risk>0.4?'#f97316':'#22c55e'} />
                    <p style={{ marginTop:10, fontSize:12, color:'#64748b' }}>Session turns analyzed: {history.length}</p>
                  </div>
                </Card>
              </div>

              {result.error && <Card title="Error" accent="#ef444444"><p style={{ fontFamily:'Space Mono, monospace', fontSize:12, color:'#ef4444' }}>{result.error}</p></Card>}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
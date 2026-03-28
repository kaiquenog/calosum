import { useCallback, useEffect, useState, useMemo } from 'react';
import { Activity, Brain, CheckCircle, Clock, Zap, History, MessageSquare, ChevronDown, ChevronRight, Terminal } from 'lucide-react';
import './App.css';

interface DashboardEvent {
  _session_id?: string;
  _recorded_at?: string;
  context_id?: string;
  emotional_labels?: string[];
  salience?: number;
  world_hypotheses?: Record<string, number>;
  lambda_signature?: string;
  reasoning_summary?: string[];
  response_text?: string;
  action_types?: string[];
  runtime_retry_count?: number;
  runtime_rejected_count?: number;
}

interface ExecutionResult {
  action_type: string;
  status: string;
  output: Record<string, unknown>;
  violations: string[];
}

interface ExecutionEvent {
  _session_id?: string;
  _recorded_at?: string;
  results: ExecutionResult[];
}

interface ReflectionScore {
  variant_id: string;
  score: number;
  reasons: string[];
}

interface ReflectionEvent {
  _session_id?: string;
  _recorded_at?: string;
  selected_variant_id?: string;
  scoreboard?: ReflectionScore[];
  notes?: string[];
  bridge_adjustments?: Record<string, unknown>;
}

interface Dashboard {
  felt: DashboardEvent[];
  thought: DashboardEvent[];
  decision: DashboardEvent[];
  execution: ExecutionEvent[];
  reflection: ReflectionEvent[];
}

// Normalized event for timeline display
interface TimelineEvent {
  id: string;
  type: 'felt' | 'thought' | 'decision' | 'execution' | 'reflection';
  timestamp: string;
  sessionId: string;
  data: any;
}

function App() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSession, setSelectedSession] = useState<string>('all');
  const [availableSessions, setAvailableSessions] = useState<string[]>([]);
  const apiBase = import.meta.env.VITE_CALOSUM_API_BASE ?? 'http://localhost:8000';

  const extractSessions = useCallback((data: Dashboard) => {
    const sessions = new Set<string>();
    
    // Extract from all channels
    Object.values(data).forEach((channelEvents: any[]) => {
      channelEvents.forEach(event => {
        if (event._session_id) {
          sessions.add(event._session_id);
        }
      });
    });
    
    setAvailableSessions(Array.from(sessions).sort());
  }, []);

  const fetchDashboard = useCallback(async (silent = false) => {
    if (!silent) {
      setLoading(true);
    }
    setError(null);
    try {
      const response = await fetch(`${apiBase}/v1/telemetry/dashboard`);
      const data = await response.json();
      if (data.status === 'ok') {
        setDashboard(data.dashboard);
        extractSessions(data.dashboard);
      } else {
        setError(data.error || 'Erro ao carregar dashboard');
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Erro de conexão');
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }, [apiBase, extractSessions]);

  useEffect(() => {
    void fetchDashboard(false);
    const intervalId = window.setInterval(() => {
      void fetchDashboard(true);
    }, 2500);

    return () => window.clearInterval(intervalId);
  }, [fetchDashboard]);

  // Helper to filter events by selected session
  const filterEvents = useCallback(<T extends { _session_id?: string }>(events: T[]): T[] => {
    if (selectedSession === 'all') return events;
    return events.filter(e => e._session_id === selectedSession);
  }, [selectedSession]);

  const timelineEvents = useMemo(() => {
    if (!dashboard) return [];
    
    const events: TimelineEvent[] = [];
    
    const processChannel = (channel: keyof Dashboard, dataArray: any[]) => {
      filterEvents(dataArray).forEach((item, index) => {
        events.push({
          id: `${channel}-${index}-${item._recorded_at || Date.now()}`,
          type: channel as any,
          timestamp: item._recorded_at || new Date().toISOString(),
          sessionId: item._session_id || 'unknown',
          data: item
        });
      });
    };

    processChannel('felt', dashboard.felt);
    processChannel('thought', dashboard.thought);
    processChannel('decision', dashboard.decision);
    processChannel('execution', dashboard.execution);
    processChannel('reflection', dashboard.reflection);

    // Sort by timestamp descending (newest first)
    return events.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  }, [dashboard, filterEvents]);

  // Group events by session for a clearer view
  const groupedEvents = useMemo(() => {
    const groups: Record<string, TimelineEvent[]> = {};
    timelineEvents.forEach(event => {
      if (!groups[event.sessionId]) {
        groups[event.sessionId] = [];
      }
      groups[event.sessionId].push(event);
    });
    return groups;
  }, [timelineEvents]);

  const [expandedSessions, setExpandedSessions] = useState<Record<string, boolean>>({});

  const toggleSession = (sessionId: string) => {
    setExpandedSessions(prev => ({
      ...prev,
      [sessionId]: !prev[sessionId]
    }));
  };

  // Initialize expanded state for new sessions
  useEffect(() => {
    if (Object.keys(groupedEvents).length > 0) {
      setExpandedSessions(prev => {
        const next = { ...prev };
        let changed = false;
        Object.keys(groupedEvents).forEach(sid => {
          if (next[sid] === undefined) {
            next[sid] = true; // Auto-expand by default
            changed = true;
          }
        });
        return changed ? next : prev;
      });
    }
  }, [groupedEvents]);

  const renderEventIcon = (type: TimelineEvent['type']) => {
    switch (type) {
      case 'felt': return <Zap className="w-5 h-5 text-yellow-400" />;
      case 'thought': return <Brain className="w-5 h-5 text-emerald-400" />;
      case 'decision': return <CheckCircle className="w-5 h-5 text-purple-400" />;
      case 'execution': return <Activity className="w-5 h-5 text-cyan-400" />;
      case 'reflection': return <History className="w-5 h-5 text-amber-300" />;
    }
  };

  const renderEventColor = (type: TimelineEvent['type']) => {
    switch (type) {
      case 'felt': return 'border-yellow-400/30 bg-yellow-400/5';
      case 'thought': return 'border-emerald-400/30 bg-emerald-400/5';
      case 'decision': return 'border-purple-400/30 bg-purple-400/5';
      case 'execution': return 'border-cyan-400/30 bg-cyan-400/5';
      case 'reflection': return 'border-amber-300/30 bg-amber-300/5';
    }
  };

  const renderEventContent = (event: TimelineEvent) => {
    const { type, data } = event;

    switch (type) {
      case 'felt':
        return (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-2">
                <span className="text-xs text-gray-500 uppercase tracking-wider">Saliência</span>
                <span className="text-sm font-mono text-yellow-300">{data.salience?.toFixed(3) || 'N/A'}</span>
              </div>
              {data.emotional_labels && data.emotional_labels.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 uppercase tracking-wider">Emoções</span>
                  <div className="flex gap-1">
                    {data.emotional_labels.map((l: string, i: number) => (
                      <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-yellow-500/20 text-yellow-200 border border-yellow-500/30">
                        {l}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
            {data.world_hypotheses && Object.keys(data.world_hypotheses).length > 0 && (
              <div className="bg-gray-900/50 rounded-md p-3 border border-gray-800">
                <span className="text-xs text-gray-500 uppercase tracking-wider block mb-2">Hipóteses de Mundo</span>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  {Object.entries(data.world_hypotheses).map(([k, v]) => (
                    <div key={k} className="flex flex-col">
                      <span className="text-xs text-gray-400 truncate" title={k}>{k.replace(/_/g, ' ')}</span>
                      <span className="text-sm font-mono text-gray-200">{(v as number).toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 'thought':
        return (
          <div className="space-y-3">
            {data.lambda_signature && (
              <div>
                <span className="text-xs text-gray-500 uppercase tracking-wider block mb-1">Programa Lambda</span>
                <code className="text-xs text-emerald-300 bg-gray-950 p-2 rounded-md block border border-gray-800 break-all">
                  {data.lambda_signature}
                </code>
              </div>
            )}
            {data.reasoning_summary && data.reasoning_summary.length > 0 && (
              <div>
                <span className="text-xs text-gray-500 uppercase tracking-wider block mb-2">Trilha de Raciocínio</span>
                <ul className="space-y-1.5">
                  {data.reasoning_summary.map((step: string, i: number) => (
                    <li key={i} className="text-sm text-gray-300 flex items-start gap-2 bg-gray-900/30 p-2 rounded border border-gray-800/50">
                      <span className="text-emerald-500 mt-0.5">▹</span>
                      <span>{step}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        );

      case 'decision':
        return (
          <div className="space-y-3">
            {data.response_text && (
              <div>
                <span className="text-xs text-gray-500 uppercase tracking-wider block mb-1">Resposta Sintetizada</span>
                <div className="bg-gray-950 border border-gray-800 rounded-md p-3 text-sm text-gray-200 whitespace-pre-wrap flex items-start gap-3">
                  <MessageSquare className="w-4 h-4 text-purple-400 mt-0.5 shrink-0" />
                  <span>{data.response_text}</span>
                </div>
              </div>
            )}
            <div className="flex flex-wrap gap-4 items-center">
              {data.action_types && data.action_types.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 uppercase tracking-wider">Ações</span>
                  <div className="flex gap-1">
                    {data.action_types.map((a: string, i: number) => (
                      <span key={i} className="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-200 border border-purple-500/30">
                        {a}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {((data.runtime_retry_count ?? 0) > 0 || (data.runtime_rejected_count ?? 0) > 0) && (
                <div className="flex gap-3 text-xs">
                  {(data.runtime_retry_count ?? 0) > 0 && (
                    <span className="flex items-center gap-1 text-yellow-500 bg-yellow-500/10 px-2 py-1 rounded border border-yellow-500/20">
                      <Clock className="w-3 h-3" /> Retries: {data.runtime_retry_count}
                    </span>
                  )}
                  {(data.runtime_rejected_count ?? 0) > 0 && (
                    <span className="flex items-center gap-1 text-red-400 bg-red-500/10 px-2 py-1 rounded border border-red-500/20">
                      <Zap className="w-3 h-3" /> Rejeições: {data.runtime_rejected_count}
                    </span>
                  )}
                </div>
              )}
            </div>
          </div>
        );

      case 'execution':
        return (
          <div className="space-y-2">
            {data.results && data.results.length > 0 ? (
              data.results.map((result: any, i: number) => (
                <div key={i} className="bg-gray-950 border border-gray-800 rounded-md p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-200 flex items-center gap-2">
                      <Terminal className="w-4 h-4 text-cyan-500" />
                      {result.action_type}
                    </span>
                    <span className={`text-xs px-2 py-0.5 rounded uppercase tracking-wider ${
                      result.status === 'executed' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' : 
                      'bg-red-500/20 text-red-300 border border-red-500/30'
                    }`}>
                      {result.status}
                    </span>
                  </div>
                  {result.violations && result.violations.length > 0 && (
                    <div className="mb-2 p-2 bg-red-950/50 border border-red-900/50 rounded">
                      <span className="text-xs text-red-400 uppercase tracking-wider block mb-1">Violações</span>
                      <ul className="text-xs text-red-300 space-y-1">
                        {result.violations.map((v: string, j: number) => <li key={j}>• {v}</li>)}
                      </ul>
                    </div>
                  )}
                  {result.output && Object.keys(result.output).length > 0 && (
                    <div className="mt-2">
                      <span className="text-xs text-gray-500 uppercase tracking-wider block mb-1">Output</span>
                      <pre className="text-xs text-gray-400 whitespace-pre-wrap break-all bg-gray-900 p-2 rounded border border-gray-800">
                        {JSON.stringify(result.output, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              ))
            ) : (
              <span className="text-sm text-gray-500 italic">Nenhuma ação executada</span>
            )}
          </div>
        );

      case 'reflection':
        return (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500 uppercase tracking-wider">Variante Escolhida</span>
              <span className="text-sm font-medium text-amber-300 bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
                {data.selected_variant_id ?? 'N/A'}
              </span>
            </div>
            
            {data.notes && data.notes.length > 0 && (
              <div>
                <span className="text-xs text-gray-500 uppercase tracking-wider block mb-1">Notas de Reflexão</span>
                <ul className="space-y-1 text-sm text-gray-300 bg-gray-900/50 p-3 rounded border border-gray-800">
                  {data.notes.map((note: string, i: number) => (
                    <li key={i} className="flex gap-2"><span className="text-amber-500/50">•</span>{note}</li>
                  ))}
                </ul>
              </div>
            )}

            {data.scoreboard && data.scoreboard.length > 0 && (
              <div>
                <span className="text-xs text-gray-500 uppercase tracking-wider block mb-2">Scoreboard</span>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {data.scoreboard.map((score: any) => (
                    <div key={score.variant_id} className="bg-gray-950 border border-gray-800 rounded-md p-3">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium text-gray-200">{score.variant_id}</span>
                        <span className="text-sm font-mono text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded">
                          {score.score.toFixed(3)}
                        </span>
                      </div>
                      <ul className="text-xs text-gray-400 space-y-1 mt-2 border-t border-gray-800/50 pt-2">
                        {score.reasons.map((reason: string, i: number) => (
                          <li key={i} className="truncate" title={reason}>- {reason}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      
      default:
        return <pre className="text-xs text-gray-500">{JSON.stringify(data, null, 2)}</pre>;
    }
  };

  const formatTime = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString(undefined, { 
        hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3 
      });
    } catch {
      return 'N/A';
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-8 font-sans">
      <header className="mb-8 flex items-center justify-between border-b border-gray-800 pb-6">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent flex items-center gap-3">
            <Brain className="w-8 h-8 text-blue-400" />
            Calosum Telemetry
          </h1>
          <p className="text-gray-400 mt-2">Monitoramento de Dualidade Cognitiva (Global)</p>
        </div>
        
        <div className="flex gap-4 items-center">
          <div className="flex items-center gap-2">
            <label htmlFor="session-select" className="text-sm text-gray-400">Sessão:</label>
            <select 
              id="session-select"
              value={selectedSession} 
              onChange={(e) => setSelectedSession(e.target.value)}
              className="bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
            >
              <option value="all">Todas as sessões</option>
              {availableSessions.map(session => (
                <option key={session} value={session}>{session}</option>
              ))}
            </select>
          </div>

          <button 
            onClick={() => void fetchDashboard(false)}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2"
          >
            <Activity className="w-4 h-4" />
            {loading ? 'Atualizando...' : 'Atualizar'}
          </button>
        </div>
      </header>

      {error && (
        <div className="bg-red-900/50 border border-red-500 text-red-200 p-4 rounded-md mb-8">
          {error}
        </div>
      )}

      {!dashboard || Object.keys(groupedEvents).length === 0 ? (
        <div className="text-center text-gray-500 py-20 bg-gray-900/30 rounded-xl border border-gray-800 border-dashed">
          <Activity className="w-12 h-12 text-gray-700 mx-auto mb-4" />
          Nenhum dado de telemetria encontrado. Execute interações no terminal ou API.
        </div>
      ) : (
        <div className="space-y-8">
          {Object.entries(groupedEvents).map(([sessionId, events]) => (
            <div key={sessionId} className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden shadow-xl transition-all">
              {/* Session Header */}
              <button 
                onClick={() => toggleSession(sessionId)}
                className="w-full bg-gradient-to-r from-slate-900 to-gray-900 p-4 border-b border-gray-800 flex items-center justify-between hover:bg-slate-800/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className={`p-1 rounded-md ${expandedSessions[sessionId] ? 'bg-blue-500/20 text-blue-400' : 'bg-gray-800 text-gray-500'}`}>
                    {expandedSessions[sessionId] ? <ChevronDown className="w-5 h-5" /> : <ChevronRight className="w-5 h-5" />}
                  </div>
                  <div>
                    <h2 className="font-semibold text-lg text-gray-200 flex items-center gap-2">
                      Sessão: <span className="font-mono text-blue-400">{sessionId}</span>
                    </h2>
                    <span className="text-xs text-gray-500">{events.length} eventos registrados</span>
                  </div>
                </div>
              </button>

              {/* Timeline Container */}
              {expandedSessions[sessionId] && (
                <div className="p-6">
                  <div className="relative pl-8 border-l border-gray-800 space-y-8 before:absolute before:inset-0 before:ml-[31px] before:-translate-x-px md:before:mx-auto md:before:translate-x-0">
                    {events.map((event, index) => (
                      <div key={event.id} className="relative group">
                        {/* Timeline dot */}
                        <div className={`absolute -left-[41px] mt-1.5 w-6 h-6 rounded-full border-2 bg-gray-950 flex items-center justify-center ${renderEventColor(event.type).split(' ')[0]}`}>
                          <div className="scale-75">
                            {renderEventIcon(event.type)}
                          </div>
                        </div>

                        {/* Event Card */}
                        <div className={`bg-gray-800/30 rounded-xl border p-5 transition-colors hover:bg-gray-800/50 ${renderEventColor(event.type)}`}>
                          <div className="flex justify-between items-start mb-4 pb-3 border-b border-gray-700/50">
                            <div className="flex items-center gap-2">
                              <span className="font-semibold uppercase tracking-widest text-sm text-gray-200">
                                {event.type}
                              </span>
                              {event.data.turn_id && (
                                <span className="text-[10px] font-mono text-gray-500 bg-gray-950 px-2 py-0.5 rounded border border-gray-800">
                                  {event.data.turn_id.split('-')[0]}
                                </span>
                              )}
                            </div>
                            <span className="text-xs font-mono text-gray-500 flex items-center gap-1">
                              <Clock className="w-3 h-3" />
                              {formatTime(event.timestamp)}
                            </span>
                          </div>
                          
                          {renderEventContent(event)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default App;

import { useCallback, useEffect, useState, useMemo, useRef } from 'react';
import { Activity, Brain, CheckCircle, Clock, Zap, History, MessageSquare, ChevronDown, ChevronRight, Terminal, Send, Bot, User } from 'lucide-react';
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

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  status?: 'sending' | 'processing' | 'done' | 'error';
}

function App() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSession, setSelectedSession] = useState<string>('all');
  const [availableSessions, setAvailableSessions] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'chat' | 'history'>('chat');
  
  // Chat state
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);
  
  // Use a fixed active session for the chat
  const activeSessionId = useMemo(() => {
    return `chat-session-${new Date().toISOString().split('T')[0]}`;
  }, []);

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
  const filterEvents = useCallback(<T extends { _session_id?: string }>(events: T[], forceSession?: string): T[] => {
    const targetSession = forceSession || selectedSession;
    if (targetSession === 'all') return events;
    return events.filter(e => e._session_id === targetSession);
  }, [selectedSession]);

  const getTimelineEvents = useCallback((forceSession?: string) => {
    if (!dashboard) return [];
    
    const events: TimelineEvent[] = [];
    
    const processChannel = (channel: keyof Dashboard, dataArray: any[]) => {
      filterEvents(dataArray, forceSession).forEach((item, index) => {
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

  const timelineEvents = useMemo(() => getTimelineEvents(), [getTimelineEvents]);
  
  const liveTimelineEvents = useMemo(() => getTimelineEvents(activeSessionId), [getTimelineEvents, activeSessionId]);
  const liveEventsDesc = useMemo(() => {
    return [...liveTimelineEvents].sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );
  }, [liveTimelineEvents]);

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

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [chatMessages]);

  // Track if chat is scrolled near bottom to show a quick affordance
  useEffect(() => {
    const el = chatScrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24;
      setIsAtBottom(nearBottom);
    };
    el.addEventListener('scroll', onScroll, { passive: true } as any);
    return () => el.removeEventListener('scroll', onScroll as any);
  }, []);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || isProcessing) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: chatInput.trim(),
      timestamp: new Date(),
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatInput('');
    setIsProcessing(true);

    const botMessageId = `bot-${Date.now()}`;
    setChatMessages(prev => [...prev, {
      id: botMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      status: 'processing'
    }]);

    try {
      // Connect to SSE endpoint
      const response = await fetch(`${apiBase}/v1/chat/sse?text=${encodeURIComponent(userMessage.content)}&session_id=${activeSessionId}`);
      
      if (!response.ok) {
        throw new Error('Failed to connect to chat API');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let botContent = '';

      if (reader) {
        while (!done) {
          const { value, done: readerDone } = await reader.read();
          done = readerDone;
          
          if (value) {
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n');
            
            let currentEvent = '';
            
            for (const line of lines) {
              if (line.startsWith('event: ')) {
                currentEvent = line.replace('event: ', '').trim();
              } else if (line.startsWith('data: ')) {
                const data = line.replace('data: ', '');
                
                if (currentEvent === 'reasoning') {
                  botContent += data + '\n';
                  setChatMessages(prev => prev.map(msg => 
                    msg.id === botMessageId 
                      ? { ...msg, content: botContent } 
                      : msg
                  ));
                } else if (currentEvent === 'action') {
                  // Only append if we haven't received reasoning yet
                  if (!botContent) {
                    setChatMessages(prev => prev.map(msg => 
                      msg.id === botMessageId 
                        ? { ...msg, content: `*[Executando Ação: ${data}]*` } 
                        : msg
                    ));
                  }
                } else if (currentEvent === 'error') {
                  setChatMessages(prev => prev.map(msg => 
                    msg.id === botMessageId 
                      ? { ...msg, content: `Erro: ${data}`, status: 'error' } 
                      : msg
                  ));
                }
              }
            }
          }
        }
      }

      setChatMessages(prev => prev.map(msg => 
        msg.id === botMessageId 
          ? { ...msg, status: 'done', content: botContent || msg.content } 
          : msg
      ));
      
      // Force dashboard refresh to get the latest telemetry
      void fetchDashboard(true);

    } catch (err) {
      console.error('Chat error:', err);
      setChatMessages(prev => prev.map(msg => 
        msg.id === botMessageId 
          ? { ...msg, status: 'error', content: 'Erro de conexão com o agente.' } 
          : msg
      ));
    } finally {
      setIsProcessing(false);
    }
  };

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

  const [historyView, setHistoryView] = useState<'timeline' | 'board3'>('board3');

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 flex flex-col font-sans">
      <header className="flex-none bg-gray-900 border-b border-gray-800 p-4 px-6 flex items-center justify-between shadow-sm z-10">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-3">
            <div className="bg-blue-500/10 p-2 rounded-lg border border-blue-500/20">
              <Brain className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
                Calosum
              </h1>
              <p className="text-xs text-gray-400 font-medium tracking-wide">COGNITIVE AGENT INTERFACE</p>
            </div>
          </div>
          
          <div className="h-8 w-px bg-gray-800 mx-2"></div>
          
          <div className="flex gap-2">
            <button 
              onClick={() => setActiveTab('chat')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === 'chat' 
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              Live Chat & Telemetry
            </button>
            <button 
              onClick={() => setActiveTab('history')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === 'history' 
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`}
            >
              <History className="w-4 h-4" />
              Session History
            </button>
          </div>
        </div>
        
        <div className="flex gap-4 items-center">
          <div className="flex items-center gap-2 text-xs font-mono text-gray-500 bg-gray-950 px-3 py-1.5 rounded-md border border-gray-800">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            API: {apiBase}
          </div>
          <button 
            onClick={() => void fetchDashboard(false)}
            disabled={loading}
            className="bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1.5 rounded-md text-sm font-medium transition-colors flex items-center gap-2 border border-gray-700"
            title="Atualizar Telemetria"
          >
            <Activity className={`w-4 h-4 ${loading ? 'animate-spin text-blue-400' : ''}`} />
          </button>
        </div>
      </header>

      {error && (
        <div className="bg-red-950/50 border-b border-red-900 text-red-200 p-3 text-sm text-center flex items-center justify-center gap-2">
          <Zap className="w-4 h-4" /> {error}
        </div>
      )}

      <main className="flex-1 flex overflow-hidden">
        {activeTab === 'chat' ? (
          <div className="flex-1 overflow-y-auto bg-[#0a0a0c]">
            <div className="max-w-5xl mx-auto p-6 space-y-6">
              <div className="bg-gray-950 border border-gray-800 rounded-xl overflow-hidden">
                <div className="p-4 border-b border-gray-800 bg-gray-900/50 flex justify-between items-center">
                  <h2 className="font-semibold text-gray-200 flex items-center gap-2">
                    <MessageSquare className="w-4 h-4 text-blue-400" />
                    Interaction
                  </h2>
                  <span className="text-[10px] font-mono text-gray-500 bg-gray-800 px-2 py-0.5 rounded border border-gray-700 truncate max-w-[150px]">
                    {activeSessionId}
                  </span>
                </div>
                <div className="p-4 border-b border-gray-800">
                  <form onSubmit={handleSendMessage} className="relative">
                    <input
                      type="text"
                      value={chatInput}
                      onChange={(e) => setChatInput(e.target.value)}
                      placeholder="Digite sua mensagem..."
                      disabled={isProcessing}
                      className="w-full bg-gray-950 border border-gray-700 rounded-xl pl-4 pr-12 py-3 text-sm text-gray-200 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all disabled:opacity-50"
                    />
                    <button
                      type="submit"
                      disabled={!chatInput.trim() || isProcessing}
                      className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 bg-blue-600 text-white rounded-lg hover:bg-blue-500 disabled:opacity-50 disabled:hover:bg-blue-600 transition-colors"
                    >
                      {isProcessing ? <Activity className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                    </button>
                  </form>
                </div>
                <div className="relative">
                  <div 
                    ref={chatScrollRef}
                    className="max-h-72 overflow-y-auto p-4 space-y-6 scroll-smooth"
                  >
                    {chatMessages.length === 0 ? (
                      <div className="h-40 flex flex-col items-center justify-center text-gray-500 space-y-4">
                        <Bot className="w-12 h-12 text-gray-800" />
                        <p className="text-sm">Envie uma mensagem para iniciar o ciclo cognitivo.</p>
                      </div>
                    ) : (
                      chatMessages.map((msg) => (
                        <div 
                          key={msg.id} 
                          className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
                        >
                          <div className={`shrink-0 w-8 h-8 rounded-full flex items-center justify-center border ${
                            msg.role === 'user' 
                              ? 'bg-blue-900/50 border-blue-500/30 text-blue-400' 
                              : 'bg-purple-900/50 border-purple-500/30 text-purple-400'
                          }`}>
                            {msg.role === 'user' ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
                          </div>
                          
                          <div className={`flex flex-col max-w-[85%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                            <div className={`p-3 rounded-2xl text-sm whitespace-pre-wrap ${
                              msg.role === 'user'
                                ? 'bg-blue-600 text-white rounded-tr-sm'
                                : 'bg-gray-800 text-gray-200 rounded-tl-sm border border-gray-700'
                            }`}>
                              {msg.content || (msg.status === 'processing' && <span className="flex gap-1 items-center h-5"><span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce"></span><span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '150ms'}}></span><span className="w-1.5 h-1.5 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '300ms'}}></span></span>)}
                            </div>
                            <span className="text-[10px] text-gray-500 mt-1 px-1">
                              {formatTime(msg.timestamp.toISOString())}
                            </span>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                  {!isAtBottom && (
                    <button
                      onClick={() => {
                        if (chatScrollRef.current) {
                          chatScrollRef.current.scrollTo({ top: chatScrollRef.current.scrollHeight, behavior: 'smooth' });
                        }
                      }}
                      className="absolute bottom-2 right-4 z-10 px-3 py-1.5 rounded-full text-xs bg-gray-800/90 text-gray-200 border border-gray-700 shadow hover:bg-gray-700"
                    >
                      Ir para a última mensagem
                    </button>
                  )}
                </div>
              </div>

              <div className="bg-gray-950 border border-gray-800 rounded-xl overflow-hidden">
                <div className="p-4 border-b border-gray-800 bg-[#0a0a0c]/80 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-emerald-400" />
                  <h3 className="font-semibold text-gray-200">Live Cognitive Trace (3 colunas)</h3>
                </div>
                <div className="p-4">
                  {liveEventsDesc.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-12 text-gray-600">
                      <Activity className="w-12 h-12 mb-4 opacity-20" />
                      <p>Aguardando atividade cognitiva...</p>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                      {(['felt','thought','decision'] as const).map((col) => {
                        const colEvents = liveEventsDesc.filter(e => 
                          col === 'decision' ? (e.type === 'decision' || e.type === 'execution' || e.type === 'reflection') : e.type === col
                        );
                        return (
                          <div key={col} className="bg-gray-900/40 border border-gray-800 rounded-xl overflow-hidden">
                            <div className="p-3 border-b border-gray-800 flex items-center gap-2">
                              <span className="uppercase tracking-widest text-xs text-gray-400">{col === 'decision' ? 'decision • execution • reflection' : col}</span>
                            </div>
                            <div className="p-3 space-y-3">
                              {colEvents.length === 0 ? (
                                <div className="text-xs text-gray-500 italic">Sem eventos</div>
                              ) : (
                                colEvents.map((ev) => (
                                  <div key={ev.id} className={`rounded-lg border p-4 ${renderEventColor(ev.type)} bg-opacity-40`}>
                                    <div className="flex items-center justify-between mb-3">
                                      <div className="flex items-center gap-2">
                                        <div className="w-5 h-5">{renderEventIcon(ev.type)}</div>
                                        <span className="text-xs uppercase tracking-wider text-gray-300">{ev.type}</span>
                                      </div>
                                      <span className="text-[10px] font-mono text-gray-500">{formatTime(ev.timestamp)}</span>
                                    </div>
                                    {renderEventContent(ev)}
                                  </div>
                                ))
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* HISTORY TAB */
          <div className="flex-1 overflow-y-auto bg-[#0a0a0c] p-8">
            <div className="max-w-5xl mx-auto space-y-6">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-2xl font-bold text-gray-100 flex items-center gap-2">
                    <History className="w-6 h-6 text-blue-400" />
                    Histórico de Sessões
                  </h2>
                  <p className="text-gray-400 text-sm mt-1">Navegue pelas execuções passadas do agente cognitivo</p>
                </div>
                
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    <label htmlFor="session-select" className="text-sm text-gray-400">Sessão:</label>
                    <select 
                      id="session-select"
                      value={selectedSession} 
                      onChange={(e) => setSelectedSession(e.target.value)}
                      className="bg-gray-900 border border-gray-700 rounded-md px-3 py-2 text-sm text-gray-200 focus:outline-none focus:border-blue-500"
                    >
                      <option value="all">Todas</option>
                      {availableSessions.map(session => (
                        <option key={session} value={session}>{session}</option>
                      ))}
                    </select>
                  </div>
                  <div className="h-6 w-px bg-gray-800" />
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-gray-400">Visualização:</span>
                    <button
                      onClick={() => setHistoryView('timeline')}
                      className={`px-3 py-1.5 rounded-md text-sm border ${historyView === 'timeline' ? 'bg-blue-600 text-white border-blue-600' : 'bg-gray-900 text-gray-300 border-gray-700 hover:bg-gray-800'}`}
                    >
                      Timeline
                    </button>
                    <button
                      onClick={() => setHistoryView('board3')}
                      className={`px-3 py-1.5 rounded-md text-sm border ${historyView === 'board3' ? 'bg-blue-600 text-white border-blue-600' : 'bg-gray-900 text-gray-300 border-gray-700 hover:bg-gray-800'}`}
                    >
                      3 Colunas
                    </button>
                  </div>
                </div>
              </div>

              {!dashboard || Object.keys(groupedEvents).length === 0 ? (
                <div className="text-center text-gray-500 py-20 bg-gray-900/30 rounded-xl border border-gray-800 border-dashed">
                  <Activity className="w-12 h-12 text-gray-700 mx-auto mb-4" />
                  Nenhum dado de telemetria encontrado no histórico.
                </div>
              ) : (
                <div className="space-y-6">
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
                          <div className="text-left">
                            <h2 className="font-semibold text-lg text-gray-200 flex items-center gap-2">
                              {sessionId}
                              {sessionId === activeSessionId && (
                                <span className="text-[10px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded uppercase tracking-wider">Ativa</span>
                              )}
                            </h2>
                            <span className="text-xs text-gray-500">{events.length} eventos registrados • Último evento: {formatTime(events[0].timestamp)}</span>
                          </div>
                        </div>
                      </button>

                      {/* Timeline / 3-Column Board */}
                      {expandedSessions[sessionId] && (
                        <div className="p-6 bg-[#0a0a0c]">
                          {historyView === 'timeline' ? (
                            <div className="relative pl-8 border-l border-gray-800 space-y-8 before:absolute before:inset-0 before:ml-[31px] before:-translate-x-px md:before:mx-auto md:before:translate-x-0">
                              {events.map((event) => (
                                <div key={event.id} className="relative group">
                                  <div className={`absolute -left-[41px] mt-1.5 w-6 h-6 rounded-full border-2 bg-gray-950 flex items-center justify-center ${renderEventColor(event.type).split(' ')[0]}`}>
                                    <div className="scale-75">
                                      {renderEventIcon(event.type)}
                                    </div>
                                  </div>
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
                          ) : (
                            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                              {(['felt','thought','decision'] as const).map((col) => {
                                const colEvents = events.filter(e => 
                                  col === 'decision' ? (e.type === 'decision' || e.type === 'execution' || e.type === 'reflection') : e.type === col
                                );
                                return (
                                  <div key={col} className="bg-gray-900/40 border border-gray-800 rounded-xl overflow-hidden">
                                    <div className="p-3 border-b border-gray-800 flex items-center gap-2">
                                      <span className="uppercase tracking-widest text-xs text-gray-400">{col === 'decision' ? 'decision • execution • reflection' : col}</span>
                                    </div>
                                    <div className="p-3 space-y-3">
                                      {colEvents.length === 0 ? (
                                        <div className="text-xs text-gray-500 italic">Sem eventos</div>
                                      ) : (
                                        colEvents.map(ev => (
                                          <div key={ev.id} className={`rounded-lg border p-4 ${renderEventColor(ev.type)} bg-opacity-40`}>
                                            <div className="flex items-center justify-between mb-3">
                                              <div className="flex items-center gap-2">
                                                <div className="w-5 h-5">{renderEventIcon(ev.type)}</div>
                                                <span className="text-xs uppercase tracking-wider text-gray-300">{ev.type}</span>
                                              </div>
                                              <span className="text-[10px] font-mono text-gray-500">{formatTime(ev.timestamp)}</span>
                                            </div>
                                            {renderEventContent(ev)}
                                          </div>
                                        ))
                                      )}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;

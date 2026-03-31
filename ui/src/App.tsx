import { useCallback, useEffect, useState, useMemo, useRef } from 'react';
import { Activity, Brain, CheckCircle, Clock, Zap, History, MessageSquare, ChevronDown, ChevronRight, Terminal, Send, Bot, User, Cpu, FileText, AlertTriangle } from 'lucide-react';
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

interface SystemInfo {
  profile: string;
  capabilities: {
    health: string;
    right_hemisphere: any;
    left_hemisphere: any;
    embeddings: any;
    knowledge_graph: any;
    routing_policy?: {
      perception_model: string;
      reason_model: string;
      reflection_model: string;
      verifier_model?: string | null;
    };
    tools: Array<{
      name: string;
      description: string;
      requires_approval: boolean;
      required_permissions: string[];
      health: string;
    }>;
  };
  routing_resolution?: Record<string, {
    requested_model?: string | null;
    active_model?: string | null;
    backend?: string | null;
    available?: boolean;
    note?: string;
    shared_with_reasoning?: boolean;
  }>;
  memory_backend: string;
  telemetry_backend: string;
  right_hemisphere_backend: string;
  left_hemisphere_backend: string;
  bridge_state_dir?: string | null;
  evolution_archive_path?: string | null;
  awareness_interval_turns: number;
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

interface ArchitectureData {
  components: Array<{
    component_id: string;
    role: string;
    adapter_class: string;
    health: string;
  }>;
  connections: Array<{
    source: string;
    target: string;
    protocol: string;
  }>;
  adaptation_surface: {
    tunable_parameters: string[];
    supported_directives: string[];
  };
  capabilities: {
    right_hemisphere: any;
    left_hemisphere: any;
    embeddings: any;
    knowledge_graph: any;
    routing_policy?: any;
    tools: any[];
  };
}

interface WorkspaceState {
  task_frame: any;
  self_model_ref: any;
  capability_snapshot: any;
  right_notes: any;
  bridge_state: any;
  left_notes: any;
  verifier_feedback: any[];
  runtime_feedback: any[];
  pending_questions: string[];
}

interface AwarenessDiagnostic {
  session_id: string;
  analyzed_turns: number;
  tool_success_rate: number;
  average_retries: number;
  average_surprise: number;
  failure_types: Record<string, number>;
  pending_approval_backlog: number;
  surprise_trend: number;
  dominant_variant?: string | null;
  dominant_variant_ratio: number;
  bottlenecks: Array<{
    bottleneck_id: string;
    description: string;
    severity: number;
    evidence: string[];
    affected_components: string[];
  }>;
}

interface EvolutionDirective {
  directive_id: string;
  directive_type: string;
  target_component: string;
  proposed_change: any;
  reasoning: string;
  status: string;
}

const toTimestampMs = (isoString?: string): number => {
  if (!isoString) return 0;
  const value = Date.parse(isoString);
  return Number.isFinite(value) ? value : 0;
};

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));

const deriveExpectedFreeEnergy = (eventData: any): number => {
  const direct = Number(eventData?.expected_free_energy);
  if (Number.isFinite(direct)) return clamp01(direct);

  const hypotheses = eventData?.world_hypotheses ?? {};
  const fromHypothesis = Number(hypotheses?.expected_free_energy);
  if (Number.isFinite(fromHypothesis)) return clamp01(fromHypothesis);

  const complexity = Number(hypotheses?.interaction_complexity);
  const semanticDensity = Number(hypotheses?.semantic_density);
  const fallback = (Number.isFinite(complexity) ? complexity : 0.0) * 0.6 + (Number.isFinite(semanticDensity) ? semanticDensity : 0.0) * 0.4;
  return clamp01(fallback);
};

const deriveSurprise = (eventData: any): number => {
  const surprise = Number(eventData?.surprise_score);
  if (Number.isFinite(surprise)) return clamp01(surprise);

  const freeEnergy = Number(eventData?.telemetry?.free_energy);
  if (Number.isFinite(freeEnergy)) return clamp01(freeEnergy);

  const salience = Number(eventData?.salience);
  if (Number.isFinite(salience)) return clamp01(salience);

  return 0.0;
};

function App() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSession, setSelectedSession] = useState<string>('all');
  const [availableSessions, setAvailableSessions] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<'chat' | 'history' | 'architecture' | 'state' | 'awareness' | 'mente'>('chat');
  
  // Extra state
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [architecture, setArchitecture] = useState<ArchitectureData | null>(null);
  const [workspaceState, setWorkspaceState] = useState<WorkspaceState | null>(null);
  const [awareness, setAwareness] = useState<AwarenessDiagnostic | null>(null);
  const [directives, setDirectives] = useState<EvolutionDirective[]>([]);
  
  // Chat state
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isForaging, setIsForaging] = useState(false);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);

  // Trigger idle foraging
  const handleIdleForage = async () => {
    if (isForaging) return;
    setIsForaging(true);
    
    // Add a system message to chat to show it's foraging
    const forageMessageId = `forage-${Date.now()}`;
    setChatMessages(prev => [...prev, {
      id: forageMessageId,
      role: 'assistant',
      content: '🧠 Iniciando Forrageamento Epistêmico (Goal Generation)...',
      timestamp: new Date(),
      status: 'processing'
    }]);

    try {
      const response = await fetch(`${apiBase}/v1/system/idle`, { method: 'POST' });
      const data = await response.json();
      
      let resultText = 'Forrageamento concluído.';
      if (data.status === 'ok') {
        const turnResult = data.result?.selected_result || data.result;
        if (turnResult?.left_result?.reasoning_summary) {
          resultText = `Forrageamento concluído.\n\n*Reflexões:* \n- ${turnResult.left_result.reasoning_summary.join('\n- ')}`;
        }
        
        if (turnResult?.left_result?.actions?.length > 0) {
          resultText += `\n\n*Ações Tomadas:* \n` + turnResult.left_result.actions.map((a: any) => `- ${a.action_type}`).join('\n');
        }
      } else {
        resultText = `Erro no forrageamento: ${data.error}`;
      }

      setChatMessages(prev => prev.map(msg => 
        msg.id === forageMessageId 
          ? { ...msg, status: 'done', content: resultText } 
          : msg
      ));
      
      void fetchDashboard(true);
    } catch {
      setChatMessages(prev => prev.map(msg => 
        msg.id === forageMessageId 
          ? { ...msg, status: 'error', content: 'Erro ao tentar forragear.' } 
          : msg
      ));
    } finally {
      setIsForaging(false);
    }
  };
  
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
      
      if (activeTab === 'architecture' || activeTab === 'state' || activeTab === 'awareness') {
        const infoRes = await fetch(`${apiBase}/v1/system/info`);
        const infoData = await infoRes.json();
        if (infoData.status === 'ok') setSystemInfo(infoData.info);
      }

      if (activeTab === 'architecture') {
        const archRes = await fetch(`${apiBase}/v1/system/architecture`);
        const archData = await archRes.json();
        if (archData.status === 'ok') setArchitecture(archData.architecture);
      } else if (activeTab === 'state') {
        const stateRes = await fetch(`${apiBase}/v1/system/state?session_id=${activeSessionId}`);
        const stateData = await stateRes.json();
        if (stateData.status === 'ok') {
          setWorkspaceState(stateData.state);
        } else {
          setWorkspaceState(null);
        }
      } else if (activeTab === 'awareness') {
        const awareRes = await fetch(`${apiBase}/v1/system/awareness?session_id=${activeSessionId}`);
        const awareData = await awareRes.json();
        if (awareData.status === 'ok') setAwareness(awareData.diagnostic);
        
        const dirRes = await fetch(`${apiBase}/v1/system/directives`);
        const dirData = await dirRes.json();
        if (dirData.status === 'ok') setDirectives(dirData.directives);
      }
      
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Erro de conexão');
    } finally {
      if (!silent) {
        setLoading(false);
      }
    }
  }, [apiBase, extractSessions, activeTab, activeSessionId]);

  useEffect(() => {
    void fetchDashboard(false);
  }, [fetchDashboard]);

  useEffect(() => {
    if (activeTab !== 'chat' && activeTab !== 'history' && activeTab !== 'mente') {
      return;
    }
    const intervalId = window.setInterval(() => {
      void fetchDashboard(true);
    }, 2500);

    return () => window.clearInterval(intervalId);
  }, [fetchDashboard, activeTab]);

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

    // Sort by timestamp descending (newest first). Invalid timestamps are treated as oldest.
    return events.sort((a, b) => toTimestampMs(b.timestamp) - toTimestampMs(a.timestamp));
  }, [dashboard, filterEvents]);

  const timelineEvents = useMemo(() => getTimelineEvents(), [getTimelineEvents]);
  
  const liveTimelineEvents = useMemo(() => getTimelineEvents(activeSessionId), [getTimelineEvents, activeSessionId]);
  const liveEventsDesc = useMemo(() => {
    return [...liveTimelineEvents].sort(
      (a, b) => toTimestampMs(b.timestamp) - toTimestampMs(a.timestamp)
    );
  }, [liveTimelineEvents]);
  const menteTimelineEvents = useMemo(
    () => (liveEventsDesc.length > 0 ? liveEventsDesc : timelineEvents),
    [liveEventsDesc, timelineEvents],
  );
  const menteFeltEvents = useMemo(
    () => menteTimelineEvents.filter((event) => event.type === 'felt'),
    [menteTimelineEvents],
  );
  const menteReflectionEvents = useMemo(
    () => menteTimelineEvents.filter((event) => event.type === 'reflection'),
    [menteTimelineEvents],
  );
  const latestMenteFelt = menteFeltEvents[0];
  const latestMenteReflection = menteReflectionEvents[0];
  const latestSurpriseScore = deriveSurprise(latestMenteFelt?.data);
  const latestExpectedFreeEnergy = deriveExpectedFreeEnergy(latestMenteFelt?.data);
  const latestCognitiveDissonance = Number(latestMenteReflection?.data?.cognitive_dissonance ?? 0);

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

    } catch (_err) {
      console.error('Chat error:', _err);
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
              className={`px-4 py-4 text-sm font-medium flex items-center gap-2 transition-colors relative ${activeTab === 'history' ? 'active-tab' : 'inactive-tab'}`}
            >
              <History className="w-4 h-4" /> Histórico
            </button>
            <button 
              onClick={() => setActiveTab('mente')}
              className={`px-4 py-4 text-sm font-medium flex items-center gap-2 transition-colors relative ${activeTab === 'mente' ? 'active-tab' : 'inactive-tab'}`}
            >
              <Zap className="w-4 h-4" /> Mente (V3)
            </button>
            <button 
              onClick={() => setActiveTab('architecture')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === 'architecture' 
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`}
            >
              <Cpu className="w-4 h-4" />
              Architecture
            </button>
            <button 
              onClick={() => setActiveTab('state')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === 'state' 
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`}
            >
              <FileText className="w-4 h-4" />
              State
            </button>
            <button 
              onClick={() => setActiveTab('awareness')}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all flex items-center gap-2 ${
                activeTab === 'awareness' 
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' 
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-800 hover:text-gray-200'
              }`}
            >
              <AlertTriangle className="w-4 h-4" />
              Awareness
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
                  <div className="flex items-center gap-3">
                    <button
                      onClick={handleIdleForage}
                      disabled={isForaging}
                      className="text-xs bg-purple-600/20 text-purple-300 hover:bg-purple-600/30 border border-purple-500/30 px-3 py-1 rounded transition-colors flex items-center gap-2"
                      title="Endogenous Goal Generation"
                    >
                      {isForaging ? <Activity className="w-3 h-3 animate-spin" /> : <Brain className="w-3 h-3" />}
                      Idle Foraging
                    </button>
                    <span className="text-[10px] font-mono text-gray-500 bg-gray-800 px-2 py-0.5 rounded border border-gray-700 truncate max-w-[150px]">
                      {activeSessionId}
                    </span>
                  </div>
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
        ) : activeTab === 'architecture' ? (
          <div className="flex-1 overflow-y-auto bg-[#0a0a0c] p-8">
            <div className="max-w-6xl mx-auto space-y-6">
              <div className="flex items-center gap-3 mb-8">
                <Cpu className="w-8 h-8 text-blue-400" />
                <h2 className="text-2xl font-bold text-gray-100">Cognitive Architecture Map</h2>
              </div>
              
              {!architecture ? (
                <div className="text-center text-gray-500 py-20">Carregando arquitetura...</div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {systemInfo && (
                    <div className="lg:col-span-2 grid grid-cols-1 xl:grid-cols-2 gap-6">
                      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                        <div className="p-4 border-b border-gray-800 bg-gray-900/50">
                          <h3 className="font-semibold text-gray-200">Modelos e Backends Ativos</h3>
                        </div>
                        <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                          {[
                            ['Perception', systemInfo.capabilities.right_hemisphere],
                            ['Reason', systemInfo.capabilities.left_hemisphere],
                            ['Embeddings', systemInfo.capabilities.embeddings],
                            ['Knowledge Graph', systemInfo.capabilities.knowledge_graph],
                          ].map(([label, model]) => (
                            <div key={label} className="bg-gray-950 p-3 rounded-lg border border-gray-800 space-y-1">
                              <div className="text-xs uppercase tracking-wider text-gray-500">{label}</div>
                              <div className="text-sm text-gray-100 font-medium">{model?.model_name ?? 'N/A'}</div>
                              <div className="text-xs text-gray-400 font-mono">{model?.backend ?? 'N/A'}</div>
                              <div className="text-xs text-gray-500">{model?.provider ?? 'N/A'}</div>
                              <span className={`inline-flex text-[10px] px-2 py-0.5 rounded border ${
                                model?.health === 'healthy'
                                  ? 'bg-emerald-500/10 text-emerald-300 border-emerald-500/20'
                                  : 'bg-amber-500/10 text-amber-300 border-amber-500/20'
                              }`}>
                                {model?.health ?? 'unknown'}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                        <div className="p-4 border-b border-gray-800 bg-gray-900/50">
                          <h3 className="font-semibold text-gray-200">Routing Policy e Permissões</h3>
                        </div>
                        <div className="p-4 space-y-4">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {Object.entries(systemInfo.routing_resolution ?? {}).map(([route, payload]) => (
                              <div key={route} className="bg-gray-950 p-3 rounded-lg border border-gray-800 space-y-1">
                                <div className="text-xs uppercase tracking-wider text-gray-500">{route}</div>
                                <div className="text-sm text-gray-100">requested: {payload.requested_model ?? 'auto'}</div>
                                <div className="text-sm text-gray-300">active: {payload.active_model ?? 'N/A'}</div>
                                <div className="text-xs text-gray-500 font-mono">{payload.backend ?? 'N/A'}</div>
                                {payload.note && <div className="text-xs text-gray-400">{payload.note}</div>}
                              </div>
                            ))}
                          </div>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                            {systemInfo.capabilities.tools.map((tool) => (
                              <div key={tool.name} className="bg-gray-950 p-3 rounded-lg border border-gray-800">
                                <div className="flex items-center justify-between gap-3">
                                  <div className="text-sm text-cyan-300 font-medium">{tool.name}</div>
                                  <span className="text-[10px] uppercase tracking-wider text-gray-500">
                                    {tool.requires_approval ? 'approval' : 'direct'}
                                  </span>
                                </div>
                                <div className="mt-2 text-xs text-gray-400">{tool.description}</div>
                                <div className="mt-2 text-[11px] text-gray-500">
                                  perms: {tool.required_permissions.length > 0 ? tool.required_permissions.join(', ') : 'none'}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Components */}
                  <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                    <div className="p-4 border-b border-gray-800 bg-gray-900/50">
                      <h3 className="font-semibold text-gray-200">Componentes Ativos</h3>
                    </div>
                    <div className="p-4 space-y-3">
                      {architecture.components.map(comp => (
                        <div key={comp.component_id} className="bg-gray-950 p-3 rounded-lg border border-gray-800 flex justify-between items-center">
                          <div>
                            <div className="text-sm font-medium text-gray-200">{comp.component_id}</div>
                            <div className="text-xs text-gray-500 font-mono">{comp.adapter_class}</div>
                          </div>
                          <span className={`text-xs px-2 py-1 rounded uppercase tracking-wider ${
                            comp.health === 'healthy' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                          }`}>
                            {comp.health}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-6">
                    {/* Connections */}
                    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                      <div className="p-4 border-b border-gray-800 bg-gray-900/50">
                        <h3 className="font-semibold text-gray-200">Topologia</h3>
                      </div>
                      <div className="p-4">
                        <ul className="space-y-2">
                          {architecture.connections.map((conn, i) => (
                            <li key={i} className="flex items-center gap-2 text-sm text-gray-300 bg-gray-950 p-2 rounded border border-gray-800">
                              <span className="text-blue-400">{conn.source}</span>
                              <ChevronRight className="w-4 h-4 text-gray-600" />
                              <span className="text-purple-400">{conn.target}</span>
                              <span className="text-xs text-gray-500 ml-auto bg-gray-900 px-2 py-0.5 rounded">{conn.protocol}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    {/* Capabilities */}
                    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                      <div className="p-4 border-b border-gray-800 bg-gray-900/50">
                        <h3 className="font-semibold text-gray-200">Capacidades e Tools</h3>
                      </div>
                      <div className="p-4">
                        <div className="flex flex-wrap gap-2">
                          {architecture.capabilities.tools.map((t, i) => (
                            <span key={i} className="text-xs bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 px-2 py-1 rounded" title={t.description}>
                              {t.name} {t.requires_approval && '⚠️'}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : activeTab === 'state' ? (
          <div className="flex-1 overflow-y-auto bg-[#0a0a0c] p-8">
            <div className="max-w-5xl mx-auto space-y-6">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-3">
                  <FileText className="w-8 h-8 text-purple-400" />
                  <div>
                    <h2 className="text-2xl font-bold text-gray-100">Cognitive Workspace</h2>
                    <p className="text-sm text-gray-400">Estado compartilhado do turno atual ou mais recente</p>
                  </div>
                </div>
                <div className="text-xs font-mono text-gray-500 bg-gray-900 px-3 py-1.5 rounded border border-gray-800">
                  {workspaceState ? workspaceState.task_frame?.session_id : 'N/A'}
                </div>
              </div>

              {!workspaceState ? (
                <div className="text-center text-gray-500 py-20 bg-gray-900/30 rounded-xl border border-gray-800 border-dashed">
                  Nenhum workspace encontrado na sessão ativa.
                </div>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {systemInfo && (
                    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden md:col-span-2">
                      <div className="p-3 border-b border-gray-800 bg-indigo-500/10 flex items-center gap-2">
                        <Cpu className="w-4 h-4 text-indigo-400" />
                        <h3 className="font-semibold text-indigo-300 text-sm uppercase tracking-wider">Runtime Envelope</h3>
                      </div>
                      <div className="p-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                        <div className="bg-gray-950 border border-gray-800 rounded p-3">
                          <div className="text-xs uppercase tracking-wider text-gray-500 mb-1">Profile</div>
                          <div className="text-gray-100">{systemInfo.profile}</div>
                        </div>
                        <div className="bg-gray-950 border border-gray-800 rounded p-3">
                          <div className="text-xs uppercase tracking-wider text-gray-500 mb-1">Bridge State Dir</div>
                          <div className="text-gray-300 text-xs break-all">{systemInfo.bridge_state_dir ?? 'N/A'}</div>
                        </div>
                        <div className="bg-gray-950 border border-gray-800 rounded p-3">
                          <div className="text-xs uppercase tracking-wider text-gray-500 mb-1">Evolution Archive</div>
                          <div className="text-gray-300 text-xs break-all">{systemInfo.evolution_archive_path ?? 'N/A'}</div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Right Notes */}
                  <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                    <div className="p-3 border-b border-gray-800 bg-yellow-500/10 flex items-center gap-2">
                      <Zap className="w-4 h-4 text-yellow-500" />
                      <h3 className="font-semibold text-yellow-500 text-sm uppercase tracking-wider">Perception (Right)</h3>
                    </div>
                    <pre className="p-4 text-xs text-gray-300 font-mono whitespace-pre-wrap overflow-auto">
                      {JSON.stringify(workspaceState.right_notes, null, 2)}
                    </pre>
                  </div>

                  {/* Bridge State */}
                  <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                    <div className="p-3 border-b border-gray-800 bg-blue-500/10 flex items-center gap-2">
                      <Brain className="w-4 h-4 text-blue-500" />
                      <h3 className="font-semibold text-blue-500 text-sm uppercase tracking-wider">Bridge Control</h3>
                    </div>
                    <pre className="p-4 text-xs text-gray-300 font-mono whitespace-pre-wrap overflow-auto">
                      {JSON.stringify(workspaceState.bridge_state, null, 2)}
                    </pre>
                  </div>

                  {/* Left Notes */}
                  <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden md:col-span-2">
                    <div className="p-3 border-b border-gray-800 bg-emerald-500/10 flex items-center gap-2">
                      <Terminal className="w-4 h-4 text-emerald-500" />
                      <h3 className="font-semibold text-emerald-500 text-sm uppercase tracking-wider">Reasoning (Left)</h3>
                    </div>
                    <pre className="p-4 text-xs text-gray-300 font-mono whitespace-pre-wrap overflow-auto">
                      {JSON.stringify(workspaceState.left_notes, null, 2)}
                    </pre>
                  </div>

                  <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                    <div className="p-3 border-b border-gray-800 bg-cyan-500/10 flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-cyan-400" />
                      <h3 className="font-semibold text-cyan-300 text-sm uppercase tracking-wider">Self Model Ref</h3>
                    </div>
                    <pre className="p-4 text-xs text-gray-300 font-mono whitespace-pre-wrap overflow-auto max-h-80">
                      {JSON.stringify(workspaceState.self_model_ref, null, 2)}
                    </pre>
                  </div>

                  <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                    <div className="p-3 border-b border-gray-800 bg-sky-500/10 flex items-center gap-2">
                      <Activity className="w-4 h-4 text-sky-400" />
                      <h3 className="font-semibold text-sky-300 text-sm uppercase tracking-wider">Capability Snapshot</h3>
                    </div>
                    <pre className="p-4 text-xs text-gray-300 font-mono whitespace-pre-wrap overflow-auto max-h-80">
                      {JSON.stringify(workspaceState.capability_snapshot, null, 2)}
                    </pre>
                  </div>

                  <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden md:col-span-2">
                    <div className="p-3 border-b border-gray-800 bg-orange-500/10 flex items-center gap-2">
                      <MessageSquare className="w-4 h-4 text-orange-400" />
                      <h3 className="font-semibold text-orange-300 text-sm uppercase tracking-wider">Pending Questions</h3>
                    </div>
                    <div className="p-4">
                      {workspaceState.pending_questions.length === 0 ? (
                        <div className="text-sm text-gray-500">Nenhuma pergunta pendente registrada no workspace.</div>
                      ) : (
                        <ul className="space-y-2">
                          {workspaceState.pending_questions.map((question, i) => (
                            <li key={i} className="text-sm text-gray-200 bg-gray-950 border border-gray-800 rounded p-3">
                              {question}
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  </div>

                  {/* Feedback */}
                  {(workspaceState.verifier_feedback.length > 0 || workspaceState.runtime_feedback.length > 0) && (
                    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden md:col-span-2">
                      <div className="p-3 border-b border-gray-800 bg-red-500/10 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-red-500" />
                        <h3 className="font-semibold text-red-500 text-sm uppercase tracking-wider">Feedback Loop</h3>
                      </div>
                      <div className="p-4 space-y-4">
                        {workspaceState.runtime_feedback.map((f, i) => (
                          <div key={i} className="text-xs bg-gray-950 border border-gray-800 p-2 rounded">
                            <span className="text-red-400 font-bold mr-2">Runtime [{f.action}]:</span>
                            <span className="text-gray-300">{f.violations?.join(', ')}</span>
                          </div>
                        ))}
                        {workspaceState.verifier_feedback.map((f, i) => (
                          <div key={i} className="text-xs bg-gray-950 border border-gray-800 p-2 rounded">
                            <span className="text-orange-400 font-bold mr-2">Verifier:</span>
                            <span className="text-gray-300">{f.issues?.join('; ')}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ) : activeTab === 'awareness' ? (
          <div className="flex-1 overflow-y-auto bg-[#0a0a0c] p-8">
            <div className="max-w-5xl mx-auto space-y-6">
              <div className="flex items-center gap-3 mb-8">
                <AlertTriangle className="w-8 h-8 text-amber-500" />
                <div>
                  <h2 className="text-2xl font-bold text-gray-100">Introspection & Awareness</h2>
                  <p className="text-sm text-gray-400">Gargalos identificados e diretivas de evolução pendentes</p>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Diagnostics */}
                <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                  <div className="p-4 border-b border-gray-800 bg-gray-900/50">
                    <h3 className="font-semibold text-gray-200">Gargalos Recentes</h3>
                  </div>
                  <div className="p-4 space-y-4">
                    {awareness && (
                      <div className="grid grid-cols-2 gap-3">
                        <div className="bg-gray-950 border border-gray-800 rounded p-3">
                          <div className="text-[10px] uppercase tracking-wider text-gray-500">Turns</div>
                          <div className="text-lg text-gray-100">{awareness.analyzed_turns}</div>
                        </div>
                        <div className="bg-gray-950 border border-gray-800 rounded p-3">
                          <div className="text-[10px] uppercase tracking-wider text-gray-500">Approval Backlog</div>
                          <div className="text-lg text-gray-100">{awareness.pending_approval_backlog}</div>
                        </div>
                        <div className="bg-gray-950 border border-gray-800 rounded p-3">
                          <div className="text-[10px] uppercase tracking-wider text-gray-500">Avg Surprise</div>
                          <div className="text-lg text-gray-100">{awareness.average_surprise.toFixed(3)}</div>
                        </div>
                        <div className="bg-gray-950 border border-gray-800 rounded p-3">
                          <div className="text-[10px] uppercase tracking-wider text-gray-500">Surprise Trend</div>
                          <div className="text-lg text-gray-100">{awareness.surprise_trend.toFixed(3)}</div>
                        </div>
                      </div>
                    )}
                    {!awareness ? (
                      <div className="text-center text-gray-500 text-sm">Nenhum diagnóstico gerado.</div>
                    ) : awareness.bottlenecks.length === 0 ? (
                      <div className="flex items-center gap-2 text-emerald-400 bg-emerald-500/10 p-3 rounded border border-emerald-500/20 text-sm">
                        <CheckCircle className="w-4 h-4" /> Sistema operando sem gargalos críticos ({awareness.analyzed_turns} turnos analisados).
                      </div>
                    ) : (
                      <div className="space-y-4">
                        {awareness.bottlenecks.map(b => (
                          <div key={b.bottleneck_id} className="bg-red-500/10 border border-red-500/30 p-3 rounded-lg">
                            <h4 className="font-semibold text-red-400 text-sm flex justify-between">
                              {b.description}
                              <span className="text-xs bg-red-950 px-2 py-0.5 rounded">Sev: {b.severity}</span>
                            </h4>
                            <ul className="mt-2 space-y-1">
                              {b.evidence.map((ev, i) => (
                                <li key={i} className="text-xs text-gray-300 flex items-start gap-2">
                                  <span className="text-red-500/50 mt-0.5">•</span> {ev}
                                </li>
                              ))}
                            </ul>
                            <div className="mt-3 flex gap-2">
                              {b.affected_components.map(c => (
                                <span key={c} className="text-[10px] uppercase tracking-wider text-gray-400 bg-gray-950 px-2 py-0.5 rounded border border-gray-800">
                                  {c}
                                </span>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                    {awareness && (
                      <div className="bg-gray-950 border border-gray-800 rounded p-3 space-y-2">
                        <div className="text-xs uppercase tracking-wider text-gray-500">Failure Types e Dominância</div>
                        <div className="text-sm text-gray-300">
                          variant: {awareness.dominant_variant ?? 'n/a'} ({awareness.dominant_variant_ratio.toFixed(3)})
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(awareness.failure_types).length === 0 ? (
                            <span className="text-sm text-gray-500">Nenhum failure type recente.</span>
                          ) : (
                            Object.entries(awareness.failure_types).map(([name, count]) => (
                              <span key={name} className="text-xs bg-red-500/10 text-red-300 border border-red-500/20 px-2 py-1 rounded">
                                {name}: {count}
                              </span>
                            ))
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Directives */}
                <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">
                  <div className="p-4 border-b border-gray-800 bg-gray-900/50 flex justify-between items-center">
                    <h3 className="font-semibold text-gray-200">Fila de Diretivas</h3>
                    <span className="bg-blue-500/20 text-blue-400 text-xs px-2 py-0.5 rounded-full border border-blue-500/30">
                      {directives.length} Pendentes
                    </span>
                  </div>
                  <div className="p-4 space-y-4">
                    {directives.length === 0 ? (
                      <div className="text-center text-gray-500 text-sm py-4">Nenhuma ação evolutiva pendente.</div>
                    ) : (
                      directives.map(dir => (
                        <div key={dir.directive_id} className="bg-gray-950 border border-gray-800 p-4 rounded-lg space-y-3">
                          <div className="flex justify-between items-start">
                            <div>
                              <span className={`text-[10px] uppercase tracking-wider px-2 py-0.5 rounded border mb-2 inline-block ${
                                dir.directive_type === 'parameter' ? 'bg-purple-500/20 text-purple-300 border-purple-500/30' : 'bg-orange-500/20 text-orange-300 border-orange-500/30'
                              }`}>
                                {dir.directive_type}
                              </span>
                              <h4 className="text-sm font-medium text-gray-200">{dir.target_component}</h4>
                            </div>
                            <button 
                              onClick={async () => {
                                try {
                                  const res = await fetch(`${apiBase}/v1/system/directives/apply`, {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({ directive_id: dir.directive_id })
                                  });
                                  if (res.ok) void fetchDashboard(false);
                                } catch (e) {
                                  console.error(e);
                                }
                              }}
                              className="text-xs bg-blue-600 hover:bg-blue-500 text-white px-3 py-1.5 rounded transition-colors"
                            >
                              Aplicar
                            </button>
                          </div>
                          
                          <div className="text-xs text-gray-400 border-l-2 border-gray-700 pl-3 py-1">
                            {dir.reasoning}
                          </div>
                          
                          <pre className="text-xs text-gray-300 bg-gray-900 p-2 rounded border border-gray-800 mt-2">
                            {JSON.stringify(dir.proposed_change, null, 2)}
                          </pre>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : activeTab !== 'history' ? (
          <div className="flex-1 overflow-y-auto bg-[#0a0a0c] p-8 flex items-center justify-center">
            <div className="text-gray-500 flex flex-col items-center gap-4">
              {activeTab === 'mente' ? (
                <div className="fixed inset-0 top-[65px] left-[320px] p-6 h-[calc(100vh-65px)] overflow-y-auto bg-gray-950 z-10">
                  <div className="max-w-5xl mx-auto space-y-8">
                    <div className="flex items-center justify-between">
                      <div>
                        <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                          <Brain className="w-8 h-8 text-blue-400" /> Paisagem Cognitiva V3
                        </h2>
                        <p className="text-gray-400 mt-1">Sinalética de Inferência Ativa e Energia Livre Esperada (G).</p>
                      </div>
                      <div className="bg-blue-500/10 border border-blue-500/20 px-4 py-2 rounded-lg flex items-center gap-3">
                        <span className="text-xs font-mono text-blue-300 uppercase">Estado Social:</span>
                        <div className="flex gap-1">
                          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                          <span className="text-xs text-emerald-400">Peer Sync Ativo</span>
                        </div>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                      <div className="bg-gray-900/50 border border-emerald-500/30 rounded-xl p-6 relative overflow-hidden backdrop-blur-sm">
                        <div className="absolute top-0 right-0 p-4 opacity-10"><Zap className="w-16 h-16" /></div>
                        <h3 className="text-sm font-mono text-emerald-400 uppercase tracking-widest mb-2">Entropy (Surprise)</h3>
                        <div className="flex items-baseline gap-2">
                          <span className="text-4xl font-bold text-white">
                            {latestSurpriseScore.toFixed(3)}
                          </span>
                          <span className="text-xs text-emerald-500/70">D_kl[Q||P]</span>
                        </div>
                      </div>

                      <div className="bg-gray-900/50 border border-blue-500/30 rounded-xl p-6 relative overflow-hidden backdrop-blur-sm">
                        <div className="absolute top-0 right-0 p-4 opacity-10"><Cpu className="w-16 h-16" /></div>
                        <h3 className="text-sm font-mono text-blue-400 uppercase tracking-widest mb-2">Expected Free Energy (G)</h3>
                        <div className="flex items-baseline gap-2">
                          <span className="text-4xl font-bold text-white">
                            {latestExpectedFreeEnergy.toFixed(3)}
                          </span>
                          <span className="text-xs text-blue-500/70">Epistemic Gain</span>
                        </div>
                      </div>

                      <div className="bg-gray-900/50 border border-amber-500/30 rounded-xl p-6 relative overflow-hidden backdrop-blur-sm">
                        <div className="absolute top-0 right-0 p-4 opacity-10"><AlertTriangle className="w-16 h-16" /></div>
                        <h3 className="text-sm font-mono text-amber-400 uppercase tracking-widest mb-2">Cognitive Dissonance</h3>
                        <div className="flex items-baseline gap-2">
                          <span className="text-4xl font-bold text-white">
                            {(Number.isFinite(latestCognitiveDissonance) ? latestCognitiveDissonance : 0).toFixed(3)}
                          </span>
                          <span className="text-xs text-amber-500/70">Model Mismatch</span>
                        </div>
                      </div>
                    </div>

                    <div className="bg-gray-900/40 border border-gray-800 rounded-xl p-8 backdrop-blur-md">
                      <h3 className="text-lg font-medium text-gray-200 mb-6 flex items-center gap-2">
                        <Activity className="w-5 h-5 text-blue-400" /> Fluxo de Ressonância Contextual
                      </h3>
                      <div className="h-64 flex items-end gap-2 px-2">
                        {menteFeltEvents.slice(0, 15).reverse().map((e, i) => (
                          <div key={i} className="flex-1 flex flex-col gap-2 group relative">
                            <div className="absolute -top-10 left-1/2 -translate-x-1/2 bg-gray-800 px-2 py-1 rounded text-[10px] opacity-0 group-hover:opacity-100 transition-opacity z-20 whitespace-nowrap">
                              EFE: {deriveExpectedFreeEnergy(e.data).toFixed(3)}
                            </div>
                            <div 
                              className="w-full bg-blue-500/40 border-t-2 border-blue-400 rounded-t-sm transition-all hover:bg-blue-500/60" 
                              style={{ height: `${Math.max(4, deriveExpectedFreeEnergy(e.data) * 100)}%` }}
                            ></div>
                            <div 
                              className="w-full bg-emerald-500/40 border-t-2 border-emerald-400 rounded-t-sm transition-all hover:bg-emerald-500/60" 
                              style={{ height: `${Math.max(4, deriveSurprise(e.data) * 100)}%` }}
                            ></div>
                          </div>
                        ))}
                      </div>
                      <div className="flex justify-between mt-4 text-[10px] text-gray-500 font-mono uppercase tracking-widest px-2">
                        <span>Estado Mais Antigo</span>
                        <div className="flex gap-6">
                          <div className="flex items-center gap-2"><div className="w-3 h-3 bg-blue-500/50 rounded-sm"></div> G (EFE)</div>
                          <div className="flex items-center gap-2"><div className="w-3 h-3 bg-emerald-500/50 rounded-sm"></div> Surpresa (VFE)</div>
                        </div>
                        <span>Real-time (Active Inference)</span>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <>
                  {activeTab === 'architecture' && <Cpu className="w-12 h-12 opacity-50" />}
                  {activeTab === 'state' && <FileText className="w-12 h-12 opacity-50" />}
                  {activeTab === 'awareness' && <AlertTriangle className="w-12 h-12 opacity-50" />}
                  <p className="text-lg">Em desenvolvimento...</p>
                </>
              )}
              {activeTab !== 'mente' && <p className="text-lg">Em desenvolvimento...</p>}
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

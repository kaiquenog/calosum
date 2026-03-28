import { useState, useEffect } from 'react';
import { Activity, Brain, CheckCircle, Clock, Zap } from 'lucide-react';
import './App.css';

interface DashboardEvent {
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

interface Dashboard {
  felt: DashboardEvent[];
  thought: DashboardEvent[];
  decision: DashboardEvent[];
  execution: any[];
  reflection: any[];
}

function App() {
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState('teste-ui-123');

  const fetchDashboard = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:8000/v1/telemetry/dashboard/${sessionId}`);
      const data = await response.json();
      if (data.status === 'ok') {
        setDashboard(data.dashboard);
      } else {
        setError(data.error || 'Erro ao carregar dashboard');
      }
    } catch (err: any) {
      setError(err.message || 'Erro de conexão');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboard();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-8 font-sans">
      <header className="mb-8 flex items-center justify-between border-b border-gray-800 pb-6">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent flex items-center gap-3">
            <Brain className="w-8 h-8 text-blue-400" />
            Calosum Telemetry
          </h1>
          <p className="text-gray-400 mt-2">Monitoramento de Dualidade Cognitiva</p>
        </div>
        
        <div className="flex gap-4">
          <input 
            type="text" 
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
            className="bg-gray-900 border border-gray-700 rounded-md px-4 py-2 text-sm focus:outline-none focus:border-blue-500"
            placeholder="Session ID"
          />
          <button 
            onClick={fetchDashboard}
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

      {!dashboard ? (
        <div className="text-center text-gray-500 py-20">Nenhum dado de telemetria encontrado para esta sessão.</div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* HEMISFÉRIO DIREITO (JEPA / SENTIMENTO) */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden shadow-xl">
            <div className="bg-gradient-to-r from-indigo-900/40 to-blue-900/20 p-4 border-b border-gray-800 flex items-center gap-3">
              <Zap className="text-yellow-400 w-5 h-5" />
              <h2 className="font-semibold text-lg text-indigo-100">Right Hemisphere (Felt)</h2>
            </div>
            <div className="p-4 space-y-4 max-h-[70vh] overflow-y-auto">
              {dashboard.felt.length === 0 ? <p className="text-gray-500 text-sm">Sem eventos</p> : 
                dashboard.felt.map((event, i) => (
                  <div key={i} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
                    <div className="flex justify-between items-start mb-3">
                      <span className="text-xs text-gray-400 font-mono">Saliência: {event.salience}</span>
                    </div>
                    <div className="mb-3">
                      <h3 className="text-xs uppercase tracking-wider text-gray-500 mb-1">Emoções</h3>
                      <div className="flex flex-wrap gap-2">
                        {event.emotional_labels?.map((l, j) => (
                          <span key={j} className="bg-indigo-900/60 text-indigo-200 text-xs px-2 py-1 rounded-full">{l}</span>
                        ))}
                      </div>
                    </div>
                    {event.world_hypotheses && (
                      <div>
                        <h3 className="text-xs uppercase tracking-wider text-gray-500 mb-1">Hipóteses de Mundo</h3>
                        <ul className="text-sm space-y-1">
                          {Object.entries(event.world_hypotheses).map(([k, v]) => (
                            <li key={k} className="flex justify-between">
                              <span className="text-gray-300">{k}</span>
                              <span className="text-blue-400 font-mono">{Number(v).toFixed(3)}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))
              }
            </div>
          </div>

          {/* HEMISFÉRIO ESQUERDO (SLM / PENSAMENTO) */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden shadow-xl">
            <div className="bg-gradient-to-r from-emerald-900/40 to-teal-900/20 p-4 border-b border-gray-800 flex items-center gap-3">
              <Brain className="text-emerald-400 w-5 h-5" />
              <h2 className="font-semibold text-lg text-emerald-100">Left Hemisphere (Thought)</h2>
            </div>
            <div className="p-4 space-y-4 max-h-[70vh] overflow-y-auto">
              {dashboard.thought.length === 0 ? <p className="text-gray-500 text-sm">Sem eventos</p> : 
                dashboard.thought.map((event, i) => (
                  <div key={i} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
                    <div className="mb-3 pb-3 border-b border-gray-700">
                      <h3 className="text-xs uppercase tracking-wider text-gray-500 mb-1">Programa Lambda</h3>
                      <code className="text-xs text-emerald-300 bg-gray-950 p-2 rounded block break-all">
                        {event.lambda_signature || 'N/A'}
                      </code>
                    </div>
                    <div>
                      <h3 className="text-xs uppercase tracking-wider text-gray-500 mb-2">Trilha de Raciocínio</h3>
                      <ul className="space-y-2 text-sm text-gray-300">
                        {event.reasoning_summary?.map((step, j) => (
                          <li key={j} className="flex items-start gap-2">
                            <span className="text-emerald-500 mt-1">▹</span>
                            <span>{step}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))
              }
            </div>
          </div>

          {/* CORPO CALOSO / ORQUESTRADOR (DECISÃO) */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden shadow-xl">
            <div className="bg-gradient-to-r from-purple-900/40 to-pink-900/20 p-4 border-b border-gray-800 flex items-center gap-3">
              <CheckCircle className="text-purple-400 w-5 h-5" />
              <h2 className="font-semibold text-lg text-purple-100">Synthesis (Decision)</h2>
            </div>
            <div className="p-4 space-y-4 max-h-[70vh] overflow-y-auto">
              {dashboard.decision.length === 0 ? <p className="text-gray-500 text-sm">Sem eventos</p> : 
                dashboard.decision.map((event, i) => (
                  <div key={i} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/50">
                    <div className="mb-4">
                      <h3 className="text-xs uppercase tracking-wider text-gray-500 mb-2">Resposta ao Usuário</h3>
                      <div className="bg-gray-950 border border-gray-700 rounded p-3 text-sm text-gray-200 whitespace-pre-wrap">
                        {event.response_text || '(Apenas ações executadas)'}
                      </div>
                    </div>
                    
                    <div className="mb-3">
                      <h3 className="text-xs uppercase tracking-wider text-gray-500 mb-2">Ações Disparadas</h3>
                      <div className="flex flex-wrap gap-2">
                        {event.action_types?.map((a, j) => (
                          <span key={j} className="bg-purple-900/40 border border-purple-700/50 text-purple-200 text-xs px-2 py-1 rounded">
                            {a}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    {(event.runtime_retry_count! > 0 || event.runtime_rejected_count! > 0) && (
                      <div className="mt-4 pt-3 border-t border-gray-700 flex items-center gap-4 text-xs">
                        <span className="flex items-center gap-1 text-yellow-500">
                          <Clock className="w-3 h-3" />
                          Retries: {event.runtime_retry_count}
                        </span>
                        <span className="text-red-400">
                          Rejeições: {event.runtime_rejected_count}
                        </span>
                      </div>
                    )}
                  </div>
                ))
              }
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

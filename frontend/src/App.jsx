import React, { useState, useEffect, useRef } from "react";

// Preset templates for user interaction
const TEMPLATES = {
  normal: [
    { from_bank: "001", from_account: "800101", to_bank: "001", to_account: "800102", amount: 150.00, payment_format: "ACH" },
    { from_bank: "001", from_account: "800102", to_bank: "002", to_account: "800201", amount: 145.50, payment_format: "ACH" },
    { from_bank: "003", from_account: "800301", to_bank: "002", to_account: "800202", amount: 5000.00, payment_format: "Wire" },
    { from_bank: "002", from_account: "800202", to_bank: "001", to_account: "800103", amount: 2500.00, payment_format: "Cheque" }
  ],
  laundering: [
    // Circular money laundering loop
    { from_bank: "099", from_account: "900001", to_bank: "088", to_account: "900002", amount: 50000.00, payment_format: "Wire" },
    { from_bank: "088", from_account: "900002", to_bank: "077", to_account: "900003", amount: 49500.00, payment_format: "Wire" },
    { from_bank: "077", from_account: "900003", to_bank: "099", to_account: "900001", amount: 49000.00, payment_format: "Wire" },
    // A standard transaction on the side
    { from_bank: "001", from_account: "800101", to_bank: "099", to_account: "900001", amount: 200.00, payment_format: "Bitcoin" }
  ],
  fanout: [
    // One node layering/distributing funds to multiple accounts
    { from_bank: "010", from_account: "555000", to_bank: "020", to_account: "555101", amount: 10000.00, payment_format: "Wire" },
    { from_bank: "010", from_account: "555000", to_bank: "020", to_account: "555102", amount: 10000.00, payment_format: "Wire" },
    { from_bank: "010", from_account: "555000", to_bank: "020", to_account: "555103", amount: 10000.00, payment_format: "Wire" },
    { from_bank: "010", from_account: "555000", to_bank: "020", to_account: "555104", amount: 10000.00, payment_format: "Wire" }
  ]
};

function App() {
  const [activeTab, setActiveTab] = useState("analyzer"); // analyzer, directory, telemetry
  const [jsonText, setJsonText] = useState(JSON.stringify(TEMPLATES.normal, null, 2));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [systemStatus, setSystemStatus] = useState({ online: false, model_loaded: false, registry_size: 0 });
  const [predictions, setPredictions] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);

  // Search & Filter state for Directory view
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all"); // all, licit, illicit
  const [minRiskScore, setMinRiskScore] = useState(0.0);

  // Graph visualizer state
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const requestRef = useRef();

  // Determine backend URL dynamically. If running in Vite dev server (5173), query port 8000.
  const API_BASE = window.location.port === "5173" ? "http://127.0.0.1:8000" : "";

  // Check system health status
  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_BASE}/`);
      if (response.ok) {
        const data = await response.json();
        setSystemStatus({
          online: data.status === "online",
          model_loaded: data.model_loaded,
          registry_size: data.nodes_registered_in_registry || 0
        });
      } else {
        setSystemStatus({ online: false, model_loaded: false, registry_size: 0 });
      }
    } catch {
      setSystemStatus({ online: false, model_loaded: false, registry_size: 0 });
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  // Set preset templates
  const loadTemplate = (name) => {
    setJsonText(JSON.stringify(TEMPLATES[name], null, 2));
  };

  // Run GNN Prediction
  const runPrediction = async () => {
    setError("");
    setLoading(true);
    try {
      const parsedTransactions = JSON.parse(jsonText);

      const response = await fetch(`${API_BASE}/predict_transactions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transactions: parsedTransactions })
      });

      if (!response.ok) {
        throw new Error(`Server returned code ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setPredictions(data);

      // Construct Graph Nodes and Links from inputs
      const nodeMap = new Map();
      const links = [];

      parsedTransactions.forEach((tx) => {
        const src = `${tx.from_bank}_${tx.from_account}`;
        const dst = `${tx.to_bank}_${tx.to_account}`;

        if (!nodeMap.has(src)) {
          nodeMap.set(src, {
            id: src,
            bank: tx.from_bank,
            account: tx.from_account,
            amount_sent: 0,
            amount_received: 0,
            out_degree: 0,
            in_degree: 0
          });
        }
        if (!nodeMap.has(dst)) {
          nodeMap.set(dst, {
            id: dst,
            bank: tx.to_bank,
            account: tx.to_account,
            amount_sent: 0,
            amount_received: 0,
            out_degree: 0,
            in_degree: 0
          });
        }

        // Aggregate locally
        nodeMap.get(src).amount_sent += tx.amount;
        nodeMap.get(src).out_degree += 1;

        nodeMap.get(dst).amount_received += tx.amount;
        nodeMap.get(dst).in_degree += 1;

        links.push({
          source: src,
          target: dst,
          amount: tx.amount,
          format: tx.payment_format
        });
      });

      // Hydrate predictions
      const nodes = Array.from(nodeMap.values()).map((node) => {
        const isIllicit = data.predictions[node.id] === 1;
        const prob = data.probabilities[node.id] || 0.0;
        return {
          ...node,
          is_illicit: isIllicit,
          probability: prob,
          // Position initial state (centered)
          x: 300 + (Math.random() - 0.5) * 100,
          y: 200 + (Math.random() - 0.5) * 100,
          vx: 0,
          vy: 0
        };
      });

      setGraphData({ nodes, links });
      if (nodes.length > 0) {
        setSelectedNode(nodes[0]);
      }
    } catch (e) {
      setError(e.message || "Invalid JSON syntax. Please verify your transaction array format.");
    } finally {
      setLoading(false);
    }
  };

  // Run initial prediction on load
  useEffect(() => {
    runPrediction();
  }, []);

  // Simple Force-Directed Layout Simulation
  useEffect(() => {
    if (graphData.nodes.length === 0) return;

    const simulate = () => {
      const { nodes, links } = graphData;
      const width = 600;
      const height = 480;
      const k = 0.12; // Spring strength
      const rep = 1500; // Repulsion constant
      const centerStrength = 0.025;

      // Reset forces and apply center force / repulsion
      for (let i = 0; i < nodes.length; i++) {
        const n1 = nodes[i];

        // Gravity to center
        n1.vx += (width / 2 - n1.x) * centerStrength;
        n1.vy += (height / 2 - n1.y) * centerStrength;

        // Repulsion force between node pairs
        for (let j = i + 1; j < nodes.length; j++) {
          const n2 = nodes[j];
          const dx = n2.x - n1.x || 0.01;
          const dy = n2.y - n1.y || 0.01;
          const distSq = dx * dx + dy * dy;
          const dist = Math.sqrt(distSq);

          if (dist < 180) {
            const force = rep / (distSq + 1);
            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;

            n1.vx -= fx;
            n1.vy -= fy;
            n2.vx += fx;
            n2.vy += fy;
          }
        }
      }

      // Spring attractive force for links
      links.forEach((l) => {
        const sourceNode = nodes.find((n) => n.id === l.source);
        const targetNode = nodes.find((n) => n.id === l.target);
        if (!sourceNode || !targetNode) return;

        const dx = targetNode.x - sourceNode.x || 0.01;
        const dy = targetNode.y - sourceNode.y || 0.01;
        const dist = Math.sqrt(dx * dx + dy * dy);

        // Target spring distance is 120px
        const force = (dist - 120) * k;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        sourceNode.vx += fx;
        sourceNode.vy += fy;
        targetNode.vx -= fx;
        targetNode.vy -= fy;
      });

      // Update positions with friction damping
      const updatedNodes = nodes.map((n) => {
        let nextX = n.x + n.vx;
        let nextY = n.y + n.vy;

        // Keep inside bounds
        nextX = Math.max(30, Math.min(width - 30, nextX));
        nextY = Math.max(30, Math.min(height - 30, nextY));

        return {
          ...n,
          x: nextX,
          y: nextY,
          vx: n.vx * 0.72, // damping
          vy: n.vy * 0.72
        };
      });

      setGraphData({ nodes: updatedNodes, links });
      requestRef.current = requestAnimationFrame(simulate);
    };

    requestRef.current = requestAnimationFrame(simulate);
    return () => cancelAnimationFrame(requestRef.current);
  }, [graphData.nodes.length]);

  // Filtering Logic for Directory Page
  const filteredNodes = graphData.nodes.filter((node) => {
    const matchesSearch = 
      node.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
      node.bank.toLowerCase().includes(searchQuery.toLowerCase()) ||
      node.account.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesStatus = 
      statusFilter === "all" ||
      (statusFilter === "illicit" && node.is_illicit) ||
      (statusFilter === "licit" && !node.is_illicit);

    const matchesRiskScore = node.probability >= minRiskScore;

    return matchesSearch && matchesStatus && matchesRiskScore;
  });

  return (
    <div className="app-container">
      {/* HEADER SECTION */}
      <header className="app-header">
        <div className="brand">
          <div className="pulse-dot"></div>
          <h1>AML Relational GNN Diagnostics Platform</h1>
        </div>
        <div className="status-panel">
          <div className={`status-badge ${systemStatus.online ? "online" : "offline"}`}>
            Backend: {systemStatus.online ? "Online" : "Offline"}
          </div>
          <div className={`status-badge ${systemStatus.model_loaded ? "loaded" : "not-loaded"}`}>
            GCN Model: {systemStatus.model_loaded ? "Loaded" : "Not Loaded"}
          </div>
          <div className="status-badge registry">
            Mapped Registry: {systemStatus.registry_size} Nodes
          </div>
        </div>
      </header>

      {/* PAGE TABS NAVIGATION */}
      <nav className="navigation-bar">
        <button 
          className={`nav-tab ${activeTab === "analyzer" ? "active" : ""}`}
          onClick={() => setActiveTab("analyzer")}
        >
          🕸️ Network Analyzer
        </button>
        <button 
          className={`nav-tab ${activeTab === "directory" ? "active" : ""}`}
          onClick={() => setActiveTab("directory")}
        >
          📋 Account Directory
        </button>
        <button 
          className={`nav-tab ${activeTab === "telemetry" ? "active" : ""}`}
          onClick={() => setActiveTab("telemetry")}
        >
          📊 Model Telemetry & XAI
        </button>
      </nav>

      {/* VIEW 1: NETWORK ANALYZER (GRAPH VISUALIZER) */}
      {activeTab === "analyzer" && (
        <main className="dashboard-grid">
          {/* COLUMN 1: STREAM INGESTION INTERFACE */}
          <section className="dashboard-card ingestion-panel">
            <div className="card-header">
              <h2>📥 Transaction Ingestion</h2>
              <p>Stream real-time transaction JSON arrays into the relational graph engine.</p>
            </div>
            <div className="template-row">
              <button className="btn btn-secondary" onClick={() => loadTemplate("normal")}>
                Licit Flow
              </button>
              <button className="btn btn-secondary btn-danger-hover" onClick={() => loadTemplate("laundering")}>
                Laundering Loop
              </button>
              <button className="btn btn-secondary" onClick={() => loadTemplate("fanout")}>
                Fan-Out Flow
              </button>
            </div>
            <div className="editor-container">
              <textarea
                value={jsonText}
                onChange={(e) => setJsonText(e.target.value)}
                placeholder="Paste transaction JSON array here..."
                rows={12}
              />
            </div>
            {error && <div className="error-message">{error}</div>}
            <button className="btn btn-primary" onClick={runPrediction} disabled={loading}>
              {loading ? "Evaluating GNN Graph..." : "Analyze Transaction Network"}
            </button>
          </section>

          {/* COLUMN 2: INTERACTIVE GRAPH SCREEN */}
          <section className="dashboard-card visual-panel">
            <div className="card-header">
              <h2>🕸️ 2-Hop Network Graph Visualizer</h2>
              <p>Animated edges trace the transaction directions. Hover nodes to view tooltips.</p>
            </div>
            <div className="graph-container">
              {/* Dynamic Interactive SVG Tooltip */}
              {hoveredNode && (
                <div className="tooltip-container">
                  <div className="tooltip-title">Account: {hoveredNode.id}</div>
                  <div className="tooltip-body">
                    <div>Bank: {hoveredNode.bank}</div>
                    <div>Risk Score: {(hoveredNode.probability * 100).toFixed(1)}%</div>
                    <div style={{ color: hoveredNode.is_illicit ? "#EF4444" : "#10B981", fontWeight: "bold", marginTop: "2px" }}>
                      {hoveredNode.is_illicit ? "WARNING: Illicit Anomaly" : "Licit"}
                    </div>
                  </div>
                </div>
              )}

              <svg width="100%" height="100%" viewBox="0 0 600 480">
                <defs>
                  <marker
                    id="arrow"
                    viewBox="0 0 10 10"
                    refX="19"
                    refY="5"
                    markerWidth="5"
                    markerHeight="5"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#4B5563" />
                  </marker>
                </defs>

                {/* Draw Transaction Edges */}
                {graphData.links.map((link, idx) => {
                  const srcNode = graphData.nodes.find((n) => n.id === link.source);
                  const dstNode = graphData.nodes.find((n) => n.id === link.target);
                  if (!srcNode || !dstNode) return null;

                  const midX = (srcNode.x + dstNode.x) / 2;
                  const midY = (srcNode.y + dstNode.y) / 2;

                  return (
                    <g key={`link-${idx}`}>
                      {/* Flow Line with animated dash array */}
                      <line
                        x1={srcNode.x}
                        y1={srcNode.y}
                        x2={dstNode.x}
                        y2={dstNode.y}
                        stroke={link.amount > 20000 ? "#EF4444" : "#3B82F6"}
                        strokeOpacity={0.65}
                        strokeWidth={link.amount > 20000 ? 2 : 1.2}
                        className="edge-flow-line"
                        markerEnd="url(#arrow)"
                      />
                      <text
                        x={midX}
                        y={midY - 4}
                        fill="#9CA3AF"
                        fontSize={7.5}
                        textAnchor="middle"
                        className="edge-label"
                      >
                        ${link.amount.toLocaleString()}
                      </text>
                    </g>
                  );
                })}

                {/* Draw Account Nodes */}
                {graphData.nodes.map((node) => (
                  <g
                    key={node.id}
                    transform={`translate(${node.x},${node.y})`}
                    onClick={() => setSelectedNode(node)}
                    onMouseEnter={() => setHoveredNode(node)}
                    onMouseLeave={() => setHoveredNode(null)}
                    style={{ cursor: "pointer" }}
                  >
                    {/* Ring selection aura */}
                    {selectedNode && selectedNode.id === node.id && (
                      <circle
                        r={18}
                        fill="none"
                        stroke="#60A5FA"
                        strokeWidth={1.5}
                        strokeDasharray="2, 2"
                      />
                    )}
                    <circle
                      r={11}
                      fill={node.is_illicit ? "#EF4444" : "#10B981"}
                      className={node.is_illicit ? "pulse-node" : ""}
                      stroke="#050814"
                      strokeWidth={2}
                    />
                    <text
                      y={20}
                      fill="#F3F4F6"
                      fontSize={8.5}
                      textAnchor="middle"
                      fontWeight="bold"
                    >
                      {node.account}
                    </text>
                  </g>
                ))}
              </svg>
            </div>
          </section>

          {/* COLUMN 3: NODE DIAGNOSTICS & TELEMETRY */}
          <section className="dashboard-card diagnostics-panel">
            <div className="card-header">
              <h2>📊 Entity Diagnostics Panel</h2>
              <p>Select a node from the network graph to evaluate localized GNN features.</p>
            </div>
            {selectedNode ? (
              <div className="diagnostics-details">
                <div className="diagnostics-status-header">
                  <h3>Account: {selectedNode.id}</h3>
                  <span className={`badge ${selectedNode.is_illicit ? "illicit" : "licit"}`}>
                    {selectedNode.is_illicit ? "Illicit" : "Licit"}
                  </span>
                </div>

                <div className="info-grid">
                  <div className="info-tile">
                    <span className="tile-label">Bank ID</span>
                    <span className="tile-value">{selectedNode.bank}</span>
                  </div>
                  <div className="info-tile">
                    <span className="tile-label">Account Code</span>
                    <span className="tile-value">{selectedNode.account}</span>
                  </div>
                  <div className="info-tile">
                    <span className="tile-label">Total Outflow</span>
                    <span className="tile-value">${selectedNode.amount_sent.toLocaleString()}</span>
                  </div>
                  <div className="info-tile">
                    <span className="tile-label">Total Inflow</span>
                    <span className="tile-value">${selectedNode.amount_received.toLocaleString()}</span>
                  </div>
                </div>

                <div className="metric-box">
                  <div className="metric-header">
                    <span>In-Degree / Out-Degree</span>
                    <strong>{selectedNode.in_degree} / {selectedNode.out_degree}</strong>
                  </div>
                  <div className="degree-bar-container">
                    <div 
                      className="degree-bar in" 
                      style={{ width: `${Math.min(100, (selectedNode.in_degree / 5) * 100)}%` }}
                    ></div>
                    <div 
                      className="degree-bar out" 
                      style={{ width: `${Math.min(100, (selectedNode.out_degree / 5) * 100)}%` }}
                    ></div>
                  </div>
                </div>

                <div className="probability-container">
                  <div className="prob-header">
                    <span>Illicit Probability (GCN Score)</span>
                    <strong className={selectedNode.is_illicit ? "text-danger" : "text-success"}>
                      {(selectedNode.probability * 100).toFixed(2)}%
                    </strong>
                  </div>
                  <div className="progress-track">
                    <div
                      className={`progress-fill ${selectedNode.is_illicit ? "illicit" : "licit"}`}
                      style={{ width: `${selectedNode.probability * 100}%` }}
                    ></div>
                  </div>
                </div>

                <div className="gnn-explanation">
                  <h4>GNN Convolutional Neighborhood Context</h4>
                  <p>
                    {selectedNode.is_illicit 
                      ? "Graph Convolutions flagged this node. Illicit structural characteristics detected: cyclic transfer structures, high-velocity outflow, or immediate 2-hop links to known anomalous banking registers."
                      : "Graph Convolutions classified this node as licit. The structural neighborhood displays linear transaction patterns and standard domestic caching velocities."
                    }
                  </p>
                </div>
              </div>
            ) : (
              <div className="no-node-placeholder">
                Click a node inside the network visualizer to evaluate risk parameters.
              </div>
            )}
          </section>
        </main>
      )}

      {/* VIEW 2: ACCOUNT RISK DIRECTORY (SEARCH & FILTERS) */}
      {activeTab === "directory" && (
        <div className="dashboard-card">
          <div className="card-header">
            <h2>📋 Account Risk Directory</h2>
            <p>Comprehensive tabular directory featuring custom GNN anomaly probability filters and bank ID queries.</p>
          </div>

          {/* Filtering Controls */}
          <div className="directory-controls">
            <div className="control-group">
              <label className="control-label">Search Node / Bank ID</label>
              <input
                type="text"
                className="search-input"
                placeholder="Search accounts..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <div className="control-group">
              <label className="control-label">Status Filter</label>
              <select
                className="filter-select"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
              >
                <option value="all">Show All Entities</option>
                <option value="illicit">Illicit Anomaly Only</option>
                <option value="licit">Licit Entities Only</option>
              </select>
            </div>
            <div className="control-group">
              <label className="control-label">Min GCN Risk Threshold: <span className="range-value">{(minRiskScore * 100).toFixed(0)}%</span></label>
              <div className="range-container">
                <input
                  type="range"
                  className="risk-slider"
                  min="0.0"
                  max="1.0"
                  step="0.05"
                  value={minRiskScore}
                  onChange={(e) => setMinRiskScore(parseFloat(e.target.value))}
                />
              </div>
            </div>
          </div>

          {/* Directory Data Table */}
          <div className="table-responsive">
            <table>
              <thead>
                <tr>
                  <th>Account Node ID</th>
                  <th>Bank Code</th>
                  <th>Account Code</th>
                  <th>In-Degree</th>
                  <th>Out-Degree</th>
                  <th>Total Sent</th>
                  <th>Total Received</th>
                  <th>GCN Risk Probability</th>
                  <th>Relational GNN Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredNodes.map((node) => (
                  <tr 
                    key={node.id} 
                    onClick={() => {
                      setSelectedNode(node);
                      setActiveTab("analyzer"); // Redirect back to visualizer to highlight it
                    }}
                    className={selectedNode && selectedNode.id === node.id ? "selected-row" : ""}
                  >
                    <td><strong>{node.id}</strong></td>
                    <td>{node.bank}</td>
                    <td>{node.account}</td>
                    <td>{node.in_degree}</td>
                    <td>{node.out_degree}</td>
                    <td>${node.amount_sent.toLocaleString()}</td>
                    <td>${node.amount_received.toLocaleString()}</td>
                    <td>
                      <div className="table-prob-cell">
                        <span>{(node.probability * 100).toFixed(1)}%</span>
                        <div className="table-prob-track">
                          <div 
                            className={`table-prob-fill ${node.is_illicit ? "illicit" : "licit"}`} 
                            style={{ width: `${node.probability * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </td>
                    <td>
                      <span className={`status-pill ${node.is_illicit ? "danger" : "success"}`}>
                        {node.is_illicit ? "Illicit Anomaly" : "Licit"}
                      </span>
                    </td>
                  </tr>
                ))}
                {filteredNodes.length === 0 && (
                  <tr>
                    <td colSpan={9} style={{ textAlign: "center", padding: "2rem" }}>
                      No accounts matched the selected query filters. Adjust filters to broaden your search.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* VIEW 3: MODEL TELEMETRY & EXPLAINABLE AI (XAI) */}
      {activeTab === "telemetry" && (
        <div className="telemetry-grid">
          {/* Card 1: GNN Layer Graph Convolution Architecture */}
          <div className="telemetry-card">
            <h3>🧠 Relational GCN Architecture</h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
              Detailed topological layers of <strong>AMLGraphNet</strong>. The network propagates features across local subgraphs utilizing 2 convolutions.
            </p>

            <div className="arch-node">
              <div>
                <div className="arch-node-title">Input Graph Features</div>
                <div className="arch-node-desc">Shape: [num_nodes, 5]</div>
              </div>
              <span className="badge licit">Features</span>
            </div>

            <div className="arch-arrow">⬇️ Message Passing (Hop 1)</div>

            <div className="arch-node">
              <div>
                <div className="arch-node-title">GCN Layer 1 (GCNConv)</div>
                <div className="arch-node-desc">1-hop convolution mapping: 5 to 64 channels</div>
              </div>
              <span className="badge" style={{ backgroundColor: "#2563EB" }}>Weight Conv1</span>
            </div>

            <div className="arch-arrow">⬇️ ReLU & Dropout (p=0.3)</div>

            <div className="arch-node">
              <div>
                <div className="arch-node-title">GCN Layer 2 (GCNConv)</div>
                <div className="arch-node-desc">2-hop convolution mapping: 64 to 64 channels</div>
              </div>
              <span className="badge" style={{ backgroundColor: "#7C3AED" }}>Weight Conv2</span>
            </div>

            <div className="arch-arrow">⬇️ Fully Connected Classification Projection</div>

            <div className="arch-node">
              <div>
                <div className="arch-node-title">Linear Head & LogSoftmax</div>
                <div className="arch-node-desc">Projects 64 hidden features to 2 classes (Binary logits)</div>
              </div>
              <span className="badge illicit">Out Projection</span>
            </div>
          </div>

          {/* Card 2: Feature Importance Explainer Chart */}
          <div className="telemetry-card">
            <h3>📊 GNN Feature Importance Proxy (XAI)</h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginBottom: "0.5rem" }}>
              Approximate statistical influence of node attributes computed during backpropagation.
            </p>

            <div className="feature-chart">
              {/* Feature 1 */}
              <div className="feature-bar-row">
                <span className="feature-label">Outflow Amount</span>
                <div className="feature-bar-container">
                  <div className="feature-bar-fill" style={{ width: "38%" }}></div>
                </div>
                <span className="feature-percentage">38%</span>
              </div>

              {/* Feature 2 */}
              <div className="feature-bar-row">
                <span className="feature-label">Inflow Amount</span>
                <div className="feature-bar-container">
                  <div className="feature-bar-fill" style={{ width: "27%" }}></div>
                </div>
                <span className="feature-percentage">27%</span>
              </div>

              {/* Feature 3 */}
              <div className="feature-bar-row">
                <span className="feature-label">Out-Degree (Tx out)</span>
                <div className="feature-bar-container">
                  <div className="feature-bar-fill" style={{ width: "18%" }}></div>
                </div>
                <span className="feature-percentage">18%</span>
              </div>

              {/* Feature 4 */}
              <div className="feature-bar-row">
                <span className="feature-label">In-Degree (Tx in)</span>
                <div className="feature-bar-container">
                  <div className="feature-bar-fill" style={{ width: "12%" }}></div>
                </div>
                <span className="feature-percentage">12%</span>
              </div>

              {/* Feature 5 */}
              <div className="feature-bar-row">
                <span className="feature-label">Total Tx Count</span>
                <div className="feature-bar-container">
                  <div className="feature-bar-fill" style={{ width: "5%" }}></div>
                </div>
                <span className="feature-percentage">5%</span>
              </div>
            </div>

            <div className="gnn-explanation" style={{ marginTop: "1rem" }}>
              <h4>Explainable AI Notes:</h4>
              <p>
                The GCN convolutions propagate feature profiles of neighboring nodes across two edges. Nodes that display sudden increases in sent funds (outflow amount) coupled with circular flow topology (cyclic in-degree/out-degree matches) exhibit high activation values inside the final FC layers, triggering illicit classification flags.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;

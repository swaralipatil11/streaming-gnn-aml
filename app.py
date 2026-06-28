import os
import uuid
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict
from contextlib import asynccontextmanager

from dataset import PAYMENT_FORMATS

# Import database utilities
from database import (
    init_db,
    save_transaction,
    get_account_stats,
    save_account_stats,
    increment_account_stats,
    register_task,
    update_task_status,
    get_task_status
)

# Define Pydantic models for endpoints

class AnomalyRequest(BaseModel):
    x: List[List[float]]
    edge_index: List[List[int]]
    edge_attr: List[List[float]] = None

class AnomalyResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]

class Transaction(BaseModel):
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount: float
    payment_format: str

class TransactionBatchRequest(BaseModel):
    transactions: List[Transaction]

class TransactionBatchResponse(BaseModel):
    predictions: Dict[str, int]
    probabilities: Dict[str, float]

class AsyncBatchResponse(BaseModel):
    task_id: str
    status: str

# Global variables for the model
model = None
in_channels = 5
hidden_channels = 64
node_to_idx = {}
node_stats = {}
edge_dim = 9

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that initializes SQLite and loads the GNN model weights
    from disk during startup.
    """
    global model, node_to_idx, node_stats, in_channels, hidden_channels, edge_dim
    
    # 1. Initialize SQLite Database
    init_db()
    
    checkpoint_path = "aml_gcn_model.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"Loading pre-trained model weights from '{checkpoint_path}'...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            in_channels = checkpoint['in_channels']
            hidden_channels = checkpoint['hidden_channels']
            node_to_idx = checkpoint.get('node_to_idx', {})
            node_stats = checkpoint.get('node_stats', {})
            edge_dim = checkpoint.get('edge_dim', 9)
            
            from model import AMLGraphNet
            model = AMLGraphNet(in_channels=in_channels, hidden_channels=hidden_channels, edge_dim=edge_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            print("Model, node mapping, and statistics loaded successfully.")
            
            # Sync historical stats to SQLite if accounts table is empty
            import sqlite3
            conn = sqlite3.connect("aml_platform.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM accounts")
            count = cursor.fetchone()[0]
            conn.close()
            
            if count == 0 and node_stats:
                print("Syncing historical checkpoint stats to SQLite accounts table...")
                for acc_id, stats in node_stats.items():
                    # stats: [in_degree, out_degree, amount_sent, amount_received]
                    save_account_stats(acc_id, stats[0], stats[1], stats[2], stats[3], 0.0, 0)
                print(f"Synced {len(node_stats)} account profiles.")
                
        except Exception as e:
            print(f"Error loading model weights: {e}. Running with uninitialized model.")
            from model import AMLGraphNet
            model = AMLGraphNet(in_channels=in_channels, hidden_channels=hidden_channels, edge_dim=edge_dim)
            model.eval()
    else:
        print(f"WARNING: Checkpoint '{checkpoint_path}' not found. Starting with uninitialized model.")
        from model import AMLGraphNet
        model = AMLGraphNet(in_channels=in_channels, hidden_channels=hidden_channels, edge_dim=edge_dim)
        model.eval()
        
    yield

from fastapi.middleware.cors import CORSMiddleware

# Create FastAPI app
app = FastAPI(
    title="Relational GAT Anomaly & Anti-Money Laundering Detection API",
    description="High-throughput inference API using a 2-hop GAT with transaction edge features to classify accounts.",
    version="1.1.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

@app.post("/predict_anomaly", response_model=AnomalyResponse)
async def predict_anomaly(req: AnomalyRequest):
    """
    Accepts raw tensor features (x) and sparse coordinates (edge_index) of a graph,
    runs GAT in an evaluation context (torch.no_grad), and returns binary predictions.
    """
    global model, edge_dim
    if model is None:
        raise HTTPException(status_code=503, detail="GNN model is not loaded.")
        
    if not req.x or not req.edge_index:
        raise HTTPException(status_code=400, detail="x and edge_index cannot be empty.")

    try:
        x_tensor = torch.tensor(req.x, dtype=torch.float32)
        edge_index_tensor = torch.tensor(req.edge_index, dtype=torch.long)
        
        if req.edge_attr is not None:
            edge_attr_tensor = torch.tensor(req.edge_attr, dtype=torch.float32)
        else:
            edge_attr_tensor = torch.zeros((edge_index_tensor.size(1), edge_dim), dtype=torch.float32)
            
        with torch.no_grad():
            out = model(x_tensor, edge_index_tensor, edge_attr=edge_attr_tensor)
            probs = torch.exp(out)
            preds = out.argmax(dim=1)
            
        return AnomalyResponse(
            predictions=preds.tolist(),
            probabilities=probs.tolist()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference processing failed: {str(e)}")

def run_gnn_inference(transactions: List[Transaction]):
    """
    Synchronous helper to run the GAT model on a list of raw transactions
    and update the SQLite accounts database with the predictions.
    """
    global model, node_stats, edge_dim
    if model is None:
        raise ValueError("GNN model is not loaded.")

    # 1. Map account string IDs to a local subgraph index system
    account_set = set()
    for tx in transactions:
        src_id = f"{tx.from_bank}_{tx.from_account}"
        dst_id = f"{tx.to_bank}_{tx.to_account}"
        account_set.add(src_id)
        account_set.add(dst_id)
        
    unique_accounts = sorted(list(account_set))
    local_node_to_idx = {acc: i for i, acc in enumerate(unique_accounts)}
    num_nodes = len(unique_accounts)
    
    # 2. Extract subgraph node attributes (degrees, transaction amounts)
    in_degree = [0] * num_nodes
    out_degree = [0] * num_nodes
    amount_sent = [0.0] * num_nodes
    amount_received = [0.0] * num_nodes
    
    edge_sources = []
    edge_targets = []
    edge_features = []
    
    for tx in transactions:
        src_id = f"{tx.from_bank}_{tx.from_account}"
        dst_id = f"{tx.to_bank}_{tx.to_account}"
        
        src_idx = local_node_to_idx[src_id]
        dst_idx = local_node_to_idx[dst_id]
        
        edge_sources.append(src_idx)
        edge_targets.append(dst_idx)
        
        out_degree[src_idx] += 1
        in_degree[dst_idx] += 1
        amount_sent[src_idx] += tx.amount
        amount_received[dst_idx] += tx.amount
        
        # Save transaction log in SQLite database
        save_transaction(tx.from_bank, tx.from_account, tx.to_bank, tx.to_account, tx.amount, tx.payment_format)
        
        # Construct dynamic GAT edge attributes: [log1p(amount), one_hot_payment_formats (8 features)]
        amt_val = np.log1p(tx.amount)
        one_hot = [0.0] * 8
        if tx.payment_format in PAYMENT_FORMATS:
            fmt_idx = PAYMENT_FORMATS.index(tx.payment_format)
        else:
            fmt_idx = 7
        one_hot[fmt_idx] = 1.0
        edge_features.append([amt_val] + one_hot)
        
    # 3. Create node features (scaled with log1p for stability)
    x_features = []
    for idx in range(num_nodes):
        acc_id = unique_accounts[idx]
        # Query latest stats from SQLite database to capture dynamic updates
        row = get_account_stats(acc_id)
        if row:
            hist_in_deg = row["in_degree"]
            hist_out_deg = row["out_degree"]
            hist_sent = row["amount_sent"]
            hist_recv = row["amount_received"]
        else:
            # Fallback to checkpoint cache
            hist_in_deg, hist_out_deg, hist_sent, hist_recv = node_stats.get(acc_id, [0, 0, 0.0, 0.0])
            
        in_deg = hist_in_deg + in_degree[idx]
        out_deg = hist_out_deg + out_degree[idx]
        sent = hist_sent + amount_sent[idx]
        recv = hist_recv + amount_received[idx]
        
        x_features.append([
            np.log1p(in_deg),
            np.log1p(out_deg),
            np.log1p(sent),
            np.log1p(recv),
            np.log1p(in_deg + out_deg)
        ])
        
    x_tensor = torch.tensor(x_features, dtype=torch.float32)
    edge_index_tensor = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attr_tensor = torch.tensor(edge_features, dtype=torch.float32)
    
    # 4. GAT inference execution
    with torch.no_grad():
        out = model(x_tensor, edge_index_tensor, edge_attr=edge_attr_tensor)
        probs = torch.exp(out)
        preds = out.argmax(dim=1)
        
    predictions_dict = {}
    probabilities_dict = {}
    
    # 5. Persist dynamic updates to SQLite database & memory checkpoint cache
    for i, acc in enumerate(unique_accounts):
        pred_val = int(preds[i].item())
        prob_val = float(probs[i, 1].item())
        predictions_dict[acc] = pred_val
        probabilities_dict[acc] = prob_val
        
        # Persist dynamic update back to database
        increment_account_stats(
            account_id=acc,
            in_deg_inc=in_degree[i],
            out_deg_inc=out_degree[i],
            sent_inc=amount_sent[i],
            recv_inc=amount_received[i],
            risk_score=prob_val,
            is_illicit=pred_val
        )
        
        # Keep node_stats checkpoint dictionary updated in memory
        if acc in node_stats:
            node_stats[acc][0] += in_degree[i]
            node_stats[acc][1] += out_degree[i]
            node_stats[acc][2] += amount_sent[i]
            node_stats[acc][3] += amount_received[i]
        else:
            node_stats[acc] = [in_degree[i], out_degree[i], amount_sent[i], amount_received[i]]

    return predictions_dict, probabilities_dict

@app.post("/predict_transactions", response_model=TransactionBatchResponse)
async def predict_transactions(req: TransactionBatchRequest):
    """
    Accepts a streaming JSON batch of raw transactions, updates the relational GAT,
    performs synchronous inference, and stores aggregated node details in SQLite.
    """
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="GNN model is not loaded.")
        
    if not req.transactions:
        return TransactionBatchResponse(predictions={}, probabilities={})

    try:
        preds, probs = run_gnn_inference(req.transactions)
        return TransactionBatchResponse(predictions=preds, probabilities=probs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Transaction batch inference failed: {str(e)}")

def run_async_prediction_task(task_id: str, transactions: List[Transaction]):
    """Background task logic that runs inference and saves outputs in SQLite."""
    try:
        preds, probs = run_gnn_inference(transactions)
        result = {
            "predictions": preds,
            "probabilities": probs
        }
        update_task_status(task_id, "COMPLETED", result_json=result)
    except Exception as e:
        error_result = {"error": str(e)}
        update_task_status(task_id, "FAILED", result_json=error_result)

@app.post("/predict_transactions_async", response_model=AsyncBatchResponse, status_code=202)
async def predict_transactions_async(req: TransactionBatchRequest, background_tasks: BackgroundTasks):
    """
    Ingests transactions asynchronously. Immediately returns a Task ID (202 Accepted)
    and executes GNN prediction in a background worker task.
    """
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="GNN model is not loaded.")
        
    if not req.transactions:
        # Generate an empty completed task
        task_id = f"empty-{uuid.uuid4().hex}"
        register_task(task_id)
        update_task_status(task_id, "COMPLETED", {"predictions": {}, "probabilities": {}})
        return AsyncBatchResponse(task_id=task_id, status="COMPLETED")

    task_id = uuid.uuid4().hex
    register_task(task_id)
    background_tasks.add_task(run_async_prediction_task, task_id, req.transactions)
    return AsyncBatchResponse(task_id=task_id, status="PENDING")

@app.get("/task_status/{task_id}")
async def get_task(task_id: str):
    """Polls the status and results of an asynchronous ingestion batch."""
    status_info = get_task_status(task_id)
    if not status_info:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found.")
    return status_info

# Serve static React frontend files from root if they are built
dist_dir = "frontend/dist"
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")
else:
    @app.get("/")
    def read_root():
        """Fallback API status check when frontend build is not found."""
        # Query total nodes from database
        import sqlite3
        try:
            conn = sqlite3.connect("aml_platform.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM accounts")
            count = cursor.fetchone()[0]
            conn.close()
        except Exception:
            count = 0
            
        return {
            "status": "online",
            "model_loaded": model is not None,
            "nodes_registered_in_registry": count,
            "message": "React frontend built folder not found. Run 'npm run build' inside the 'frontend' folder to serve the visual dashboard."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict

class StreamGraphBuilder:
    """
    StreamGraphBuilder reads a massive transaction log CSV in chunks,
    dynamically maps string/hex node IDs into continuous integers,
    aggregates node and edge features, and constructs a PyTorch Geometric Data object.
    """
    def __init__(self, filepath: str, chunksize: int = 200000):
        self.filepath = filepath
        self.chunksize = chunksize
        self.node_to_idx = {}
        self.payment_formats = ['Reinvestment', 'Cheque', 'Credit Card', 'ACH', 'Cash', 'Wire', 'Bitcoin']
        self.format_to_idx = {fmt: idx for idx, fmt in enumerate(self.payment_formats)}

    def build_graph(self, max_rows: int = None) -> Data:
        """
        Builds and returns a PyTorch Geometric Data object from the CSV log.
        
        Args:
            max_rows: Optional limit on the number of rows to read (useful for quick local testing).
        """
        edge_sources = []
        edge_targets = []
        edge_amounts = []
        edge_formats = []

        # Node statistics accumulators
        node_in_degree = defaultdict(int)
        node_out_degree = defaultdict(int)
        node_amount_sent = defaultdict(float)
        node_amount_received = defaultdict(float)
        node_is_laundering = defaultdict(int)

        # Columns we need to load
        usecols = ['From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Payment Format', 'Is Laundering']
        
        print(f"Starting graph construction from {self.filepath} in chunks of {self.chunksize}...")
        
        chunk_iter = pd.read_csv(self.filepath, chunksize=self.chunksize, usecols=usecols)
        
        rows_processed = 0
        for chunk_idx, chunk in enumerate(chunk_iter):
            # Align column names to avoid duplication quirks in pandas
            chunk = chunk.copy()
            chunk.columns = ['From Bank', 'Account', 'To Bank', 'Account.1', 'Amount Received', 'Payment Format', 'Is Laundering']
            
            # Handle max_rows cutoff
            if max_rows is not None and rows_processed >= max_rows:
                print(f"Reached max_rows limit of {max_rows}. Stopping ingestion.")
                break
                
            chunk_len = len(chunk)
            if max_rows is not None and rows_processed + chunk_len > max_rows:
                chunk = chunk.iloc[:max_rows - rows_processed]
                chunk_len = len(chunk)
                
            rows_processed += chunk_len

            # 1. Vectorized Node Registration
            src_ids = chunk['From Bank'].astype(str) + '_' + chunk['Account'].astype(str)
            dst_ids = chunk['To Bank'].astype(str) + '_' + chunk['Account.1'].astype(str)
            
            unique_nodes = pd.concat([src_ids, dst_ids]).unique()
            new_nodes = set(unique_nodes) - set(self.node_to_idx.keys())
            
            # Deterministic sorting to keep indexing predictable
            for node in sorted(new_nodes):
                self.node_to_idx[node] = len(self.node_to_idx)
                
            # Map node IDs to indices
            src_indices = src_ids.map(self.node_to_idx).values
            dst_indices = dst_ids.map(self.node_to_idx).values
            
            # 2. Append Edges Info
            edge_sources.append(src_indices)
            edge_targets.append(dst_indices)
            
            amounts = pd.to_numeric(chunk['Amount Received'], errors='coerce').fillna(0.0).values
            edge_amounts.append(amounts)
            
            formats = chunk['Payment Format'].map(self.format_to_idx).fillna(len(self.payment_formats)).astype(int).values
            edge_formats.append(formats)
            
            # 3. Aggregate Node Statistics (Vectorized groupby per chunk to minimize iterations)
            chunk['src_node_id'] = src_ids
            chunk['dst_node_id'] = dst_ids
            chunk['amount_numeric'] = amounts
            chunk['laundering_numeric'] = pd.to_numeric(chunk['Is Laundering'], errors='coerce').fillna(0).astype(int).values
            
            src_stats = chunk.groupby('src_node_id').agg(
                out_degree=('amount_numeric', 'count'),
                amount_sent=('amount_numeric', 'sum'),
                is_laundering_sent=('laundering_numeric', 'max')
            )
            dst_stats = chunk.groupby('dst_node_id').agg(
                in_degree=('amount_numeric', 'count'),
                amount_received=('amount_numeric', 'sum'),
                is_laundering_recv=('laundering_numeric', 'max')
            )
            
            # Update global node stats dictionaries
            for node_id, row in src_stats.iterrows():
                idx = self.node_to_idx[node_id]
                node_out_degree[idx] += row['out_degree']
                node_amount_sent[idx] += row['amount_sent']
                if row['is_laundering_sent'] == 1:
                    node_is_laundering[idx] = 1

            for node_id, row in dst_stats.iterrows():
                idx = self.node_to_idx[node_id]
                node_in_degree[idx] += row['in_degree']
                node_amount_received[idx] += row['amount_received']
                if row['is_laundering_recv'] == 1:
                    node_is_laundering[idx] = 1
                    
            print(f"Chunk {chunk_idx + 1}: Processed {rows_processed} rows. Total unique nodes registered: {len(self.node_to_idx)}.")
            
        # Compile final node arrays
        num_nodes = len(self.node_to_idx)
        print(f"Graph construction completed. Total nodes: {num_nodes}, Total edges: {rows_processed}.")
        
        # Build node feature matrix x
        # Features: [log1p(in_degree), log1p(out_degree), log1p(amount_sent), log1p(amount_received), log1p(total_tx_count)]
        x = np.zeros((num_nodes, 5), dtype=np.float32)
        y = np.zeros(num_nodes, dtype=np.int64)
        
        for idx in range(num_nodes):
            in_deg = node_in_degree[idx]
            out_deg = node_out_degree[idx]
            sent_amt = node_amount_sent[idx]
            recv_amt = node_amount_received[idx]
            
            x[idx, 0] = np.log1p(in_deg)
            x[idx, 1] = np.log1p(out_deg)
            x[idx, 2] = np.log1p(sent_amt)
            x[idx, 3] = np.log1p(recv_amt)
            x[idx, 4] = np.log1p(in_deg + out_deg)
            
            y[idx] = node_is_laundering[idx]
            
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Assemble edge indices
        edge_index = np.vstack([np.concatenate(edge_sources), np.concatenate(edge_targets)])
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        
        # Assemble edge attributes: log1p(amount) + one-hot encoded payment format
        edge_amounts_arr = np.log1p(np.concatenate(edge_amounts))
        edge_formats_arr = np.concatenate(edge_formats)
        
        # Number of formats is len(payment_formats) + 1 (for 'Unknown')
        num_formats = len(self.payment_formats) + 1
        one_hot_formats = np.zeros((len(edge_formats_arr), num_formats), dtype=np.float32)
        one_hot_formats[np.arange(len(edge_formats_arr)), edge_formats_arr] = 1.0
        
        edge_attr = np.hstack([edge_amounts_arr[:, np.newaxis], one_hot_formats])
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
        
        # Create PyG Data object
        data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y_tensor)
        return data

if __name__ == "__main__":
    # Quick debug run
    import sys
    path = "d:/AML/dataset/HI-Small_Trans.csv"
    builder = StreamGraphBuilder(path)
    data = builder.build_graph(max_rows=10000)
    print("Graph Info:")
    print(data)
    print("Node feature shape:", data.x.shape)
    print("Node label sum (laundering count):", data.y.sum().item())

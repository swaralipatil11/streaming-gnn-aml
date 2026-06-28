import subprocess
import time
import sys
import os
import requests
import json

def run_integration_test():
    print("=== STEP 1: Running Model Training on a subset ===")
    # Train model on a subset of data (100,000 rows, 5 epochs) to generate aml_gcn_model.pth
    train_cmd = [
        sys.executable, "train.py",
        "--max_rows", "100000",
        "--epochs", "5",
        "--save_path", "aml_gcn_model.pth",
        "--cpu"
    ]
    print(f"Running command: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Model training failed!")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        sys.exit(1)
    print("Model training finished successfully.")
    
    # Confirm model weights file is present
    if not os.path.exists("aml_gcn_model.pth"):
        print("Error: aml_gcn_model.pth was not created!")
        sys.exit(1)
    print("Confirmed aml_gcn_model.pth exists.")
    
    print("\n=== STEP 2: Launching FastAPI server on port 8085 ===")
    server_process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app:app",
        "--host", "127.0.0.1",
        "--port", "8085"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Wait for server to startup using a robust retry loop (up to 15 seconds)
    print("Waiting for server to start and become responsive...")
    base_url = "http://127.0.0.1:8085"
    server_ready = False
    
    for i in range(15):
        time.sleep(1)
        # Check if server process is still alive
        if server_process.poll() is not None:
            print("Error: FastAPI server failed to start!")
            stdout, stderr = server_process.communicate()
            print("stdout:", stdout)
            print("stderr:", stderr)
            sys.exit(1)
            
        try:
            response = requests.get(f"{base_url}/", timeout=1)
            if response.status_code == 200:
                server_ready = True
                print(f"Server is responsive after {i+1} seconds.")
                break
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            pass
            
    if not server_ready:
        print("Error: FastAPI server did not become responsive within timeout!")
        server_process.terminate()
        server_process.wait()
        sys.exit(1)
    
    try:
        # 1. Test Root Endpoint
        print("\nChecking Root Endpoint...")
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        assert response.status_code == 200
        
        # Root endpoint can return either JSON status payload or HTML React client
        content_type = response.headers.get("Content-Type", "")
        if "application/json" in content_type:
            data = response.json()
            print(f"Response (JSON): {data}")
            assert data["model_loaded"] is True
        else:
            print("Response served HTML frontend dashboard (production mode).")
            assert "<html" in response.text.lower() or "<div id=\"root\"" in response.text.lower()
        
        # 2. Test predict_anomaly Endpoint (Raw Graph Tensors)
        print("\nChecking /predict_anomaly Endpoint...")
        anomaly_payload = {
            "x": [
                [0.0, 0.0, 0.0, 0.0, 0.0],  # node 0 features
                [1.0, 1.0, 100.0, 100.0, 2.0],  # node 1 features
                [2.0, 0.5, 50.0, 10.0, 1.0]  # node 2 features
            ],
            "edge_index": [
                [0, 1, 1],  # source nodes
                [1, 2, 0]   # target nodes
            ]
        }
        response = requests.post(f"{base_url}/predict_anomaly", json=anomaly_payload)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {data}")
        assert response.status_code == 200
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        assert len(data["probabilities"]) == 3
        
        # 3. Test predict_transactions Endpoint (Raw Streaming JSON)
        print("\nChecking /predict_transactions Endpoint...")
        transaction_payload = {
            "transactions": [
                {
                    "from_bank": "123",
                    "from_account": "ACC_A",
                    "to_bank": "456",
                    "to_account": "ACC_B",
                    "amount": 1250.50,
                    "payment_format": "Wire"
                },
                {
                    "from_bank": "456",
                    "from_account": "ACC_B",
                    "to_bank": "789",
                    "to_account": "ACC_C",
                    "amount": 10.0,
                    "payment_format": "Cash"
                }
            ]
        }
        response = requests.post(f"{base_url}/predict_transactions", json=transaction_payload)
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {data}")
        assert response.status_code == 200
        assert "predictions" in data
        assert "123_ACC_A" in data["predictions"]
        assert "456_ACC_B" in data["predictions"]
        assert "789_ACC_C" in data["predictions"]
        
        # 4. Test predict_transactions_async and task_status Endpoint (Asynchronous Queue)
        print("\nChecking Asynchronous Endpoints...")
        async_response = requests.post(f"{base_url}/predict_transactions_async", json=transaction_payload)
        print(f"Status: {async_response.status_code}")
        assert async_response.status_code == 202
        async_data = async_response.json()
        print(f"Response: {async_data}")
        assert "task_id" in async_data
        task_id = async_data["task_id"]
        
        # Poll status
        completed = False
        for _ in range(10):
            time.sleep(0.5)
            status_response = requests.get(f"{base_url}/task_status/{task_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()
            print(f"Polling status: {status_data['status']}")
            if status_data["status"] == "COMPLETED":
                completed = True
                assert "result" in status_data
                assert "predictions" in status_data["result"]
                assert "123_ACC_A" in status_data["result"]["predictions"]
                break
            elif status_data["status"] == "FAILED":
                print("Task failed:", status_data["result"])
                break
                
        assert completed is True
        
        print("\n=== INTEGRATION TESTS PASSED SUCCESSFULLY ===")
        
    except AssertionError as e:
        print(f"\nVerification Assertion Failed! {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nIntegration test encountered an unexpected error: {e}")
        sys.exit(1)
    finally:
        print("\nShutting down FastAPI test server...")
        server_process.terminate()
        server_process.wait()
        print("Server shutdown complete.")

if __name__ == "__main__":
    run_integration_test()

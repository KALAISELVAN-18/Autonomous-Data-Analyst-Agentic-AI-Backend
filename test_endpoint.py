import requests
import json
import os

def test():
    # 1. Upload
    os.makedirs("data", exist_ok=True)
    with open("data/test.csv", "w") as f:
        f.write("A,B\n1,2\n3,4")
    
    with open("data/test.csv", "rb") as f:
        r1 = requests.post("http://localhost:8000/api/v1/upload", files={"file": ("test.csv", f, "text/csv")})
    
    print("Upload response:", r1.status_code, r1.text)
    session_id = r1.json()["session_id"]

    # 2. Analyze
    r2 = requests.post(f"http://localhost:8000/api/v1/analyze?session_id={session_id}&prompt=test")
    print("Analyze response:", r2.status_code, r2.text)

if __name__ == "__main__":
    test()

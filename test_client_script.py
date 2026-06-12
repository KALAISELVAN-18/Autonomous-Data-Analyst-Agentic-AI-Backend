from fastapi.testclient import TestClient
from src.api.routes import router
from main import app
import os

client = TestClient(app)

def test_exception():
    os.makedirs("data", exist_ok=True)
    with open("data/test.csv", "w") as f:
        f.write("A,B\n1,2\n3,4")
    
    with open("data/test.csv", "rb") as f:
        r1 = client.post("/api/v1/upload", files={"file": ("test.csv", f, "text/csv")})
    print(r1.json())
    session_id = r1.json()["session_id"]

    try:
        r2 = client.post(f"/api/v1/analyze?session_id={session_id}&prompt=test")
        print(r2.status_code)
        print(r2.json())
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_exception()

import requests
import json

def test_simple():
    """Simple test with clean output"""
    
    url = "http://localhost:8000/hackrx/run"
    
    data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer d742ec2aaf3cd69400711966ec8db56a156c9f0404f7cce41808e3c6e9ede8c8"
    }
    
    print("Testing API...")
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ SUCCESS\n")
            
            for i, (question, answer) in enumerate(zip(data['questions'], result['answers']), 1):
                print(f"Q{i}: {question}")
                print(f"A{i}: {answer}\n")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_simple()

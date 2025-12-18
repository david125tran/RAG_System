import json
import sys
import requests

API_URL = "http://127.0.0.1:8000/api/chat"

def chat(message: str, history: list) -> str:
    # Your endpoint expects multipart/form-data with fields "message" and "history"
    data = {
        "message": message,
        "history": json.dumps(history),
    }

    r = requests.post(API_URL, data=data, timeout=120)
    r.raise_for_status()

    payload = r.json()
    return payload.get("reply", "")

def main():
    history = []  # keep as list of {"role": "...", "content": "..."} or {"from": "...", "text": "..."}
    print("Chat client. Type 'exit' to quit.\n")

    while True:
        user_msg = input("You: ").strip()
        if not user_msg:
            continue
        if user_msg.lower() in {"exit", "quit"}:
            break

        try:
            reply = chat(user_msg, history)
        except requests.RequestException as e:
            print(f"\n[HTTP error] {e}\n")
            continue

        print(f"\nBot: {reply}\n")

        # store history in the format your converter supports
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

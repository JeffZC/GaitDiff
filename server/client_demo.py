import requests

SERVER = "http://127.0.0.1:8000"

def init_with_file(path):
    files = {"file": open(path, "rb")}
    r = requests.post(SERVER + "/init", files=files)
    print(r.status_code, r.text)
    return r.json()


def send_message(convo_id, message):
    r = requests.post(SERVER + "/message", json={"conversation_id": convo_id, "message": message})
    print(r.status_code, r.text)
    return r.json()


if __name__ == "__main__":
    # Example usage: run the server, then run this script
    res = init_with_file("runs/20260120_225240/results.json")
    cid = res.get("conversation_id")
    if cid:
        out = send_message(cid, "Explain the step_length difference between video A and B.")
        print("Reply:\n", out.get("reply"))

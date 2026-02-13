import requests

r=requests.post(
 "http://127.0.0.1:8000/ask",
 json={"question":"what is aviation","debug":True}
)

print(r.json())

import requests

query = {"question": "what's the cheapest accomodation type and how much its costs ?."}
res = requests.post("http://127.0.0.1:8000/query", json=query)


print(res.json())
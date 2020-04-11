import requests

API_ENDPOINT = "http://34.231.225.65/similar"


def call_covid_api(sentences, method="union", limit=10):
    payload = { "method": method, "limit": limit}
    data = {"sentences": sentences}
    res = requests.post(API_ENDPOINT, params=payload, json=data)
    return res.json()

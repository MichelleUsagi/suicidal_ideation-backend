import requests

url = " https://suicide-api-hnu6.onrender.com/predict"

data = {
    "text": "I feel tired and overwhelmed with life"
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
try:
    print("Response:", response.json())
except Exception as e:
    print("Raw Response Text:", response.text)
    print("Error:", e)

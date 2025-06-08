import json

# a Python object (dict):
x = {
  "name": "John",
  "age": 30,
  "city": "New York"
}

# convert into JSON:
y = json.dumps(x)
with open('data.json', 'w') as f:
    json.dump(y, f)
# the result is a JSON string:
print(y)
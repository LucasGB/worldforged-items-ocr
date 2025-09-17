import json
import boto3
from dynamodb_json import json_util as ddjson
from datetime import datetime

with open("data/Arathi Highlands.json", "r", encoding="utf-8") as f:
    items = json.load(f)

for item in items:
    now = datetime.now()
    item["createdAt"] = now.strftime("%d-%m-%Y %H:%M:%S")
    item["updatedAt"] = item["createdAt"]

# Converte para formato DynamoDB JSON (dicts prontos para client)
dynamodb_items = [json.loads(ddjson.dumps(item)) for item in items]

client = boto3.client("dynamodb")

# batch_write_item só aceita no máximo 25 itens por request
BATCH_SIZE = 25
for i in range(0, len(dynamodb_items), BATCH_SIZE):
    batch = dynamodb_items[i:i + BATCH_SIZE]

    response = client.batch_write_item(
        RequestItems={
            "GameItems": [{"PutRequest": {"Item": item}} for item in batch]
        }
    )

    if response.get("UnprocessedItems"):
        print(f"⚠️ Alguns itens não foram processados: {response['UnprocessedItems']}")

print(f"Successfully wrote {len(dynamodb_items)} items to GameItems")

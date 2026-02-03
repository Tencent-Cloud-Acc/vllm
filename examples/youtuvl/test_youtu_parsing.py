#!/usr/bin/env python3
import base64, json, requests, time

API = "http://localhost:8000/v1/chat/completions"
MODEL = "/path/to/Youtu-Parsing"
IMAGE = "/path/to/test.png"

b64 = base64.b64encode(open(IMAGE, "rb").read()).decode()
payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}],
    "stream": False,
    "max_tokens": 4096,
    "temperature": 0,
    "top_p": 0.3,
    "repetition_penalty": 1.0,
    "stop_token_ids": [128001],
    "youtuvl_mode": "document",
    "parse_mode": "sequential",  
    "ocr_batch_size": 1, 
    "mm_processor_kwargs": {
        "max_num_patches": 4096
    }
}
start = time.time()
r = requests.post(API, json=payload)
end = time.time()

r.raise_for_status()
print(json.dumps(json.loads(r.json()["choices"][0]["message"]["content"]), ensure_ascii=False, indent=2))

print("time taken: ", end - start)

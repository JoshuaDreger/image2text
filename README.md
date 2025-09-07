# image2text
A simple image to text tool


docker compose build image2text
docker compose up -d image2text


python -m venv .venv && source .venv/bin/activate
pip install -r requirements.image2text.txt
export OUTPUT_DIR=/tmp/i2t
python -m uvicorn image2text.api:app --reload --port 8001
# In another shell:
streamlit run image2text/web_app.py --server.port 8502



curl -s -X POST "http://localhost:8001/api/describe" \
  -F "files=@/path/to/photo.jpg" \
  -F 'prompt=Please describe the scene and any text in the sign.' \
  | jq .

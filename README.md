# Texte auf inhaltliche Ähnlichkeit vergleichen mit OpenAI text embedding

! You'll have to add your own paid OpenAI Api-Key in .env


# Setup
```
git clone https://github.com/Ollilauch/compare_texts.git
cd compare_texts
touch .env
add OPENAI_API_KEY="<YOUR_API_KEY>" in .env without <>
```

# Running
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 main.py <path to textfile1> <path to textfile2> ...
```


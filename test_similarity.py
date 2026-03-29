from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

labels = ["neutro", "urgente", "emergencia", "triste", "ansioso", "feliz", "frustrado", "raiva", "medo", "preocupado", "dor"]
texts = [
    "como podemos resolver o problema do aquecimento global , estou preocupado", 
    "o sistema caiu de vez, preciso de ajuda urgente!", 
    "oi tudo bem",
    "hoje o dia esta maravilhoso",
    "estou sentindo uma dor no peito e muito medo",
    "eu não sei mais o que fazer, me ajuda"
]

label_embs = model.encode(labels)
text_embs = model.encode(texts)

for i, text in enumerate(texts):
    print(f"\nText: {text}")
    sims = []
    for j, label in enumerate(labels):
        sim = np.dot(text_embs[i], label_embs[j]) / (np.linalg.norm(text_embs[i]) * np.linalg.norm(label_embs[j]))
        sims.append((label, sim))
    
    sims.sort(key=lambda x: x[1], reverse=True)
    for label, sim in sims[:4]:
        print(f"  {label}: {sim:.3f}")

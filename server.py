from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import random

# ------------------------------------
# Anaya Brain V2
# ------------------------------------

class AnayaBrainV2(nn.Module):
    def __init__(self, input_dim=399, hidden_dim=64, num_actions=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ------------------------------------
# Anaya V2
# ------------------------------------

class AnayaV2:
    ACTIONS = [
        "cry", "smile", "coo", "babble", "look_around",
        "look_away", "reach_out", "kick_legs", "stay_calm", "fuss"
    ]

    def __init__(self):
        # personality
        self.calmness = random.uniform(0.3, 0.9)
        self.sensitivity = random.uniform(0.3, 0.9)
        self.curiosity_level = random.uniform(0.3, 0.9)
        self.sociability = random.uniform(0.3, 0.9)
        self.fearfulness = random.uniform(0.3, 0.9)

        # internal state
        self.comfort = 0.5
        self.energy = 0.5
        self.mood = 0.5
        self.curiosity = 0.5
        self.stress = 0.3
        self.age_steps = 0

        # memory
        self.mem_happy = 0.5
        self.mem_fear = 0.5
        self.mem_calm = 0.5
        self.mem_stim = 0.5
        self.mem_comfort = 0.5

        # brain
        self.brain = AnayaBrainV2()
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)

        # text encoder
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def _clip(self, x):
        return max(0.0, min(1.0, x))

    def get_internal_vec(self):
        return torch.tensor([self.comfort, self.energy, self.mood,
                             self.curiosity, self.stress], dtype=torch.float32)

    def get_memory_vec(self):
        return torch.tensor([self.mem_happy, self.mem_fear, self.mem_calm,
                             self.mem_stim, self.mem_comfort], dtype=torch.float32)

    def get_personality_vec(self):
        return torch.tensor([self.calmness, self.sensitivity, self.curiosity_level,
                             self.sociability, self.fearfulness], dtype=torch.float32)

    def apply_experience(self, text):
        t = text.lower()
        self.age_steps += 1

        if "hug" in t or "smile" in t or "cuddle" in t:
            self.comfort = self._clip(self.comfort + 0.1)
            self.mood = self._clip(self.mood + 0.1)

        if any(w in t for w in ["toy", "light", "play"]):
            self.curiosity = self._clip(self.curiosity + 0.1)

        if any(w in t for w in ["loud", "bark", "shout", "angry"]):
            self.stress = self._clip(self.stress + 0.15)
            self.comfort = self._clip(self.comfort - 0.1)

        emb = self.encoder.encode(text, convert_to_tensor=True).float()

        full_vec = torch.cat([
            self.get_internal_vec(),
            self.get_memory_vec(),
            self.get_personality_vec(),
            emb
        ])

        with torch.no_grad():
            scores = self.brain(full_vec)
            idx = torch.argmax(scores).item()

        return self.ACTIONS[idx]

# ------------------------------------
# FastAPI
# ------------------------------------

app = FastAPI()
anaya = AnayaV2()

class Event(BaseModel):
    text: str

@app.post("/event")
def event(req: Event):
    action = anaya.apply_experience(req.text)
    return {"action": action, "state": "ok"}

@app.get("/")
def root():
    return {"message": "Anaya Baby API is running!"}

# local run support
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8080)

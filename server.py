import json
import random
from collections import deque
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer


# =========================
#   ANAYA BRAIN V2
# =========================

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


# =========================
#   ANAYA BABY V2
# =========================

class AnayaV2:
    ACTIONS = [
        "cry", "smile", "coo", "babble", "look_around",
        "look_away", "reach_out", "kick_legs", "stay_calm", "fuss"
    ]

    def __init__(self):
        # Personality
        self.calmness = random.uniform(0.3, 0.9)
        self.sensitivity = random.uniform(0.3, 0.9)
        self.curiosity_level = random.uniform(0.3, 0.9)
        self.sociability = random.uniform(0.3, 0.9)
        self.fearfulness = random.uniform(0.3, 0.9)

        # Internal state
        self.comfort = 0.5
        self.energy = 0.5
        self.mood = 0.5
        self.curiosity = 0.5
        self.stress = 0.3
        self.age_steps = 0

        # Short-term memory
        self.mem_happy = 0.5
        self.mem_fear = 0.5
        self.mem_calm = 0.5
        self.mem_stim = 0.5
        self.mem_comfort = 0.5

        # Brain
        self.brain = AnayaBrainV2()
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)

        # Text embedding model
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # ----- helpers -----
    def _clip(self, x: float) -> float:
        return max(0.0, min(1.0, x))

    def get_state(self):
        return {
            "comfort": self.comfort,
            "energy": self.energy,
            "mood": self.mood,
            "curiosity": self.curiosity,
            "stress": self.stress,
            "mem_happy": self.mem_happy,
            "mem_fear": self.mem_fear,
            "mem_calm": self.mem_calm,
            "mem_stim": self.mem_stim,
            "mem_comfort": self.mem_comfort,
            "calmness": self.calmness,
            "sensitivity": self.sensitivity,
            "curiosity_level": self.curiosity_level,
            "sociability": self.sociability,
            "fearfulness": self.fearfulness,
            "age_steps": self.age_steps,
        }

    def get_internal_vec(self):
        return torch.tensor(
            [self.comfort, self.energy, self.mood, self.curiosity, self.stress],
            dtype=torch.float32,
        )

    def get_memory_vec(self):
        return torch.tensor(
            [self.mem_happy, self.mem_fear, self.mem_calm, self.mem_stim, self.mem_comfort],
            dtype=torch.float32,
        )

    def get_personality_vec(self):
        return torch.tensor(
            [
                self.calmness,
                self.sensitivity,
                self.curiosity_level,
                self.sociability,
                self.fearfulness,
            ],
            dtype=torch.float32,
        )

    # ----- memory -----
    def _update_memory(self, text: str):
        t = text.lower()
        decay = 0.9

        for name in ["mem_happy", "mem_fear", "mem_calm", "mem_stim", "mem_comfort"]:
            val = getattr(self, name)
            setattr(self, name, 0.5 + decay * (val - 0.5))

        if any(w in t for w in ["hug", "love", "smile", "cuddle", "feed", "milk"]):
            self.mem_happy = self._clip(self.mem_happy + 0.1)
            self.mem_comfort = self._clip(self.mem_comfort + 0.1)
            self.mem_calm = self._clip(self.mem_calm + 0.05)

        if any(w in t for w in ["toy", "shiny", "play", "light", "rattle", "peekaboo"]):
            self.mem_stim = self._clip(self.mem_stim + 0.1)

        if any(w in t for w in ["loud", "bark", "shout", "angry", "bang", "scream"]):
            self.mem_fear = self._clip(self.mem_fear + 0.1)

    # ----- main experience -----
    def apply_experience(self, text: str) -> str:
        self.age_steps += 1
        t = text.lower()

        # simple emotional rules
        if any(w in t for w in ["hug", "kiss", "soft", "smile", "cuddle"]):
            self.comfort = self._clip(self.comfort + 0.1)
            self.mood = self._clip(self.mood + 0.1)
            self.stress = self._clip(self.stress - 0.05)

        if any(w in t for w in ["toy", "play", "rattle", "light", "shiny"]):
            self.curiosity = self._clip(self.curiosity + 0.1 * self.curiosity_level)

        if any(w in t for w in ["loud", "bark", "angry", "shout", "bang"]):
            self.stress = self._clip(self.stress + 0.1 * self.sensitivity)
            self.comfort = self._clip(self.comfort - 0.1)
            self.mood = self._clip(self.mood - 0.05)

        if "alone" in t:
            self.comfort = self._clip(self.comfort - 0.1)
            self.stress = self._clip(self.stress + 0.1 * self.fearfulness)

        self._update_memory(text)

        with torch.no_grad():
            emb = self.text_encoder.encode(text, convert_to_tensor=True).float()

        full_vec = torch.cat(
            [self.get_internal_vec(), self.get_memory_vec(), self.get_personality_vec(), emb]
        )

        with torch.no_grad():
            scores = self.brain(full_vec)
            idx = torch.argmax(scores).item()

        return self.ACTIONS[idx]


# =========================
#   FASTAPI APP
# =========================

app = FastAPI(title="Anaya AI Baby API")

# CORS for browser & apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production you can restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

anaya = AnayaV2()
replay_buffer = deque(maxlen=1000)
EXPERIENCE_LOG_PATH = "anaya_experience_log.json"

try:
    with open(EXPERIENCE_LOG_PATH, "r") as f:
        experience_log = json.load(f)
except FileNotFoundError:
    experience_log = []


class EventReq(BaseModel):
    text: str


class FeedbackReq(BaseModel):
    correct_action: Optional[str] = None
    event_text: Optional[str] = None
    action_taken: Optional[str] = None


@app.get("/")
def root():
    return {"message": "Anaya AI Baby backend is running."}


@app.post("/event")
def event(req: EventReq):
    action = anaya.apply_experience(req.text)

    log = {
        "event": req.text,
        "action": action,
        "state": anaya.get_state(),
    }
    experience_log.append(log)
    with open(EXPERIENCE_LOG_PATH, "w") as f:
        json.dump(experience_log, f, indent=2)

    return {"action": action, "state": anaya.get_state()}


@app.post("/feedback")
def feedback(req: FeedbackReq):
    correct = req.correct_action or req.action_taken
    if not correct or not req.event_text:
        return {"message": "missing correct_action or event_text"}

    with torch.no_grad():
        emb = anaya.text_encoder.encode(
            req.event_text, convert_to_tensor=True
        ).float()

    full_vec = torch.cat(
        [
            anaya.get_internal_vec(),
            anaya.get_memory_vec(),
            anaya.get_personality_vec(),
            emb,
        ]
    ).tolist()

    target_idx = AnayaV2.ACTIONS.index(correct)
    replay_buffer.append((full_vec, target_idx))

    return {"message": "feedback saved"}


@app.get("/state")
def state():
    return anaya.get_state()


@app.post("/train")
def train(epochs: int = 50, batch_size: int = 32):
    if len(replay_buffer) < batch_size:
        return {"message": f"not enough samples, have {len(replay_buffer)}"}

    for _ in range(epochs):
        batch = random.sample(replay_buffer, batch_size)
        x = torch.tensor([b[0] for b in batch], dtype=torch.float32)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long)

        pred = anaya.brain(x)
        loss = F.cross_entropy(pred, y)

        anaya.optimizer.zero_grad()
        loss.backward()
        anaya.optimizer.step()

    return {"message": "training complete"}

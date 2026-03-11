import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import threading

from .train import CrackTrainTool

app = FastAPI()

trainer = None
current_epoch = 0
lock = threading.Lock()


class TrainStartReq(BaseModel):
    config_path: str


@app.get("/", response_class=HTMLResponse)
def index():
    with open("../view/train.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/train_start")
def train_start(config: dict):
    global trainer, current_epoch

    config_path = "runtime_config.yaml"

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)

    with lock:
        trainer = CrackTrainTool(config_path)
        current_epoch = 1

    return {
        "msg": "training initialized",
        "total_epoch": trainer.epochs
    }


@app.post("/train_epoch")
def train_epoch():
    global trainer, current_epoch

    if trainer is None:
        return {"error": "training not started"}

    if current_epoch > trainer.epochs:
        return {"msg": "training finished"}

    loss, dice, iou = trainer.train_epoch(current_epoch)

    result = {
        "epoch": current_epoch,
        "loss": float(loss),
        "dice": float(dice),
        "iou": float(iou)
    }

    current_epoch += 1

    return result
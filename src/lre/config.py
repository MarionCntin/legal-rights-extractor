from pathlib import Path
from pydantic import BaseModel

class Settings(BaseModel):
    raw_dir: Path = Path("data/raw")
    out_dir: Path = Path("data/out")

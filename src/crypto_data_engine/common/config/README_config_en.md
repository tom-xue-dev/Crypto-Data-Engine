# ⚙️ Configuration System (Pydantic + YAML)

This module provides a clean, modular configuration system based on [pydantic](https://docs.pydantic.dev) and YAML.

## ✅ Features

- Centralized configuration class definitions
- Auto-generate configuration YAML templates with default values
- Clean separation: definition ≠ loading ≠ template generation
- Human-readable, strongly typed, and extensible

---

## 🗂️ File Structure

```
config/
├── config_settings.py              # Config class definitions + template init
├── load_config.py                  # General YAML loader for config instances
├── config_templates/               # Auto-generated YAML templates
│   ├── data_scraper_config.yaml
│   └── ray_config.yaml
```

---

## 📐 Configuration Definition

Defined in `config_settings.py` using Pydantic BaseModel:

```python
class RayConfig(BaseModel):
    address: Optional[str] = None
    num_cpus: Optional[int] = None
    ...

class DataScraperConfig(BaseModel):
    source_url: str = "https://example.com/api"
    cache_days: int = 30
```

---

## 🛠 Generate YAML Templates

To auto-generate config files with default values:

```bash
python config_settings.py --init
```

Output:

- `config_templates/ray_config.yaml`
- `config_templates/data_scraper_config.yaml`

File names follow: `ClassName` → `snake_case` → `.yaml`

---

## 📥 Load Configuration at Runtime

Use `load_config()` from `load_config.py`:

```python
from load_config import load_config
from config_settings import DataScraperConfig, RayConfig

scraper_cfg = load_config(DataScraperConfig)
ray_cfg = load_config(RayConfig)

print(scraper_cfg.source_url)
print(ray_cfg.num_cpus)
```

---

## ✅ Design Benefits

| Feature | Description |
|--------|-------------|
| ✅ Decoupled loading | No side effects on import |
| ✅ Centralized class registry | Easy to add new config types |
| ✅ Strong typing with defaults | Clear structure and safe |
| ✅ Compatible with services | Import configs from a central point |

---

## 🔧 Future Improvements (TODO)

- [ ] `.env` support and environment switching
- [ ] Hot reload (watch mode)
- [ ] Config validation and hints
- [ ] Dev/test/prod mode management

---

## 🧠 Pro Tip

To add a new config:
- Define a new Pydantic class in `config_settings.py`
- Add its instance to the list in `create_all_templates()`

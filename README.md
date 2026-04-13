# ml-stand-cicd

Экспериментальный стенд для проверки `pymlsec` в GitHub Actions CI/CD pipeline.

## Состав стенда

- Titanic dataset
- обучение `RandomForestClassifier`
- сохранение модели в `joblib`
- FastAPI API
- pytest
- Docker
- аудит зависимостей через `pymlsec`

## Локальный запуск

### 1. Установка зависимостей

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

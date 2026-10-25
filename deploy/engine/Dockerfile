# The website's engine (tools/engine_server.py) on Cloud Run. Build folder: python deploy/engine/assemble.py
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PXR_USDC_EMIT_DEPRECATION_WARNINGS=0 FRT_NO_BODIES=1 \
    FRT_ENGINE_WORKER=/app/assets/worker.glb FRT_ENGINE_SOLVERS=4
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# a quick self-test while building: the character loads and the skin measurement is found (no Blender here)
RUN python -c "import sys; sys.path[:0]=['tools','exts/frt.core']; from frt_core.people.mixamo import load; \
c = load('/app/assets/worker.glb'); assert c.skeleton.torso, 'skin cache missing'; print('worker ok', c.id)"
CMD ["python", "tools/engine_server.py"]

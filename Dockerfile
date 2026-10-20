FROM python:3.12-slim AS build
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
RUN useradd --create-home app
COPY --from=build /install /usr/local
WORKDIR /app
COPY visrag ./visrag
USER app
EXPOSE 8000
# pass the key at run time:  docker run -e GROQ_API_KEY=... -p 8000:8000 visloc-rag
CMD ["uvicorn", "visrag.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.9-slim
WORKDIR /app
COPY Pipfile Pipfile.lock ./
RUN pip install pipenv && pipenv install --system
COPY ./app ./app
COPY ./tests ./tests
CMD ["pipenv", "run", "python", "-m", "app.emotion_detector"]
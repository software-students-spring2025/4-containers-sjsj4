FROM python:3.9-slim
WORKDIR /app
COPY Pipfile Pipfile.lock ./
RUN pip install pipenv && pipenv install --system
COPY ./app ./app
COPY ./tests ./tests
EXPOSE 5000
CMD ["pipenv", "run", "python", "-m", "flask", "run", "--host=0.0.0.0"]
FROM python:3.9.13-slim

WORKDIR /root/ 

RUN mkdir output
RUN python -m pip install --upgrade pipenv wheel

COPY Pipfile .
COPY Pipfile.lock .

RUN pipenv install

COPY src ./src 

COPY data ./data

RUN pipenv run python src/create_model.py

EXPOSE 8080

ENTRYPOINT ["pipenv"]

CMD ["run", "serve"]


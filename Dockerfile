FROM python:3.9.13-slim

WORKDIR /root/ 

RUN mkdir output
RUN python -m pip install --upgrade pipenv wheel

COPY Pipfile .
COPY Pipfile.lock .

RUN pipenv install

COPY src ./src 

COPY data ./data

COPY config.yaml ./config.yaml

RUN pipenv run download_data

RUN pipenv run create

EXPOSE 8080

ENTRYPOINT ["pipenv"]

CMD ["run", "serve"]


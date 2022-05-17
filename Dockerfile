FROM python:3.10.4

WORKDIR /root/ 

COPY requirements.txt . 

RUN mkdir output && python -m pip install --upgrade pip && pip install -r requirements.txt 

COPY src ./src 

COPY data ./data

RUN python src/create_model.py

EXPOSE 8080

ENTRYPOINT ["python"]

CMD ["src/serve_model.py"]


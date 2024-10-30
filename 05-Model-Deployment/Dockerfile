FROM python:3.12-slim

RUN pip install pipenv

COPY ["Pipfile","Pipfile.lock","./"]

RUN pipenv install --system --deploy

COPY ["model_C=1.0.bin","predict.py","./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn","--bind=0.0.0.0:9696","predict:app"]
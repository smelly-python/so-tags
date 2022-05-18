"""
Module responsible for serving the model via HTTP requests.
"""
from flask import Flask, jsonify, request
from flasgger import Swagger

from joblib import load

app = Flask(__name__)
swagger = Swagger(app)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict what tags a SO post would have.
    ---
    consumes:
      - application/json
    parameters:
        - name: input_data
          in: body
          description: message to be classified.
          required: True
          schema:
            type: object
            required: SO post
            properties:
                so_title:
                    type: string
                    example: This is an example of an SMS.
    responses:
      200:
        description: "The result of the classification: List of strings with tags."
    """

    print(request)
    print(request.json)
    input_data = request.get_json()
    so_title = input_data.get('so_title')
    clf = load('output/model.joblib')
    clf.predict(so_title)

    res = {
        "result": clf.predict(so_title),
        "so_title": so_title
    }

    print(res)
    return jsonify(res)


@app.route("/dumbpredict", methods=['POST'])
def dumb_predict():
    """
    Return standard response, similar to the /predict endpoint.
    """
    res = {
        "result": "some tag",
        "so_title": "my cool java application!"
    }
    return jsonify(res)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

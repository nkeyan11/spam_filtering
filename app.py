# importing the necessary dependencies
from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle

app = Flask(__name__)  # initializing a flask app

@app.route('/', methods=['GET'])  # route to display the home page
@cross_origin()
def homePage():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])  # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            my_review = request.form['message']
            my_review = my_review.lower()

            transf_filename = 'count_vect.pickle'
            transformer = pickle.load(open(transf_filename, 'rb'))

            model_filename = 'spam_detect.pickle'
            loaded_model = pickle.load(open(model_filename, 'rb'))

            test = transformer.transform([my_review])
            test = test.toarray()
            result = loaded_model.predict(test)[0]

            print('prediction is', result)
            # showing the prediction results in a UI
            if result == 0:
                return render_template('results.html', result= 'Not Spam')
            else:
                return render_template('results.html', result= 'Spam')

        except Exception as e:
            print('The Exception message is: ', e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True)  # running the app











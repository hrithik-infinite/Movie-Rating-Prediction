from flask import Flask, render_template, url_for, request, jsonify
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import numpy as np

app = Flask(__name__)

# Load the TF-IDF vocabulary specific to the category
with open(r"actors_vect.pkl", "rb") as f:
    act = pickle.load(f)

with open(r"director_vect.pkl", "rb") as f:
    dire = pickle.load(f)

with open(r"genres_vect.pkl", "rb") as f:
    gen = pickle.load(f)

with open(r"overview_vect.pkl", "rb") as f:
    over = pickle.load(f)

# Load the pickled RDF models
with open(r"ml_model.pkl", "rb") as f:
    model = pickle.load(f)

#TF-IDF Vectorizers
# tf_idf_overview=TfidfVectorizer(max_features=10000, stop_words='english')
# tf_idf_director=TfidfVectorizer(stop_words='english')
# tf_idf_actors=TfidfVectorizer(max_features=10000, stop_words='english')
# tf_idf4_genres=TfidfVectorizer(stop_words='english')

# Render the HTML file for the home page
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    # Take a string input from user
    user_input1 = request.form['overview']
    data_over = [user_input1]
    d1 = over.transform(data_over).toarray()

    user_input2 = request.form['directors']
    data_dire = [user_input2]
    d2 = dire.transform(data_dire).toarray()

    user_input3 = request.form['actors']
    data_act = [user_input3]
    d3 = act.transform(data_act).toarray()

    user_input4 = request.form['genre']
    data_gen = [user_input4]
    d4 = gen.transform(data_gen).toarray()

    result_arr = np.concatenate((d1, d2, d3, d4), axis=1)

    # vect = act.transform(result_arr)
    pred_final = model.predict(result_arr)

    print(pred_final)

    return render_template('index.html', 
                            pred_final = 'Expected Rating is: {}'.format(pred_final)                     
                            )
     
# Server reloads itself if code changes so no need to keep restarting:
if __name__=="__main__":
    app.run(debug=True)


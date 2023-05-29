from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Import model
model = pickle.load(open('model.pkl', 'rb'))

# create flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Fosfor'])
    K = int(request.form['Kalium'])
    temp = float(request.form['Temperatur'])
    hum = float(request.form['Kelembaban'])
    ph = float(request.form['pH'])
    rain = float(request.form['Curah_hujan'])
    data = [N, P, K, temp, hum, ph, rain]
    single_prediksi = np.array(data).reshape(1, -1)

    prediksi = model.predict(single_prediksi)
    tanaman_list = ["Padi", "Jagung", "Serat", "Kapas", "Kelapa", "Pepaya", "Jeruk",
                "Apel", "Semangka", "Melon", "Anggur", "Mangga", "Pisang",
                "Delima", "Kacang Lentil", "Kacang Hitam", "Kacang Hijau", "Kacang Ngokilo",
                "Kacang Kuda", "Kacang Merah", "Kacang Arab", "Kopi"]

    if prediksi[0] < len(tanaman_list):
        tanaman_rekomendasi = tanaman_list[prediksi[0]]
        result = "{} adalah tanaman yang sangat cocok ditanam di daerah dengan kondisi tersebut.".format(
            tanaman_rekomendasi)
    else:
        result = "Maaf, kami tidak dapat menentukan tanaman terbaik untuk dibudidayakan dengan data yang tersedia."

    nitrogen = request.form['Nitrogen']
    fosfor = request.form['Fosfor']
    kalium = request.form['Kalium']
    temperatur = request.form['Temperatur']
    kelembaban = request.form['Kelembaban']
    pH = request.form['pH']
    curah_hujan = request.form['Curah_hujan']

    return render_template('output.html', prediksi=result, nitrogen=nitrogen, fosfor=fosfor, kalium=kalium,
                           temperatur=temperatur, kelembaban=kelembaban, pH=pH, curah_hujan=curah_hujan)

# python main
if __name__ == "__main__":
    app.run(debug=True)

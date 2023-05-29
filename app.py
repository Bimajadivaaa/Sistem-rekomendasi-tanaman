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
    tanaman_dict = {1: "Padi", 2: "Jagung", 3: "Serat", 4: "Kapas", 5: "Kelapa", 6: "Pepaya", 7: "Jeruk",
                    8: "Apel", 9: "Semangka", 10: "Melon", 11: "Anggur", 12: "Mangga", 13: "Pisang",
                    14: "Delima", 15: "Kacang Lentil", 16: "Kacang Hitam", 17: "Kacang Hijau", 18: "Kacang Ngokilo",
                    19: "Kacang Kuda", 20: "Kacang Merah", 21: "Kacang Arab", 22: "Kopi"}

    if prediksi[0] in tanaman_dict:
        tanaman_rekomendasi = tanaman_dict[prediksi[0]]
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

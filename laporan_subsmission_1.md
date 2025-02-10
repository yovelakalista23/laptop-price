# Laporan Proyek Machine Learning - Yovela Kalista Avansa

## Domain Proyek
Di era digital saat ini, laptop menjadi kebutuhan utama bagi berbagai kalangan, mulai dari pelajar, pekerja profesional, hingga gamer. Dengan banyaknya pilihan laptop di pasaran, calon pembeli sering kali kesulitan dalam menentukan pilihan yang sesuai dengan kebutuhan dan anggaran mereka.

Harga laptop sangat bervariasi, dipengaruhi oleh berbagai faktor seperti merek (Company), tipe laptop (TypeName), ukuran layar (Inches), resolusi layar (ScreenResolution), jenis prosesor (Cpu), kapasitas RAM (Ram), jenis penyimpanan (Memory), kartu grafis (Gpu), sistem operasi (OpSys), dan berat perangkat (Weight).

Oleh karena itu, proyek ini bertujuan untuk membangun model machine learning yang dapat memprediksi harga laptop berdasarkan spesifikasinya.

Mengapa Masalah Ini Perlu Diselesaikan?
1. Membantu calon pembeli dalam memperkirakan harga laptop berdasarkan spesifikasi yang diinginkan.
2. Mendukung strategi penetapan harga bagi penjual agar tetap kompetitif di pasar.
3. Mengoptimalkan keputusan pembelian dengan memberikan informasi yang lebih akurat.
4. Mengurangi kesalahan estimasi harga, terutama bagi mereka yang belum memahami faktor-faktor yang memengaruhi harga laptop.

Referensi : [Factors Affecting the Consumers Buying Behavior with Reference to Laptops](https://www.indianjournals.com/ijor.aspx?target=ijor:ijmmr&volume=2&issue=10&article=008&type=pdf)


## Bussiness Understanding
### Problem Statement

Dalam industri elektronik, harga laptop sangat bervariasi tergantung pada berbagai faktor seperti spesifikasi, merek, dan fitur tambahan. Oleh karena itu, membangun model prediksi harga laptop yang akurat dapat membantu konsumen dan penjual dalam mengambil keputusan. Berdasarkan permasalahan tersebut, beberapa pernyataan masalah yang diangkat adalah:
1. Bagaimana faktor-faktor seperti Merk dan Tipe Laptop memengaruhi harga laptop?
2. Bagaimana berat dan tipe laptop bervariasi berdasarkan mereknya?
3. Seberapa akurat model machine learning dalam memprediksi harga laptop berdasarkan fitur yang tersedia?
4. Bagaimana performa model XGBoost Regressor dan Random Forest Regressor dalam melakukan prediksi harga laptop?
5. Apakah tuning hyperparameter dapat meningkatkan akurasi model dibandingkan dengan model baseline?

### Goals

Berdasarkan pernyataan masalah di atas, tujuan dari proyek ini adalah:

1. Menganalisis faktor-faktor utama seperti Merk dan Tipe Laptop yang berkontribusi terhadap harga laptop.
2. Melakukan analisis untuk memahami bagaimana berat dan tipe laptop bervariasi berdasarkan mereknya.
3. Membangun model prediksi harga laptop menggunakan algoritma XGBoost Regressor dan Random Forest Regressor untuk menentukan laptop mana yang memiliki harga sesuai dengan spesifikasinya berdasarkan dataset yang tersedia.
4. Membandingkan performa model sebelum dan sesudah tuning hyperparameter dengan menggunakan metrik evaluasi seperti Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), dan R² Score.
5. Meningkatkan performa model menggunakan RandomizedSearchCV untuk hyperparameter tuning, lalu membandingkan hasilnya dengan baseline model.

### Solution Statements

Untuk mencapai tujuan yang telah ditentukan, solusi yang diterapkan dalam proyek ini adalah:

1. Membangun dua model machine learning yaitu XGBoost Regressor dan Random Forest Regressor untuk memprediksi harga laptop berdasarkan fitur yang tersedia.
2. Melakukan Data Preparation seperti encoding features, memisahkan fitur X dan Y, split dataset, dan standarisasi.
3. Membandingkan performa model baseline tanpa tuning hyperparameter untuk melihat bagaimana model bekerja dengan pengaturan default.
4. Melakukan hyperparameter tuning menggunakan RandomizedSearchCV untuk meningkatkan akurasi model dengan mencari kombinasi parameter terbaik.
5. Mengevaluasi hasil model menggunakan MSE, RMSE, MAE, dan R² Score, serta membandingkan hasil sebelum dan sesudah tuning.

## Data Understanding
**Deskripsi Dataset**

Dataset yang digunakan dalam proyek ini berisi informasi mengenai spesifikasi laptop dan harga jualnya. Dataset ini dikumpulkan dari berbagai merek laptop dengan spesifikasi yang berbeda-beda, seperti ukuran layar, prosesor, RAM, kapasitas penyimpanan, kartu grafis, sistem operasi, dan berat laptop.

Dataset ini dapat digunakan untuk menganalisis faktor-faktor yang mempengaruhi harga laptop serta membangun model prediksi harga yang lebih akurat.
[Laptop Price Prediction](https://www.kaggle.com/datasets/eslamelsolya/laptop-price-prediction/data)

### Jumlah data:
shape awalnya yaitu (1303, 11), jadi 1303 baris dan 11 kolom

Kemudian setelah melakukan assessing dataset ditemukan duplikat, dan dataset bersihnya menjadi 

(1274, 11), 1274 baris dan 11 kolom

### Variabel-variabel pada Laptop Price Prediction dataset adalah sebagai berikut:
Berikut adalah deskripsi dari setiap fitur yang terdapat dalam dataset:

1. Company → Merek atau produsen laptop (contoh: Apple, Dell, Asus).
2. TypeName → Jenis laptop, seperti Ultrabook, Gaming, Netbook, Notebook, Workstation.
3. Inches → Ukuran layar laptop dalam satuan inci.
4. ScreenResolution → Resolusi layar laptop (contoh: 1920x1080, Retina Display).
5. Cpu → Jenis dan kecepatan prosesor yang digunakan pada laptop.
6. Ram → Kapasitas RAM pada laptop dalam satuan GB.
7. Memory → Kapasitas penyimpanan internal laptop.
8. Gpu → Jenis kartu grafis yang digunakan pada laptop, bisa berupa Integrated atau Dedicated GPU.
9. OpSys → Sistem operasi yang digunakan pada laptop (contoh: Windows, macOS, Linux, atau tanpa OS).
10. Weight → Berat laptop dalam satuan kg.
11. Price → Harga laptop (target variabel yang akan diprediksi).

### Exploratory Data Analysis (EDA)
Untuk memahami dataset lebih dalam, dilakukan beberapa tahapan Exploratory Data Analysis (EDA), seperti:

1. Melihat korelasi fitur antara Company dan Tipe Laptop terhadap Harga laptop.
   **Penjelasan Bar Chart dari komponen Company & Price**

- **Razer** memiliki harga rata-rata tertinggi dengan variasi harga yang besar, menunjukkan adanya produk dalam berbagai kategori harga. 
- **Apple, LG, MSI, dan Google** cenderung memiliki harga rata-rata tinggi dibandingkan merek lain.  
- **HP, Lenro, dan MediaCom** memiliki harga rata-rata terendah, menunjukkan kemungkinan fokus pada pasar low-end.  
- Rentang erroovo, Dell, dan Asus** berada dalam kategori harga menengah.  
- **Chuwi, Ver bars yang besar pada beberapa merek (seperti Razer dan Microsoft) mengindikasikan variasi harga yang signifikan dalam produk mereka.

**Analisis Harga Tipe Laptop Berdasarkan tipe laptop**

- **Perbedaan harga antar tipe laptop cukup signifikan**, terutama untuk tipe **Gaming** dan **Workstation**, yang umumnya memiliki harga lebih tinggi dibanding tipe lainnya.  
- **Razer memiliki harga Gaming laptop tertinggi**, sementara **Workstation dari beberapa brand seperti HP dan Lenovo juga cukup mahal**.  
- **Notebook dan Ultrabook tersebar merata di berbagai perusahaan dengan harga yang lebih terjangkau** dibandingkan Gaming atau Workstation.  
- **Beberapa perusahaan seperti Apple, Microsoft, dan Huawei lebih fokus pada segmen Ultrabook**, sementara perusahaan seperti MSI dan Asus lebih banyak memiliki laptop Gaming.  
- **Vero dan MediaCom memiliki harga laptop yang relatif lebih rendah**, kemungkinan menyasar pasar entry-level. 

2. Melihat korelasi fitur antara berat dengan harga laptop.
   
   **Analisis Berat dari Tipe Laptop Berdasarkan Perusahaan**

- Laptop Gaming umumnya lebih berat, terutama pada brand seperti Asus, Dell, dan Lenovo.
- Ultrabook cenderung lebih ringan dibandingkan tipe lain.
- Beberapa brand memiliki variasi laptop yang luas, seperti HP, Dell, dan Lenovo yang menawarkan berbagai tipe laptop dengan berat yang berbeda.
- Brand seperti Apple dan Microsoft memiliki laptop dengan bobot yang lebih ringan, kemungkinan karena mereka lebih fokus pada ultrabook atau perangkat tipis.
  
3. Melihat korelasi antara fitur dan harga laptop menggunakan heatmap correlation untuk mengidentifikasi fitur mana yang memiliki hubungan kuat dengan harga.
Korelasi positif yang bisa dilihat dengan jelas yaitu antara 'Weight' dan 'Inches' kemudian antara 'Price' dan 'RAM'


   Dalam dataset awal masih terdapat fitur kategorikal, model machine learning tidak dapat langsung memahami data dalam format teks, sehingga diperlukan proses encoding untuk mengubahnya menjadi nilai numerik.

## Data Preparation
Beberapa hal yang dilakukan dalam data preparation, yaitu:

1. Handling Data Duplikat
   Saat melakukan assessing data, terdapat data duplikat yang harus diselesaikan.
2. Encoding Features
   Teknik encoding yang digunakan yaitu LabelEncoding.
3. Memisahkan Fitur (X) dan Target (y) dengan Log Transformation
   Menggunakan log transformation (np.log) untuk mengubah distribusi harga menjadi lebih normal.
4. Membagi Dataset dengan train_test_split
   Agar model bisa diuji pada data baru yang belum pernah dilihat sebelumnya. random_state=42 digunakan agar hasil split selalu konsisten.
5. standarisasi

## Modeling
Dalam proyek ini menggunakan dua algoritma machine learning untuk memprediksi harga laptop, yaitu:

1. XGBoost Regressor
   
   XGBoost Regressor menggunakan teknik gradient boosting untuk meningkatkan akurasi dengan meminimalkan kesalahan secara bertahap. Parameter utama yang digunakan adalah n_estimators=100, max_depth=5, dan learning_rate=0.1.
   
2. Random Forest Regressor

   Random Forest Regressor adalah metode bagging yang menggabungkan banyak pohon keputusan untuk meningkatkan kestabilan prediksi. Parameter yang digunakan adalah n_estimators=100 dan max_depth=5.

**Tuning**

Tuning yang digunakan untuk meningkatkan performa model yaitu RandomizedSearchCV dengan menentukan parameter yang akan diuji.

**XGBoost Regressor**

- max_depth → Menentukan kedalaman pohon keputusan dalam XGBoost.
- learning_rate → Mengontrol seberapa besar model belajar dari error sebelumnya.
- n_estimators → Jumlah pohon yang digunakan dalam boosting.
- min_child_weight → Mengontrol ukuran minimum node untuk mengurangi overfitting.
- subsample → Persentase data yang digunakan dalam setiap iterasi boosting.
- colsample_bytree → Persentase fitur yang dipilih dalam setiap pohon.

**Random Forest Regressor**

- max_depth → Menentukan kedalaman maksimum pohon dalam hutan.
- n_estimators → Jumlah pohon dalam model Random Forest.
- min_samples_split → Jumlah minimum sampel yang dibutuhkan untuk membagi node.
- min_samples_leaf → Jumlah minimum sampel pada setiap leaf node.
- bootstrap → Menentukan apakah akan menggunakan bootstrap sampling atau tidak.

Agar tuning lebih cepat dibandingkan Grid Search (yang mencoba semua kombinasi), digunakan RandomizedSearchCV, yang mencoba 20 kombinasi acak dari parameter yang telah ditentukan.

Setelah menjalankan RandomizedSearchCV, didapatkan parameter terbaik untuk setiap model.

XGB_tuned Best Parameters: {'subsample': 0.8, 'n_estimators': 300, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.1, 'colsample_bytree': 0.8}

RF_tuned Best Parameters: {'n_estimators': 100, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': 9, 'bootstrap': True}

Setelah melakukan training dengan parameter default, selanjutnya melakukan improvement model dengan hyperparameter tuning menggunakan RandomizedSearchCV.
- XGBoost lebih baik dibandingkan Random Forest, karena memiliki MSE yang lebih kecil dan R² yang lebih tinggi.
- Setelah tuning, kedua model mengalami peningkatan performa secara signifikan.
- XGBoost lebih unggul dalam menangani data ini karena setelah tuning, R² mencapai 0.9103, menunjukkan model ini bisa menjelaskan ~91% variabilitas data.

## Evaluation

Dalam proyek ini menggunakan model regresi untuk memprediksi harga laptop berdasarkan fitur-fitur seperti spesifikasi perangkat keras, sistem operasi, dan faktor lainnya. Oleh karena itu, metrik evaluasi yang digunakan harus sesuai dengan model regresi, yang berbeda dengan metrik untuk klasifikasi seperti akurasi, precision, atau recall.

|            Model          | MSE |    RMSE        |                    MAE                    |                                                       R2 Score                                                      |
| :-------------------------: | :--------: | :----------------: | :----------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
|        XGBoost   | 0.0416 | 0.2039    |   0.1520   |           0.8909           |
|      Random Forest    | 0.0689 | 0.2625    |     0.2042      |   0.8193  |
|   💡 **XGBoost Tuned**         | **0.0342** | **0.1849**    |     **0.1387**      |            **0.9103** 💡            |
|      Random Forest Tuned        | 0.0504 | 0.2244     |    0.1709  |    0.8679    |
         

Model yang telah dievaluasi berhasil menjawab setiap problem statement dan mencapai goals yang ditetapkan.

1. Analisis Faktor yang Mempengaruhi Harga Laptop

Hasil eksplorasi menunjukkan bahwa Merk dan Tipe Laptop memiliki pengaruh terhadap harga. Hal ini sesuai dengan tujuan untuk menganalisis faktor utama yang memengaruhi harga laptop.

2. Analisis Berat dan Tipe Laptop Berdasarkan Merek

Distribusi berat dan tipe laptop telah dianalisis berdasarkan merek, menunjukkan bahwa ultrabook cenderung lebih ringan dibandingkan tipe lain. Analisis ini membantu memahami bagaimana berat dan tipe laptop bervariasi berdasarkan mereknya.

3. Membangun Model Prediksi Harga Laptop

Model XGBoost Regressor dan Random Forest Regressor berhasil digunakan untuk memprediksi harga laptop.
Model mampu memperkirakan harga laptop berdasarkan spesifikasi dengan akurasi tinggi, sesuai dengan tujuan proyek.

4. Evaluasi Performa Model Sebelum dan Sesudah Hyperparameter Tuning

Metrik evaluasi menunjukkan bahwa XGBoost lebih unggul dibandingkan Random Forest, dengan MSE = 0.0342 dan R² = 0.9103 setelah tuning.
Perbandingan performa sebelum dan sesudah tuning menunjukkan peningkatan akurasi.

5. Peningkatan Model dengan Hyperparameter Tuning

Penggunaan RandomizedSearchCV meningkatkan performa model, membuktikan bahwa tuning parameter memberikan hasil lebih baik dibandingkan model baseline.

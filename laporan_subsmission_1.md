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
1. Bagaimana faktor-faktor spesifikasi seperti Merk dan Tipe Laptop memengaruhi harga laptop?
2. Bagaimana faktor berat dari Tipe Laptop di tiap Merknya?
3. Seberapa akurat model machine learning dalam memprediksi harga laptop berdasarkan fitur yang tersedia?
4. Bagaimana performa model XGBoost Regressor dan Random Forest Regressor dalam melakukan prediksi harga laptop?
5. Apakah tuning hyperparameter dapat meningkatkan akurasi model dibandingkan dengan model baseline?

### Goals

Berdasarkan pernyataan masalah di atas, tujuan dari proyek ini adalah:

1. Menganalisis faktor-faktor utama yang berkontribusi terhadap harga laptop berdasarkan dataset yang tersedia.
2. Membangun model prediksi harga laptop menggunakan algoritma XGBoost Regressor dan Random Forest Regressor untuk menentukan laptop mana yang memiliki harga sesuai dengan spesifikasinya.
3. Membandingkan performa model sebelum dan sesudah tuning hyperparameter dengan menggunakan metrik evaluasi seperti Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), dan R² Score.
4. Menghasilkan model dengan performa terbaik yang dapat digunakan untuk estimasi harga laptop secara otomatis.

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
2. Melihat korelasi fitur antara berat dengan harga laptop.
3. Melihat korelasi antara fitur dan harga laptop menggunakan heatmap correlation untuk mengidentifikasi fitur mana yang memiliki hubungan kuat dengan harga.
4. Melakukan data featuring, seperti encoding untuk fitur kategorikal, label encoding, serta menggunakan log transformation (np.log) untuk mengubah distribusi harga menjadi lebih normal.

## Data Preparation
Beberapa hal yang dilakukan dalam data preparation, yaitu:

1. Encoding Features
   Teknik encoding yang digunakan yaitu LabelEncoding.
2. Memisahkan Fitur (X) dan Target (y) dengan Log Transformation
   Menggunakan log transformation (np.log) untuk mengubah distribusi harga menjadi lebih normal.
3. Membagi Dataset dengan train_test_split
   Agar model bisa diuji pada data baru yang belum pernah dilihat sebelumnya. random_state=42 digunakan agar hasil split selalu konsisten.
4. standarisasi

## Modeling
Dalam proyek ini, saya menggunakan dua algoritma machine learning untuk memprediksi harga laptop, yaitu:

1. XGBoost Regressor
2. Random Forest Regressor

Setelah melakukan training dengan parameter default, saya melakukan improvement model dengan hyperparameter tuning menggunakan RandomizedSearchCV.
- XGBoost lebih baik dibandingkan Random Forest, karena memiliki MSE yang lebih kecil dan R² yang lebih tinggi.
- Setelah tuning, kedua model mengalami peningkatan performa secara signifikan.
- XGBoost lebih unggul dalam menangani data ini karena setelah tuning, R² mencapai 0.9103, menunjukkan model ini bisa menjelaskan ~91% variabilitas data.

## Evaluation

Dalam proyek ini, saya menggunakan model regresi untuk memprediksi harga laptop berdasarkan fitur-fitur seperti spesifikasi perangkat keras, sistem operasi, dan faktor lainnya. Oleh karena itu, metrik evaluasi yang digunakan harus sesuai dengan model regresi, yang berbeda dengan metrik untuk klasifikasi seperti akurasi, precision, atau recall.

|            Model          | MSE |    RMSE        |                    MAE                    |                                                       R2 Score                                                      |
| :-------------------------: | :--------: | :----------------: | :----------------------------------------: | :-----------------------------------------------------------------------------------------------------------------: |
|        XGBoost   | 0.0416 | 0.2039    |   0.1520   |           0.8909           |
|      Random Forest    | 0.0689 | 0.2625    |     0.2042      |   0.8193  |
|   💡 **XGBoost Tuned**         | **0.0342** | **0.1849**    |     **0.1387**      |            **0.9103** 💡            |
|      Random Forest Tuned        | 0.0504 | 0.2244     |    0.1709  |    0.8679    |
         


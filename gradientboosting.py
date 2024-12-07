import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import datetime

# Veri Yükleme ve Filtreleme
data = pd.read_csv('used_cars.csv')
data = data[(data['price'] > 500) & (data['price'] < 80000)]  # Anormal fiyatları filtreleme
data = data[['price', 'year', 'manufacturer', 'fuel', 'odometer', 'transmission', 'drive', 'paint_color', 'type']]  # Gereksiz sütunları çıkarın
data = data.sample(frac=0.3, random_state=42)  # Verinin %30'unu kullan

# Eksik Verilerin Doldurulması (KNNImputer)
numerical_columns = ['year', 'odometer']
imputer = KNNImputer(n_neighbors=5)
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

# Aykırı Değerlerin Tespiti ve Kaldırılması (Z-Score)
data['odometer_zscore'] = zscore(data['odometer'])
data = data[data['odometer_zscore'].abs() < 3]  # Z-Score değeri 3'ten büyük olanlar çıkarıldı
data.drop(columns=['odometer_zscore'], inplace=True)

# Kategorik Değişkenlerin Dönüştürülmesi
categorical_columns = ['manufacturer', 'fuel', 'transmission', 'drive', 'paint_color', 'type']
label_encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])

# Yeni Özellikler Eklenmesi
current_year = datetime.datetime.now().year
data['car_age'] = current_year - data['year']  # Araç yaşı
data['price_per_km'] = data['price'] / (data['odometer'] + 1)  # Fiyat / Kilometre (0 bölme hatasını önlemek için 1 eklendi)

# Özellik Etkileşimleri
data['odometer_fuel'] = data['odometer'] * data['fuel']

# Anlamsız Özelliklerin Çıkarılması
X = data.drop('price', axis=1)
y = data['price']
importance = mutual_info_regression(X, y)
importance_dict = dict(zip(X.columns, importance))
print("Feature Importances (Mutual Info):", importance_dict)

# Yüksek Korelasyonlu Özelliklerin Seçimi
selected_features = [feature for feature, score in importance_dict.items() if score > 0.01]
X = X[selected_features]

# Veriyi Eğitim ve Test Olarak Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Gradient Boosting Modeli Eğitimi
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X_train, y_train)

# Tahmin ve Performans Değerlendirme
y_pred = gbr_model.predict(X_test)

def evaluate_model(y_test, y_pred, model_name="Model"):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} Performance:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print("-" * 40)
    return mae, mse, r2

evaluate_model(y_test, y_pred, "Gradient Boosting Regressor")

# Özellik Önemini Görselleştirme
def visualize_feature_importance(model, feature_names):
    importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances, color='skyblue')
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title("Feature Importances (Gradient Boosting)")
    plt.show()

visualize_feature_importance(gbr_model, X.columns)

# Tahmin ve Gerçek Değerleri Görselleştirme
def visualize_predictions(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Car Prices")
    plt.show()

visualize_predictions(y_test, y_pred)

# Hata Analizi (Residual Plot)
def visualize_residuals(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, color='orange', edgecolor='black')
    plt.xlabel("Residuals (Errors)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors (Residuals)")
    plt.show()
    

visualize_residuals(y_test, y_pred)

results = gbr_model.staged_predict(X_test)
test_errors = [mean_squared_error(y_test, y_pred) for y_pred in results]

plt.plot(test_errors, label="Test Error")
plt.xlabel("Number of Trees")
plt.ylabel("Mean Squared Error")
plt.title("Gradient Boosting Performance")
plt.legend()
plt.show()

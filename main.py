import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D  # 3B grafikler için gerekli


# =============================================================================
# HBNAnalysisSystem Sınıfı
# =============================================================================
class HBNAnalysisSystem:
    def __init__(self, random_state=42):
        """
        Sınıf başlatılır; veri, model, ölçekleyiciler ve diğer parametreler tanımlanır.
        """
        self.random_state = random_state
        self.data = None               # Ham veri (DataFrame)
        self.model = None              # Keras modeli
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.X_scaled = None           # Ölçeklendirilmiş giriş verisi
        self.y_scaled = None           # Ölçeklendirilmiş hedef veri
        self.history = None            # Eğitim geçmişi
        self.performance_metrics = {}  # Çapraz doğrulama metrikleri
        self.final_metrics = {}        # Nihai model metrikleri

    # ------------------------------------------------
    # 1) Veri Yükleme ve Outlier Temizleme
    # ------------------------------------------------
    def load_data(self, remove_outliers=False, outlier_factor=1.5):
        """
        'data.csv' dosyasını okur, beklenen kolonları kontrol eder, eksik verileri doldurur 
        ve isteğe bağlı olarak aykırı değerleri temizler.
        """
        self.data = pd.read_csv('data.csv', delimiter=';')
        self.data.columns = self.data.columns.str.strip()
        expected_columns = [
            '% hBN', 'Cure Temperature (°C)', '% Functionalization',
            'Elastic Modulus (GPa)', 'Tensile Strength (MPa)',
            'Glass Transition Temp (°C)', 'High Temp Modulus (GPa)',
            'High Temp Strength (MPa)'
        ]
        missing_columns = [col for col in expected_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"CSV dosyası şu kolonları içermiyor: {', '.join(missing_columns)}")
        self.data.fillna(self.data.mean(), inplace=True)
        if remove_outliers:
            self._remove_outliers_iqr(factor=outlier_factor)

    def _remove_outliers_iqr(self, factor=1.5):
        """
        Aykırı değerleri IQR yöntemiyle temizler.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            self.data = self.data[(self.data[col] >= lower) & (self.data[col] <= upper)]

    # ------------------------------------------------
    # 2) Veri Hazırlama ve Sentetik Veri Üretimi (Augmentasyon)
    # ------------------------------------------------
    def prepare_data(self, augment_data=True, noise_level=0.02, num_samples=500):
        """
        Veriyi hazırlar; % hBN'nin karesi eklenir, özellik ve hedef sütunları ayrılır,
        '% Functionalization' sütunu 0 veya 1 olarak dönüştürülür,
        istenirse sentetik veri eklenir ve veriler StandardScaler ile normalize edilir.
        """
        # % Functionalization sütununu binary'ye çevir (değer 0.5'in üzerindeyse 1, aksi halde 0)
        self.data['% Functionalization'] = (self.data['% Functionalization'] > 0.5).astype(int)
        self.data['hBN_squared'] = self.data['% hBN'] ** 2

        feature_columns = [
            '% hBN',
            'Cure Temperature (°C)',
            '% Functionalization',
            'hBN_squared'
        ]
        target_columns = [
            'Elastic Modulus (GPa)',
            'Tensile Strength (MPa)',
            'Glass Transition Temp (°C)',
            'High Temp Strength (MPa)',
            'High Temp Modulus (GPa)'
        ]
        self.X = self.data[feature_columns].values
        self.y = self.data[target_columns].values

        print("\n=== Hedef Değişken İstatistikleri ===")
        print(pd.DataFrame(self.y, columns=target_columns).describe())

        if augment_data:
            X_syn, y_syn = self._generate_synthetic_data(noise_level=noise_level, num_samples=num_samples)
            self.X = np.vstack((self.X, X_syn))
            self.y = np.vstack((self.y, y_syn))
        # Normalize et
        self.X_scaled = self.X_scaler.fit_transform(self.X)
        self.y_scaled = self.y_scaler.fit_transform(self.y)
        return self.X, self.y

    def _generate_synthetic_data(self, noise_level=0.02, num_samples=100):
        """
        Mevcut veri örneklerine küçük gürültü ekleyerek sentetik veri üretir.
        """
        synthetic_X = []
        synthetic_y = []
        hbn_values = self.data['% hBN'].unique()
        for _ in range(num_samples):
            if np.random.random() < 0.7:
                hbn = np.random.uniform(min(hbn_values), max(hbn_values))
            else:
                hbn = np.random.uniform(min(hbn_values) - 0.1, max(hbn_values) + 0.1)
            func = np.random.uniform(0, 5)
            cure_temp = 80.0
            hbn_sq = hbn ** 2
            nearest = self.data[(abs(self.data['% hBN'] - hbn) < 0.3)]
            if len(nearest) > 0:
                weights = 1 / (abs(nearest['% hBN'] - hbn) + 1e-6)
                weights /= weights.sum()
                mod_ = (nearest['Elastic Modulus (GPa)'] * weights).sum()
                str_ = (nearest['Tensile Strength (MPa)'] * weights).sum()
                tg_  = (nearest['Glass Transition Temp (°C)'] * weights).sum()
                hts_ = (nearest['High Temp Strength (MPa)'] * weights).sum()
                htm_ = (nearest['High Temp Modulus (GPa)'] * weights).sum()
                mod_ += np.random.normal(0, noise_level * mod_)
                str_ += np.random.normal(0, noise_level * str_)
                tg_  += np.random.normal(0, noise_level * tg_)
                hts_ += np.random.normal(0, noise_level * hts_)
                htm_ += np.random.normal(0, noise_level * htm_)
                x = np.array([hbn, cure_temp, func, hbn_sq])
                y = np.array([mod_, str_, tg_, hts_, htm_])
                synthetic_X.append(x)
                synthetic_y.append(y)
        return np.array(synthetic_X), np.array(synthetic_y)

    # ------------------------------------------------
    # 3) Korelasyon Analizi
    # ------------------------------------------------
    def analyze_correlation(self):
        """
        Veri içindeki sütunlar arası korelasyonu ısı haritası olarak gösterir.
        """
        if self.data is not None:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title("Korelasyon Matrisi")
            plt.tight_layout()
            plt.show()
        else:
            print("Veri yüklenmemiş. Korelasyon analizi yapılamaz.")

    # ------------------------------------------------
    # 4) Model Oluşturma (Adam Optimizasyonu)
    # ------------------------------------------------
    def create_model(self, input_dim=4, n_neurons=128, dropout_rate=0.2, l2_reg=1e-5,
                     learning_rate=0.0005, activation='selu', loss_fn='mse'):
        """
        Model mimarisini oluşturur ve Adam optimizatörü ile derler.
        """
        model = Sequential([
            Dense(n_neurons, input_dim=input_dim, kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Dense(n_neurons, activation=activation, kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Dropout(dropout_rate),
            Dense(n_neurons // 2, activation=activation, kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Dropout(dropout_rate * 0.7),
            Dense(n_neurons // 4, activation=activation, kernel_regularizer=l2(l2_reg)),
            BatchNormalization(),
            Dropout(dropout_rate * 0.7),
            Dense(5, activation='linear')
        ])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae', 'mse'])
        return model

    # ------------------------------------------------
    # 5) K-Katlı Çapraz Doğrulama
    # ------------------------------------------------
    def cross_validate_model(self, n_splits=5, n_repeats=1, n_neurons=128, dropout_rate=0.2,
                             l2_reg=1e-5, learning_rate=0.0005, activation='selu',
                             epochs=200, batch_size=16):
        """
        Veriyi K-Katlı veya tekrarlı çapraz doğrulamaya tabi tutarak model performansını değerlendirir.
        """
        if self.X_scaled is None or self.y_scaled is None:
            raise ValueError("Veriler ölçeklendirilip hazırlanmalıdır.")
        kfold = (RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.random_state)
                 if n_repeats > 1 else KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state))
        mse_scores, mae_scores, r2_scores = [], [], []
        input_dim = self.X_scaled.shape[1]
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
        ]
        for train_idx, val_idx in kfold.split(self.X_scaled):
            X_train, X_val = self.X_scaled[train_idx], self.X_scaled[val_idx]
            y_train, y_val = self.y_scaled[train_idx], self.y_scaled[val_idx]
            temp_model = self.create_model(input_dim=input_dim, n_neurons=n_neurons,
                                           dropout_rate=dropout_rate, l2_reg=l2_reg,
                                           learning_rate=learning_rate, activation=activation)
            temp_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                           epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)
            y_pred_val = temp_model.predict(X_val, verbose=0)
            y_pred_val_inv = self.y_scaler.inverse_transform(y_pred_val)
            y_val_inv = self.y_scaler.inverse_transform(y_val)
            mse_scores.append(mean_squared_error(y_val_inv, y_pred_val_inv))
            mae_scores.append(mean_absolute_error(y_val_inv, y_pred_val_inv))
            r2_list = [r2_score(y_val_inv[:, i], y_pred_val_inv[:, i]) for i in range(y_val_inv.shape[1])]
            r2_scores.append(np.mean(r2_list))
        self.performance_metrics = {
            'CV_MSE_Mean': np.mean(mse_scores),
            'CV_MSE_Std': np.std(mse_scores),
            'CV_MAE_Mean': np.mean(mae_scores),
            'CV_MAE_Std': np.std(mae_scores),
            'CV_R2_Mean': np.mean(r2_scores),
            'CV_R2_Std': np.std(r2_scores)
        }
        print("\n========== K-Katlı Çapraz Doğrulama METRİKLERİ ==========")
        print(f"Ortalama MSE: {self.performance_metrics['CV_MSE_Mean']:.4f} ± {self.performance_metrics['CV_MSE_Std']:.4f}")
        print(f"Ortalama MAE: {self.performance_metrics['CV_MAE_Mean']:.4f} ± {self.performance_metrics['CV_MAE_Std']:.4f}")
        print(f"Ortalama R2 : {self.performance_metrics['CV_R2_Mean']:.4f} ± {self.performance_metrics['CV_R2_Std']:.4f}")
        return self.performance_metrics

    # ------------------------------------------------
    # 6) Nihai Model Eğitimi
    # ------------------------------------------------
    def train_final_model(self, n_neurons=128, dropout_rate=0.2, l2_reg=1e-5,
                          learning_rate=0.0005, activation='selu', loss_fn='mse',
                          epochs=200, batch_size=16):
        """
        Tüm veriyi kullanarak nihai modeli eğitir ve performans metriklerini kaydeder.
        """
        if self.X_scaled is None or self.y_scaled is None:
            raise ValueError("Veriler ölçeklendirilip hazırlanmalıdır.")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-6)
        ]
        self.model = self.create_model(input_dim=self.X_scaled.shape[1], n_neurons=n_neurons,
                                       dropout_rate=dropout_rate, l2_reg=l2_reg,
                                       learning_rate=learning_rate, activation=activation,
                                       loss_fn=loss_fn)
        history = self.model.fit(self.X_scaled, self.y_scaled, validation_split=0.2,
                                 epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1)
        self.history = history
        y_pred_sample = self.model.predict(self.X_scaled[:5])
        print("\n--- Örnek Tahminler vs Gerçek Değerler ---")
        print("Tahminler (inverse scaled):")
        print(self.y_scaler.inverse_transform(y_pred_sample))
        print("Gerçek Değerler:")
        print(self.y_scaler.inverse_transform(self.y_scaled[:5]))
        self.final_metrics = {
            "Final_Train_Loss": history.history['loss'][-1],
            "Final_Val_Loss": history.history['val_loss'][-1],
            "Final_Train_MAE": history.history['mae'][-1],
            "Final_Val_MAE": history.history['val_mae'][-1],
            "Final_Train_MSE": history.history['mse'][-1],
            "Final_Val_MSE": history.history['val_mse'][-1]
        }
        print("\n========== Nihai Model Performansı ==========")
        for k, v in self.final_metrics.items():
            print(f"{k}: {v:.4f}")
        if abs(self.final_metrics["Final_Train_Loss"] - self.final_metrics["Final_Val_Loss"]) > 0.5:
            print("WARNING: Model overfitting yapıyor olabilir. Dropout veya L2 regularizasyon arttırılabilir.")
        return history

    # ------------------------------------------------
    # 7) Eğitim Grafikleri
    # ------------------------------------------------
    def plot_results(self):
        """
        Eğitim sürecindeki loss, MAE ve MSE değerlerinin epoch bazında dağılımını çizer.
        """
        if self.history is not None:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.plot(self.history.history['loss'], label='Train')
            plt.plot(self.history.history['val_loss'], label='Validation')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.subplot(1, 3, 2)
            plt.plot(self.history.history['mae'], label='Train')
            plt.plot(self.history.history['val_mae'], label='Validation')
            plt.title('MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            plt.subplot(1, 3, 3)
            plt.plot(self.history.history['mse'], label='Train')
            plt.plot(self.history.history['val_mse'], label='Validation')
            plt.title('MSE')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------
    # 8) Model Tahmin Grafikleri (PDF Raporu)
    # ------------------------------------------------
    def plot_output_vs_inputs(self, n_points=50):
        """
        Belirlenen % hBN aralığında modelin tahmin ettiği çıktı değerlerini çizer
        ve çapraz doğrulama ile nihai model metriklerini içeren PDF raporu oluşturur.
        """
        if not self.model:
            print("Eğitilmiş model bulunamadı. Lütfen önce model eğitin.")
            return
        pdf = PdfPages("model_report.pdf")
        cure_temp = 80.0
        func_degree = 2.0
        hbn_range = np.linspace(0, 1, n_points)
        predictions = []
        for h in hbn_range:
            x_in = np.array([h, cure_temp, func_degree, h**2]).reshape(1, -1)
            x_scaled = self.X_scaler.transform(x_in)
            y_scaled = self.model.predict(x_scaled)
            y_pred = self.y_scaler.inverse_transform(y_scaled)
            predictions.append(y_pred[0])
        preds = np.array(predictions)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(hbn_range, preds[:, 0], label='Elastic Modulus (GPa)')
        ax.plot(hbn_range, preds[:, 1], label='Tensile Strength (MPa)')
        ax.plot(hbn_range, preds[:, 2], label='Glass Transition Temp (°C)')
        ax.plot(hbn_range, preds[:, 3], label='High Temp Strength (MPa)')
        ax.plot(hbn_range, preds[:, 4], label='High Temp Modulus (GPa)')
        ax.set_xlabel('% hBN')
        ax.set_ylabel('Tahmin Değerleri')
        ax.set_title(f'Çıktıların % hBN ile Değişimi\n(CureT={cure_temp}, Func={func_degree})')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)
        fig2 = plt.figure(figsize=(6, 2))
        text_content = "K-Katlı Çapraz Doğrulama Metrikleri:\n"
        for k, v in self.performance_metrics.items():
            text_content += f"{k}: {v:.4f}\n"
        text_content += "\nNihai Model Metrikleri:\n"
        for k, v in self.final_metrics.items():
            text_content += f"{k}: {v:.4f}\n"
        plt.text(0.1, 0.5, text_content, fontsize=12)
        plt.axis('off')
        pdf.savefig(fig2)
        plt.close(fig2)
        pdf.close()
        print("model_report.pdf oluşturuldu ve rapor kaydedildi.")

    # ------------------------------------------------
    # 9) Random Search ile En İyi Girdinin Bulunması
    # ------------------------------------------------
    def find_optimal_inputs(self, search_size=1000):
        """
        Rastgele oluşturulan giriş örnekleri arasından, modelin çıktı toplamını maksimize eden
        girdiyi ve buna karşılık gelen çıktıyı bulur.
        """
        if not self.model:
            print("Model henüz eğitilmemiş. Optimal girdi aranamaz.")
            return None
        best_sum = -np.inf
        best_input = None
        best_output = None
        candidates = []
        for _ in range(search_size):
            hbn_val = np.random.uniform(0, 1)
            cure_val = np.random.uniform(50, 120)
            func_val = np.random.uniform(0, 5)
            hbn_sq = hbn_val ** 2
            x = np.array([hbn_val, cure_val, func_val, hbn_sq]).reshape(1, -1)
            candidates.append(x)
        candidates = np.vstack(candidates)
        X_scaled = self.X_scaler.transform(candidates)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        sums = y_pred.sum(axis=1)
        idx_best = np.argmax(sums)
        best_input = candidates[idx_best, :]
        best_output = y_pred[idx_best, :]
        print("\n=== Optimum Girdi Arama Sonucu (Random Search) ===")
        print(f"En yüksek çıktı toplamı: {sums[idx_best]:.2f}")
        print("Girdi değerleri:")
        print(f"  % hBN = {best_input[0]:.3f}")
        print(f"  Cure Temp (°C) = {best_input[1]:.3f}")
        print(f"  % Functionalization = {best_input[2]:.3f}")
        print("Tahmin edilen çıktılar:")
        print(f"  Elastic Modulus (GPa): {best_output[0]:.2f}")
        print(f"  Tensile Strength (MPa): {best_output[1]:.2f}")
        print(f"  Glass Transition Temp (°C): {best_output[2]:.2f}")
        print(f"  High Temp Strength (MPa): {best_output[3]:.2f}")
        print(f"  High Temp Modulus (GPa): {best_output[4]:.2f}")
        return best_input, best_output

    # ------------------------------------------------
    # 10) Random Search Sonuçlarının 2D ve 3D Görselleştirilmesi
    # ------------------------------------------------
    def visualize_random_search(self, search_size=1000):
        """
        Rastgele seçilen giriş değerleri için; sol tarafta 2D scatter plot (% hBN vs. Tensile Strength, 
        nokta renkleri Cure Temperature), sağ tarafta ise 3B scatter plot (% hBN, Cure Temperature, 
        Tensile Strength) oluşturur.
        """
        if not self.model:
            print("Model mevcut değil, eğitim yapmadınız.")
            return
        hbn_vals = np.random.uniform(0, 1, search_size)
        cure_vals = np.random.uniform(50, 120, search_size)
        func_vals = np.random.uniform(0, 5, search_size)
        hbn_sq_vals = hbn_vals ** 2
        X_cands = np.column_stack((hbn_vals, cure_vals, func_vals, hbn_sq_vals))
        X_cands_scaled = self.X_scaler.transform(X_cands)
        y_pred_scaled = self.model.predict(X_cands_scaled)
        y_preds = self.y_scaler.inverse_transform(y_pred_scaled)
        fig = plt.figure(figsize=(14, 6))
        # 2D Scatter Plot
        ax1 = fig.add_subplot(1, 2, 1)
        sc = ax1.scatter(hbn_vals, y_preds[:, 1], c=cure_vals, cmap='viridis', alpha=0.8)
        fig.colorbar(sc, ax=ax1, label='Cure Temperature (°C)')
        ax1.set_xlabel('% hBN')
        ax1.set_ylabel('Tensile Strength (Tahmin) [MPa]')
        ax1.set_title('2D Random Search - hBN vs. Tensile Strength')
        # 3D Scatter Plot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        p = ax2.scatter(hbn_vals, cure_vals, y_preds[:, 1], c=cure_vals, cmap='viridis', alpha=0.8)
        fig.colorbar(p, ax=ax2, label='Cure Temperature (°C)')
        ax2.set_xlabel('% hBN')
        ax2.set_ylabel('Cure Temperature (°C)')
        ax2.set_zlabel('Tensile Strength (Tahmin) [MPa]')
        ax2.set_title('3D Random Search')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    # 11) Tüm Çıktılar için 3B Surface Plot Görselleştirmesi
    # ------------------------------------------------
    def visualize_all_outputs_3d(self, hbn_points=20, cure_points=20):
        """
        % hBN ve Cure Temperature'nin etkisini, her çıktı için 3B surface plot olarak gösterir.
        '% Functionalization' sabit (örneğin 2.0) alınır; 'hBN_squared' otomatik hesaplanır.
        """
        if not self.model:
            print("Eğitilmiş model bulunamadı. Lütfen önce model eğitin.")
            return
        hbn_grid = np.linspace(0, 1, hbn_points)
        cure_grid = np.linspace(50, 120, cure_points)
        HBN, CURE = np.meshgrid(hbn_grid, cure_grid)
        func_fixed = 2.0  # Sabit functionalization değeri
        HBN_squared = HBN ** 2
        num_samples = HBN.size
        X_input = np.column_stack((
            HBN.flatten(),
            CURE.flatten(),
            np.full(num_samples, func_fixed),
            HBN_squared.flatten()
        ))
        X_input_scaled = self.X_scaler.transform(X_input)
        y_pred_scaled = self.model.predict(X_input_scaled)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        output_names = [
            "Elastic Modulus (GPa)",
            "Tensile Strength (MPa)",
            "Glass Transition Temp (°C)",
            "High Temp Strength (MPa)",
            "High Temp Modulus (GPa)"
        ]
        fig = plt.figure(figsize=(18, 12))
        for i, name in enumerate(output_names):
            ax = fig.add_subplot(3, 2, i + 1, projection='3d')
            Z = y_pred[:, i].reshape(CURE.shape)
            surf = ax.plot_surface(HBN, CURE, Z, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_xlabel('% hBN')
            ax.set_ylabel('Cure Temperature (°C)')
            ax.set_zlabel(name)
            ax.set_title(f"3B Surface Plot: {name}")
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    # 12) Keras Tuner ile Hiperparametre Araması
    # ------------------------------------------------
    def tune_with_keras_tuner(self, max_trials=20, executions_per_trial=1, epochs=100):
        """
        Keras Tuner kullanarak modelin hiperparametre aramasını yapar.
        """
        import keras_tuner as kt
        if self.X_scaled is None or self.y_scaled is None:
            raise ValueError("Veriler ölçeklendirilip hazırlanmalıdır.")
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_scaled, self.y_scaled, test_size=0.2, random_state=self.random_state
        )
        def build_model(hp):
            input_dim = self.X.shape[1]
            n_neurons = hp.Choice('n_neurons', values=[64, 128, 256])
            l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-3, sampling='log')
            dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.05)
            learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log')
            activation = hp.Choice('activation', values=['selu', 'relu', 'tanh'])
            model = Sequential([
                Dense(n_neurons, input_dim=input_dim, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dense(n_neurons, activation=activation, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate),
                Dense(n_neurons // 2, activation=activation, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate * 0.7),
                Dense(n_neurons // 4, activation=activation, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate * 0.7),
                Dense(5, activation='linear')
            ])
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
            return model
        tuner = kt.RandomSearch(
            build_model,
            objective=kt.Objective("val_loss", direction="min"),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='kt_dir',
            project_name='hbn_tuning'
        )
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
        ]
        print("\n========== Keras Tuner Araması Başlıyor ==========")
        tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),
                     callbacks=callbacks, verbose=1)
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\nEn İyi Hiperparametreler:")
        print(f" - n_neurons: {best_hyperparameters.get('n_neurons')}")
        print(f" - dropout_rate: {best_hyperparameters.get('dropout_rate')}")
        print(f" - l2_reg: {best_hyperparameters.get('l2_reg')}")
        print(f" - learning_rate: {best_hyperparameters.get('learning_rate')}")
        print(f" - activation: {best_hyperparameters.get('activation')}")
        self.model = tuner.hypermodel.build(best_hyperparameters)
        history = self.model.fit(X_train, y_train, epochs=epochs,
                                 validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
        self.history = history
        y_val_pred = self.model.predict(X_val)
        y_val_pred_inv = self.y_scaler.inverse_transform(y_val_pred)
        y_val_inv = self.y_scaler.inverse_transform(y_val)
        mse_val = mean_squared_error(y_val_inv, y_val_pred_inv)
        mae_val = mean_absolute_error(y_val_inv, y_val_pred_inv)
        r2_scores = [r2_score(y_val_inv[:, i], y_val_pred_inv[:, i]) for i in range(y_val_inv.shape[1])]
        avg_r2 = np.mean(r2_scores)
        print(f"\nTuned Model Validation MSE: {mse_val:.4f}")
        print(f"Tuned Model Validation MAE: {mae_val:.4f}")
        print(f"Tuned Model Average R²: {avg_r2:.4f}")
        print("\n=== Tuner Sonrası En İyi Model ile Maksimum Girdi Araması ===")
        best_in_tuner, best_out_tuner = self.find_optimal_inputs(search_size=2000)
        print("Optimal girdi ve çıktı değerleri yukarıda listelendi.")
        print("\n=== K-Katlı Çapraz Doğrulama METRİK RAPORU ===")
        best_trials = tuner.oracle.get_best_trials(num_trials=10)
        for trial in best_trials:
            print(f"Trial {trial.trial_id} summary")
            for hp_name, hp_value in trial.hyperparameters.values.items():
                print(f" - {hp_name}: {hp_value}")
            print(f"Score: {trial.score:.4f}\n")
        return tuner, best_hyperparameters, history

    # ------------------------------------------------
    #  Sinir Ağı Mimarisi Görselleştirme
    # ------------------------------------------------
    def visualize_model_architecture(self, filename="model_architecture.png"):
        """
        Modelin mimarisini metinsel olarak özetler (model.summary()) ve 
        grafiksel olarak 'filename' adlı dosyaya kaydeder.
        """
        if self.model is None:
            print("Model henüz oluşturulmamış.")
            return
        self.model.summary()
        from tensorflow.keras.utils import plot_model
        plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)
        print(f"Model mimarisi {filename} dosyasına kaydedildi.")

    # ------------------------------------------------
    #  Kalan (Residual) Grafikleri
    # ------------------------------------------------
    def plot_residuals(self, X_test, y_test):
        """
        Test verileri üzerinde modelin tahminleri ile gerçek değerler arasındaki farkları (residuals)
        her çıktı için histogram şeklinde görselleştirir.
        """
        y_pred = self.model.predict(X_test)
        y_pred_inv = self.y_scaler.inverse_transform(y_pred)
        y_test_inv = self.y_scaler.inverse_transform(y_test)
        residuals = y_test_inv - y_pred_inv
        target_columns = ["Elastic Modulus (GPa)",
                          "Tensile Strength (MPa)",
                          "Glass Transition Temp (°C)",
                          "High Temp Strength (MPa)",
                          "High Temp Modulus (GPa)"]
        plt.figure(figsize=(15, 8))
        for i, col in enumerate(target_columns):
            plt.subplot(2, 3, i+1)
            plt.hist(residuals[:, i], bins=20, color='skyblue', edgecolor='black')
            plt.title(f"Residuals for {col}")
            plt.xlabel("Error")
            plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    #  Özellik Önem Analizi (SHAP)
    # ------------------------------------------------
    def analyze_feature_importance(self, num_samples=100):
        """
        SHAP kullanarak modelin özellik duyarlılığını analiz eder ve her çıktı için özet grafiği oluşturur.
        """
        try:
            import shap
        except ImportError:
            print("SHAP kütüphanesi yüklü değil. Lütfen 'pip install shap' komutu ile yükleyiniz.")
            return
        if self.X_scaled is None:
            print("Önceden ölçeklendirilmiş veriler bulunamadı.")
            return
        # Rastgele seçilmiş bir arka plan (background) seti
        background = self.X_scaled[np.random.choice(self.X_scaled.shape[0], size=50, replace=False)]
        # KernelExplainer kullanarak modelin tahminlerini açıklamaya çalışın
        explainer = shap.KernelExplainer(self.model.predict, background)
        X_sample = self.X_scaled[:num_samples]
        shap_values = explainer.shap_values(X_sample)
        # Eğer modeliniz çok çıktılı ise shap_values liste olarak döner; her çıktı için ayrı özet çizdirilir.
        for i in range(len(shap_values)):
            plt.figure()
            shap.summary_plot(shap_values[i], X_sample, feature_names=["% hBN", "Cure Temperature (°C)", "% Functionalization", "hBN_squared"], show=False)
            plt.title(f"Feature Importance for Output {i+1}")
            plt.show()

    # ------------------------------------------------
    #  İnteraktif Tahmin Grafiği (Plotly)
    # ------------------------------------------------
    def plot_interactive_predictions(self, output_index=0):
        """
        Plotly Express kullanarak seçilen çıktı için tahmin ve gerçek değerleri interaktif olarak görselleştirir.
        """
        try:
            import plotly.express as px
        except ImportError:
            print("Plotly yüklü değil. Lütfen 'pip install plotly' komutu ile yükleyiniz.")
            return
        # Örnek olarak ilk 100 veri örneği kullanılıyor
        X_sample = self.X_scaled[:100]
        y_sample = self.y_scaled[:100]
        y_pred = self.model.predict(X_sample)
        y_pred_inv = self.y_scaler.inverse_transform(y_pred)
        y_actual = self.y_scaler.inverse_transform(y_sample)
        df = pd.DataFrame({
            "Actual": y_actual[:, output_index],
            "Predicted": y_pred_inv[:, output_index]
        })
        fig = px.scatter(df, x="Actual", y="Predicted", title=f"Interactive Plot for Output {output_index+1}")
        fig.show()

    # ------------------------------------------------
    #  Çapraz Doğrulama Güven Aralıkları Raporu
    # ------------------------------------------------
    def report_cv_confidence_intervals(self, n_folds):
        """
        Çapraz doğrulama metrikleri için 95% güven aralıklarını hesaplayıp raporlar.
        n_folds: Çapraz doğrulama sırasında kullanılan toplam fold sayısı.
        """
        metrics = self.performance_metrics
        print("95% Confidence Intervals for CV Metrics:")
        for metric_base in ['CV_MSE_Mean', 'CV_MAE_Mean', 'CV_R2_Mean']:
            mean_val = metrics[metric_base]
            std_val = metrics[metric_base.replace("Mean", "Std")]
            ci = 1.96 * std_val / np.sqrt(n_folds)
            print(f"{metric_base}: {mean_val:.4f} ± {ci:.4f}")

    # ------------------------------------------------
    # 10) Random Search Sonuçlarının 2D ve 3D Görselleştirilmesi
    # ------------------------------------------------
    def visualize_random_search(self, search_size=1000):
        """
        Rastgele seçilen giriş değerleri için; sol tarafta 2D scatter plot (% hBN vs. Tensile Strength, 
        nokta renkleri Cure Temperature), sağ tarafta ise 3B scatter plot (% hBN, Cure Temperature, 
        Tensile Strength) oluşturur.
        """
        if not self.model:
            print("Model mevcut değil, eğitim yapmadınız.")
            return
        hbn_vals = np.random.uniform(0, 1, search_size)
        cure_vals = np.random.uniform(50, 120, search_size)
        func_vals = np.random.uniform(0, 5, search_size)
        hbn_sq_vals = hbn_vals ** 2
        X_cands = np.column_stack((hbn_vals, cure_vals, func_vals, hbn_sq_vals))
        X_cands_scaled = self.X_scaler.transform(X_cands)
        y_pred_scaled = self.model.predict(X_cands_scaled)
        y_preds = self.y_scaler.inverse_transform(y_pred_scaled)
        fig = plt.figure(figsize=(14, 6))
        # 2D Scatter Plot
        ax1 = fig.add_subplot(1, 2, 1)
        sc = ax1.scatter(hbn_vals, y_preds[:, 1], c=cure_vals, cmap='viridis', alpha=0.8)
        fig.colorbar(sc, ax=ax1, label='Cure Temperature (°C)')
        ax1.set_xlabel('% hBN')
        ax1.set_ylabel('Tensile Strength (Tahmin) [MPa]')
        ax1.set_title('2D Random Search - hBN vs. Tensile Strength')
        # 3D Scatter Plot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        p = ax2.scatter(hbn_vals, cure_vals, y_preds[:, 1], c=cure_vals, cmap='viridis', alpha=0.8)
        fig.colorbar(p, ax=ax2, label='Cure Temperature (°C)')
        ax2.set_xlabel('% hBN')
        ax2.set_ylabel('Cure Temperature (°C)')
        ax2.set_zlabel('Tensile Strength (Tahmin) [MPa]')
        ax2.set_title('3D Random Search')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    # 11) Tüm Çıktılar için 3B Surface Plot Görselleştirmesi
    # ------------------------------------------------
    def visualize_all_outputs_3d(self, hbn_points=20, cure_points=20):
        """
        % hBN ve Cure Temperature'nin etkisini, her çıktı için 3B surface plot olarak gösterir.
        '% Functionalization' sabit (örneğin 2.0) alınır; 'hBN_squared' otomatik hesaplanır.
        """
        if not self.model:
            print("Eğitilmiş model bulunamadı. Lütfen önce model eğitin.")
            return
        hbn_grid = np.linspace(0, 1, hbn_points)
        cure_grid = np.linspace(50, 120, cure_points)
        HBN, CURE = np.meshgrid(hbn_grid, cure_grid)
        func_fixed = 2.0  # Sabit functionalization değeri
        HBN_squared = HBN ** 2
        num_samples = HBN.size
        X_input = np.column_stack((
            HBN.flatten(),
            CURE.flatten(),
            np.full(num_samples, func_fixed),
            HBN_squared.flatten()
        ))
        X_input_scaled = self.X_scaler.transform(X_input)
        y_pred_scaled = self.model.predict(X_input_scaled)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        output_names = [
            "Elastic Modulus (GPa)",
            "Tensile Strength (MPa)",
            "Glass Transition Temp (°C)",
            "High Temp Strength (MPa)",
            "High Temp Modulus (GPa)"
        ]
        fig = plt.figure(figsize=(18, 12))
        for i, name in enumerate(output_names):
            ax = fig.add_subplot(3, 2, i + 1, projection='3d')
            Z = y_pred[:, i].reshape(CURE.shape)
            surf = ax.plot_surface(HBN, CURE, Z, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_xlabel('% hBN')
            ax.set_ylabel('Cure Temperature (°C)')
            ax.set_zlabel(name)
            ax.set_title(f"3B Surface Plot: {name}")
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    # 12) Keras Tuner ile Hiperparametre Araması
    # ------------------------------------------------
    def tune_with_keras_tuner(self, max_trials=20, executions_per_trial=1, epochs=100):
        """
        Keras Tuner kullanarak modelin hiperparametre aramasını yapar.
        """
        import keras_tuner as kt
        if self.X_scaled is None or self.y_scaled is None:
            raise ValueError("Veriler ölçeklendirilip hazırlanmalıdır.")
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_scaled, self.y_scaled, test_size=0.2, random_state=self.random_state
        )
        def build_model(hp):
            input_dim = self.X.shape[1]
            n_neurons = hp.Choice('n_neurons', values=[64, 128, 256])
            l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-3, sampling='log')
            dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.05)
            learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log')
            activation = hp.Choice('activation', values=['selu', 'relu', 'tanh'])
            model = Sequential([
                Dense(n_neurons, input_dim=input_dim, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dense(n_neurons, activation=activation, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate),
                Dense(n_neurons // 2, activation=activation, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate * 0.7),
                Dense(n_neurons // 4, activation=activation, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate * 0.7),
                Dense(5, activation='linear')
            ])
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
            return model
        tuner = kt.RandomSearch(
            build_model,
            objective=kt.Objective("val_loss", direction="min"),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='kt_dir',
            project_name='hbn_tuning'
        )
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
        ]
        print("\n========== Keras Tuner Araması Başlıyor ==========")
        tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),
                     callbacks=callbacks, verbose=1)
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\nEn İyi Hiperparametreler:")
        print(f" - n_neurons: {best_hyperparameters.get('n_neurons')}")
        print(f" - dropout_rate: {best_hyperparameters.get('dropout_rate')}")
        print(f" - l2_reg: {best_hyperparameters.get('l2_reg')}")
        print(f" - learning_rate: {best_hyperparameters.get('learning_rate')}")
        print(f" - activation: {best_hyperparameters.get('activation')}")
        self.model = tuner.hypermodel.build(best_hyperparameters)
        history = self.model.fit(X_train, y_train, epochs=epochs,
                                 validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
        self.history = history
        y_val_pred = self.model.predict(X_val)
        y_val_pred_inv = self.y_scaler.inverse_transform(y_val_pred)
        y_val_inv = self.y_scaler.inverse_transform(y_val)
        mse_val = mean_squared_error(y_val_inv, y_val_pred_inv)
        mae_val = mean_absolute_error(y_val_inv, y_val_pred_inv)
        r2_scores = [r2_score(y_val_inv[:, i], y_val_pred_inv[:, i]) for i in range(y_val_inv.shape[1])]
        avg_r2 = np.mean(r2_scores)
        print(f"\nTuned Model Validation MSE: {mse_val:.4f}")
        print(f"Tuned Model Validation MAE: {mae_val:.4f}")
        print(f"Tuned Model Average R²: {avg_r2:.4f}")
        print("\n=== Tuner Sonrası En İyi Model ile Maksimum Girdi Araması ===")
        best_in_tuner, best_out_tuner = self.find_optimal_inputs(search_size=2000)
        print("Optimal girdi ve çıktı değerleri yukarıda listelendi.")
        print("\n=== K-Katlı Çapraz Doğrulama METRİK RAPORU ===")
        best_trials = tuner.oracle.get_best_trials(num_trials=10)
        for trial in best_trials:
            print(f"Trial {trial.trial_id} summary")
            for hp_name, hp_value in trial.hyperparameters.values.items():
                print(f" - {hp_name}: {hp_value}")
            print(f"Score: {trial.score:.4f}\n")
        return tuner, best_hyperparameters, history

    # ------------------------------------------------
    #  Sinir Ağı Mimarisi Görselleştirme
    # ------------------------------------------------
    def visualize_model_architecture(self, filename="model_architecture.png"):
        """
        Modelin mimarisini metinsel olarak özetler (model.summary()) ve 
        grafiksel olarak 'filename' adlı dosyaya kaydeder.
        """
        if self.model is None:
            print("Model henüz oluşturulmamış.")
            return
        self.model.summary()
        from tensorflow.keras.utils import plot_model
        plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)
        print(f"Model mimarisi {filename} dosyasına kaydedildi.")

    # ------------------------------------------------
    #  Kalan (Residual) Grafikleri
    # ------------------------------------------------
    def plot_residuals(self, X_test, y_test):
        """
        Test verileri üzerinde modelin tahminleri ile gerçek değerler arasındaki farkları (residuals)
        her çıktı için histogram şeklinde görselleştirir.
        """
        y_pred = self.model.predict(X_test)
        y_pred_inv = self.y_scaler.inverse_transform(y_pred)
        y_test_inv = self.y_scaler.inverse_transform(y_test)
        residuals = y_test_inv - y_pred_inv
        target_columns = ["Elastic Modulus (GPa)",
                          "Tensile Strength (MPa)",
                          "Glass Transition Temp (°C)",
                          "High Temp Strength (MPa)",
                          "High Temp Modulus (GPa)"]
        plt.figure(figsize=(15, 8))
        for i, col in enumerate(target_columns):
            plt.subplot(2, 3, i+1)
            plt.hist(residuals[:, i], bins=20, color='skyblue', edgecolor='black')
            plt.title(f"Residuals for {col}")
            plt.xlabel("Error")
            plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    #  Özellik Önem Analizi (SHAP)
    # ------------------------------------------------
    def analyze_feature_importance(self, num_samples=100):
        """
        SHAP kullanarak modelin özellik duyarlılığını analiz eder ve her çıktı için özet grafiğini oluşturur.
        """
        try:
            import shap
        except ImportError:
            print("SHAP kütüphanesi yüklü değil. Lütfen 'pip install shap' komutu ile yükleyiniz.")
            return
        if self.X_scaled is None:
            print("Ölçeklendirilmiş veriler bulunamadı.")
            return
        background = self.X_scaled[np.random.choice(self.X_scaled.shape[0], size=50, replace=False)]
        explainer = shap.KernelExplainer(self.model.predict, background)
        X_sample = self.X_scaled[:num_samples]
        shap_values = explainer.shap_values(X_sample)
        for i in range(len(shap_values)):
            plt.figure()
            shap.summary_plot(shap_values[i], X_sample, feature_names=["% hBN", "Cure Temperature (°C)", "% Functionalization", "hBN_squared"], show=False)
            plt.title(f"Feature Importance for Output {i+1}")
            plt.show()

    # ------------------------------------------------
    #  İnteraktif Tahmin Grafiği (Plotly)
    # ------------------------------------------------
    def plot_interactive_predictions(self, output_index=0):
        """
        Plotly Express kullanarak seçilen çıktı için tahmin ve gerçek değerleri interaktif olarak görselleştirir.
        """
        try:
            import plotly.express as px
        except ImportError:
            print("Plotly yüklü değil. Lütfen 'pip install plotly' komutu ile yükleyiniz.")
            return
        X_sample = self.X_scaled[:100]
        y_sample = self.y_scaled[:100]
        y_pred = self.model.predict(X_sample)
        y_pred_inv = self.y_scaler.inverse_transform(y_pred)
        y_actual = self.y_scaler.inverse_transform(y_sample)
        df = pd.DataFrame({
            "Actual": y_actual[:, output_index],
            "Predicted": y_pred_inv[:, output_index]
        })
        fig = px.scatter(df, x="Actual", y="Predicted", title=f"Interactive Plot for Output {output_index+1}")
        fig.show()

    # ------------------------------------------------
    #  Çapraz Doğrulama Güven Aralıkları Raporu
    # ------------------------------------------------
    def report_cv_confidence_intervals(self, n_folds):
        """
        Çapraz doğrulama metrikleri için 95% güven aralıklarını hesaplayıp raporlar.
        n_folds: Çapraz doğrulama sırasında kullanılan toplam fold sayısı.
        """
        metrics = self.performance_metrics
        print("95% Confidence Intervals for CV Metrics:")
        for metric_base in ['CV_MSE_Mean', 'CV_MAE_Mean', 'CV_R2_Mean']:
            mean_val = metrics[metric_base]
            std_val = metrics[metric_base.replace("Mean", "Std")]
            ci = 1.96 * std_val / np.sqrt(n_folds)
            print(f"{metric_base}: {mean_val:.4f} ± {ci:.4f}")

    # ------------------------------------------------
    # 10) Random Search Sonuçlarının 2D ve 3D Görselleştirilmesi
    # ------------------------------------------------
    def visualize_random_search(self, search_size=1000):
        """
        Rastgele seçilen giriş değerleri için; sol tarafta 2D scatter plot (% hBN vs. Tensile Strength, 
        nokta renkleri Cure Temperature), sağ tarafta ise 3B scatter plot (% hBN, Cure Temperature, 
        Tensile Strength) oluşturur.
        """
        if not self.model:
            print("Model mevcut değil, eğitim yapmadınız.")
            return
        hbn_vals = np.random.uniform(0, 1, search_size)
        cure_vals = np.random.uniform(50, 120, search_size)
        func_vals = np.random.uniform(0, 5, search_size)
        hbn_sq_vals = hbn_vals ** 2
        X_cands = np.column_stack((hbn_vals, cure_vals, func_vals, hbn_sq_vals))
        X_cands_scaled = self.X_scaler.transform(X_cands)
        y_pred_scaled = self.model.predict(X_cands_scaled)
        y_preds = self.y_scaler.inverse_transform(y_pred_scaled)
        fig = plt.figure(figsize=(14, 6))
        # 2D Scatter Plot
        ax1 = fig.add_subplot(1, 2, 1)
        sc = ax1.scatter(hbn_vals, y_preds[:, 1], c=cure_vals, cmap='viridis', alpha=0.8)
        fig.colorbar(sc, ax=ax1, label='Cure Temperature (°C)')
        ax1.set_xlabel('% hBN')
        ax1.set_ylabel('Tensile Strength (Tahmin) [MPa]')
        ax1.set_title('2D Random Search - hBN vs. Tensile Strength')
        # 3D Scatter Plot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        p = ax2.scatter(hbn_vals, cure_vals, y_preds[:, 1], c=cure_vals, cmap='viridis', alpha=0.8)
        fig.colorbar(p, ax=ax2, label='Cure Temperature (°C)')
        ax2.set_xlabel('% hBN')
        ax2.set_ylabel('Cure Temperature (°C)')
        ax2.set_zlabel('Tensile Strength (Tahmin) [MPa]')
        ax2.set_title('3D Random Search')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    # 11) Tüm Çıktılar için 3B Surface Plot Görselleştirmesi
    # ------------------------------------------------
    def visualize_all_outputs_3d(self, hbn_points=20, cure_points=20):
        """
        % hBN ve Cure Temperature'nin etkisini, her çıktı için 3B surface plot olarak gösterir.
        '% Functionalization' sabit (örneğin 2.0) alınır; 'hBN_squared' otomatik hesaplanır.
        """
        if not self.model:
            print("Eğitilmiş model bulunamadı. Lütfen önce model eğitin.")
            return
        hbn_grid = np.linspace(0, 1, hbn_points)
        cure_grid = np.linspace(50, 120, cure_points)
        HBN, CURE = np.meshgrid(hbn_grid, cure_grid)
        func_fixed = 2.0  # Sabit functionalization değeri
        HBN_squared = HBN ** 2
        num_samples = HBN.size
        X_input = np.column_stack((
            HBN.flatten(),
            CURE.flatten(),
            np.full(num_samples, func_fixed),
            HBN_squared.flatten()
        ))
        X_input_scaled = self.X_scaler.transform(X_input)
        y_pred_scaled = self.model.predict(X_input_scaled)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        output_names = [
            "Elastic Modulus (GPa)",
            "Tensile Strength (MPa)",
            "Glass Transition Temp (°C)",
            "High Temp Strength (MPa)",
            "High Temp Modulus (GPa)"
        ]
        fig = plt.figure(figsize=(18, 12))
        for i, name in enumerate(output_names):
            ax = fig.add_subplot(3, 2, i + 1, projection='3d')
            Z = y_pred[:, i].reshape(CURE.shape)
            surf = ax.plot_surface(HBN, CURE, Z, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_xlabel('% hBN')
            ax.set_ylabel('Cure Temperature (°C)')
            ax.set_zlabel(name)
            ax.set_title(f"3B Surface Plot: {name}")
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    # 12) Keras Tuner ile Hiperparametre Araması
    # ------------------------------------------------
    def tune_with_keras_tuner(self, max_trials=20, executions_per_trial=1, epochs=100):
        """
        Keras Tuner kullanarak modelin hiperparametre aramasını yapar.
        """
        import keras_tuner as kt
        if self.X_scaled is None or self.y_scaled is None:
            raise ValueError("Veriler ölçeklendirilip hazırlanmalıdır.")
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_scaled, self.y_scaled, test_size=0.2, random_state=self.random_state
        )
        def build_model(hp):
            input_dim = self.X.shape[1]
            n_neurons = hp.Choice('n_neurons', values=[64, 128, 256])
            l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-3, sampling='log')
            dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.05)
            learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log')
            activation = hp.Choice('activation', values=['selu', 'relu', 'tanh'])
            model = Sequential([
                Dense(n_neurons, input_dim=input_dim, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dense(n_neurons, activation=activation, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate),
                Dense(n_neurons // 2, activation=activation, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate * 0.7),
                Dense(n_neurons // 4, activation=activation, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate * 0.7),
                Dense(5, activation='linear')
            ])
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
            return model
        tuner = kt.RandomSearch(
            build_model,
            objective=kt.Objective("val_loss", direction="min"),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='kt_dir',
            project_name='hbn_tuning'
        )
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
        ]
        print("\n========== Keras Tuner Araması Başlıyor ==========")
        tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),
                     callbacks=callbacks, verbose=1)
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\nEn İyi Hiperparametreler:")
        print(f" - n_neurons: {best_hyperparameters.get('n_neurons')}")
        print(f" - dropout_rate: {best_hyperparameters.get('dropout_rate')}")
        print(f" - l2_reg: {best_hyperparameters.get('l2_reg')}")
        print(f" - learning_rate: {best_hyperparameters.get('learning_rate')}")
        print(f" - activation: {best_hyperparameters.get('activation')}")
        self.model = tuner.hypermodel.build(best_hyperparameters)
        history = self.model.fit(X_train, y_train, epochs=epochs,
                                 validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
        self.history = history
        y_val_pred = self.model.predict(X_val)
        y_val_pred_inv = self.y_scaler.inverse_transform(y_val_pred)
        y_val_inv = self.y_scaler.inverse_transform(y_val)
        mse_val = mean_squared_error(y_val_inv, y_val_pred_inv)
        mae_val = mean_absolute_error(y_val_inv, y_val_pred_inv)
        r2_scores = [r2_score(y_val_inv[:, i], y_val_pred_inv[:, i]) for i in range(y_val_inv.shape[1])]
        avg_r2 = np.mean(r2_scores)
        print(f"\nTuned Model Validation MSE: {mse_val:.4f}")
        print(f"Tuned Model Validation MAE: {mae_val:.4f}")
        print(f"Tuned Model Average R²: {avg_r2:.4f}")
        print("\n=== Tuner Sonrası En İyi Model ile Maksimum Girdi Araması ===")
        best_in_tuner, best_out_tuner = self.find_optimal_inputs(search_size=2000)
        print("Optimal girdi ve çıktı değerleri yukarıda listelendi.")
        print("\n=== K-Katlı Çapraz Doğrulama METRİK RAPORU ===")
        best_trials = tuner.oracle.get_best_trials(num_trials=10)
        for trial in best_trials:
            print(f"Trial {trial.trial_id} summary")
            for hp_name, hp_value in trial.hyperparameters.values.items():
                print(f" - {hp_name}: {hp_value}")
            print(f"Score: {trial.score:.4f}\n")
        return tuner, best_hyperparameters, history

    # ------------------------------------------------
    # Sinir Ağı Mimarisi Görselleştirme
    # ------------------------------------------------
    def visualize_model_architecture(self, filename="model_architecture.png"):
        """
        Modelin mimarisini metinsel olarak özetler (model.summary()) ve 
        grafiksel olarak 'filename' adlı dosyaya kaydeder.
        """
        if self.model is None:
            print("Model henüz oluşturulmamış.")
            return
        self.model.summary()
        from tensorflow.keras.utils import plot_model
        plot_model(self.model, to_file=filename, show_shapes=True, show_layer_names=True)
        print(f"Model mimarisi {filename} dosyasına kaydedildi.")

    # ------------------------------------------------
    # Kalan (Residual) Grafikleri
    # ------------------------------------------------
    def plot_residuals(self, X_test, y_test):
        """
        Test verileri üzerinde modelin tahminleri ile gerçek değerler arasındaki farkları (residuals)
        her çıktı için histogram şeklinde görselleştirir.
        """
        y_pred = self.model.predict(X_test)
        y_pred_inv = self.y_scaler.inverse_transform(y_pred)
        y_test_inv = self.y_scaler.inverse_transform(y_test)
        residuals = y_test_inv - y_pred_inv
        target_columns = ["Elastic Modulus (GPa)",
                          "Tensile Strength (MPa)",
                          "Glass Transition Temp (°C)",
                          "High Temp Strength (MPa)",
                          "High Temp Modulus (GPa)"]
        plt.figure(figsize=(15, 8))
        for i, col in enumerate(target_columns):
            plt.subplot(2, 3, i+1)
            plt.hist(residuals[:, i], bins=20, color='skyblue', edgecolor='black')
            plt.title(f"Residuals for {col}")
            plt.xlabel("Error")
            plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    # Özellik Önem Analizi (SHAP)
    # ------------------------------------------------
    def analyze_feature_importance(self, num_samples=100):
        """
        SHAP kullanarak modelin özellik duyarlılığını analiz eder ve her çıktı için özet grafiğini oluşturur.
        """
        try:
            import shap
        except ImportError:
            print("SHAP kütüphanesi yüklü değil. Lütfen 'pip install shap' komutu ile yükleyiniz.")
            return
        if self.X_scaled is None:
            print("Ölçeklendirilmiş veriler bulunamadı.")
            return
        background = self.X_scaled[np.random.choice(self.X_scaled.shape[0], size=50, replace=False)]
        explainer = shap.KernelExplainer(self.model.predict, background)
        X_sample = self.X_scaled[:num_samples]
        shap_values = explainer.shap_values(X_sample)
        for i in range(len(shap_values)):
            plt.figure()
            shap.summary_plot(shap_values[i], X_sample,
                              feature_names=["% hBN", "Cure Temperature (°C)", "% Functionalization", "hBN_squared"],
                              show=False)
            plt.title(f"Feature Importance for Output {i+1}")
            plt.show()

    # ------------------------------------------------
    # İnteraktif Tahmin Grafiği (Plotly)
    # ------------------------------------------------
    def plot_interactive_predictions(self, output_index=0):
        """
        Plotly Express kullanarak seçilen çıktı için tahmin ve gerçek değerleri interaktif olarak görselleştirir.
        """
        try:
            import plotly.express as px
        except ImportError:
            print("Plotly yüklü değil. Lütfen 'pip install plotly' komutu ile yükleyiniz.")
            return
        X_sample = self.X_scaled[:100]
        y_sample = self.y_scaled[:100]
        y_pred = self.model.predict(X_sample)
        y_pred_inv = self.y_scaler.inverse_transform(y_pred)
        y_actual = self.y_scaler.inverse_transform(y_sample)
        df = pd.DataFrame({
            "Actual": y_actual[:, output_index],
            "Predicted": y_pred_inv[:, output_index]
        })
        fig = px.scatter(df, x="Actual", y="Predicted", title=f"Interactive Plot for Output {output_index+1}")
        fig.show()

    # ------------------------------------------------
    # Çapraz Doğrulama Güven Aralıkları Raporu
    # ------------------------------------------------
    def report_cv_confidence_intervals(self, n_folds):
        """
        Çapraz doğrulama metrikleri için 95% güven aralıklarını hesaplayıp raporlar.
        n_folds: Çapraz doğrulama sırasında kullanılan toplam fold sayısı.
        """
        metrics = self.performance_metrics
        print("95% Confidence Intervals for CV Metrics:")
        
        # Print available keys in metrics for debugging
        print("Available metrics keys:", metrics.keys())
        
        for metric_base in ['CV_MSE_Mean', 'CV_MAE_Mean', 'CV_R2_Mean']:
            if metric_base in metrics:
                mean_val = metrics[metric_base]
                std_val = metrics[metric_base.replace("Mean", "Std")]
                ci = 1.96 * std_val / np.sqrt(n_folds)
                print(f"{metric_base}: {mean_val:.4f} ± {ci:.4f}")
            else:
                print(f"Warning: {metric_base} not found in metrics.")

    # ------------------------------------------------
    # 10) Random Search Sonuçlarının 2D ve 3D Görselleştirilmesi
    # ------------------------------------------------
    def visualize_random_search(self, search_size=1000):
        """
        Rastgele seçilen giriş değerleri için; sol tarafta 2D scatter plot (% hBN vs. Tensile Strength, 
        nokta renkleri Cure Temperature), sağ tarafta ise 3B scatter plot (% hBN, Cure Temperature, 
        Tensile Strength) oluşturur.
        """
        if not self.model:
            print("Model mevcut değil, eğitim yapmadınız.")
            return
        hbn_vals = np.random.uniform(0, 1, search_size)
        cure_vals = np.random.uniform(50, 120, search_size)
        func_vals = np.random.uniform(0, 5, search_size)
        hbn_sq_vals = hbn_vals ** 2
        X_cands = np.column_stack((hbn_vals, cure_vals, func_vals, hbn_sq_vals))
        X_cands_scaled = self.X_scaler.transform(X_cands)
        y_pred_scaled = self.model.predict(X_cands_scaled)
        y_preds = self.y_scaler.inverse_transform(y_pred_scaled)
        fig = plt.figure(figsize=(14, 6))
        # 2D Scatter Plot
        ax1 = fig.add_subplot(1, 2, 1)
        sc = ax1.scatter(hbn_vals, y_preds[:, 1], c=cure_vals, cmap='viridis', alpha=0.8)
        fig.colorbar(sc, ax=ax1, label='Cure Temperature (°C)')
        ax1.set_xlabel('% hBN')
        ax1.set_ylabel('Tensile Strength (Tahmin) [MPa]')
        ax1.set_title('2D Random Search - hBN vs. Tensile Strength')
        # 3D Scatter Plot
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        p = ax2.scatter(hbn_vals, cure_vals, y_preds[:, 1], c=cure_vals, cmap='viridis', alpha=0.8)
        fig.colorbar(p, ax=ax2, label='Cure Temperature (°C)')
        ax2.set_xlabel('% hBN')
        ax2.set_ylabel('Cure Temperature (°C)')
        ax2.set_zlabel('Tensile Strength (Tahmin) [MPa]')
        ax2.set_title('3D Random Search')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    # 11) Tüm Çıktılar için 3B Surface Plot Görselleştirmesi
    # ------------------------------------------------
    def visualize_all_outputs_3d(self, hbn_points=20, cure_points=20):
        """
        % hBN ve Cure Temperature'nin etkisini, her çıktı için 3B surface plot olarak gösterir.
        '% Functionalization' sabit (örneğin 2.0) alınır; 'hBN_squared' otomatik hesaplanır.
        """
        if not self.model:
            print("Eğitilmiş model bulunamadı. Lütfen önce model eğitin.")
            return
        hbn_grid = np.linspace(0, 1, hbn_points)
        cure_grid = np.linspace(50, 120, cure_points)
        HBN, CURE = np.meshgrid(hbn_grid, cure_grid)
        func_fixed = 2.0  # Sabit functionalization değeri
        HBN_squared = HBN ** 2
        num_samples = HBN.size
        X_input = np.column_stack((
            HBN.flatten(),
            CURE.flatten(),
            np.full(num_samples, func_fixed),
            HBN_squared.flatten()
        ))
        X_input_scaled = self.X_scaler.transform(X_input)
        y_pred_scaled = self.model.predict(X_input_scaled)
        y_pred = self.y_scaler.inverse_transform(y_pred_scaled)
        output_names = [
            "Elastic Modulus (GPa)",
            "Tensile Strength (MPa)",
            "Glass Transition Temp (°C)",
            "High Temp Strength (MPa)",
            "High Temp Modulus (GPa)"
        ]
        fig = plt.figure(figsize=(18, 12))
        for i, name in enumerate(output_names):
            ax = fig.add_subplot(3, 2, i + 1, projection='3d')
            Z = y_pred[:, i].reshape(CURE.shape)
            surf = ax.plot_surface(HBN, CURE, Z, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_xlabel('% hBN')
            ax.set_ylabel('Cure Temperature (°C)')
            ax.set_zlabel(name)
            ax.set_title(f"3B Surface Plot: {name}")
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------
    # 12) Keras Tuner ile Hiperparametre Araması
    # ------------------------------------------------
    def tune_with_keras_tuner(self, max_trials=20, executions_per_trial=1, epochs=100):
        """
        Keras Tuner kullanarak modelin hiperparametre aramasını yapar.
        """
        import keras_tuner as kt
        if self.X_scaled is None or self.y_scaled is None:
            raise ValueError("Veriler ölçeklendirilip hazırlanmalıdır.")
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_scaled, self.y_scaled, test_size=0.2, random_state=self.random_state
        )
        def build_model(hp):
            input_dim = self.X.shape[1]
            n_neurons = hp.Choice('n_neurons', values=[64, 128, 256])
            l2_reg = hp.Float('l2_reg', min_value=1e-5, max_value=1e-3, sampling='log')
            dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.05)
            learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log')
            activation = hp.Choice('activation', values=['selu', 'relu', 'tanh'])
            model = Sequential([
                Dense(n_neurons, input_dim=input_dim, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dense(n_neurons, activation=activation, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate),
                Dense(n_neurons // 2, activation=activation, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate * 0.7),
                Dense(n_neurons // 4, activation=activation, kernel_regularizer=l2(l2_reg)),
                BatchNormalization(),
                Dropout(dropout_rate * 0.7),
                Dense(5, activation='linear')
            ])
            optimizer = Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
            return model
        tuner = kt.RandomSearch(
            build_model,
            objective=kt.Objective("val_loss", direction="min"),
            max_trials=max_trials,
            executions_per_trial=executions_per_trial,
            directory='kt_dir',
            project_name='hbn_tuning'
        )
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)
        ]
        print("\n========== Keras Tuner Araması Başlıyor ==========")
        tuner.search(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val),
                     callbacks=callbacks, verbose=1)
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\nEn İyi Hiperparametreler:")
        print(f" - n_neurons: {best_hyperparameters.get('n_neurons')}")
        print(f" - dropout_rate: {best_hyperparameters.get('dropout_rate')}")
        print(f" - l2_reg: {best_hyperparameters.get('l2_reg')}")
        print(f" - learning_rate: {best_hyperparameters.get('learning_rate')}")
        print(f" - activation: {best_hyperparameters.get('activation')}")
        self.model = tuner.hypermodel.build(best_hyperparameters)
        history = self.model.fit(X_train, y_train, epochs=epochs,
                                 validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
        self.history = history
        y_val_pred = self.model.predict(X_val)
        y_val_pred_inv = self.y_scaler.inverse_transform(y_val_pred)
        y_val_inv = self.y_scaler.inverse_transform(y_val)
        mse_val = mean_squared_error(y_val_inv, y_val_pred_inv)
        mae_val = mean_absolute_error(y_val_inv, y_val_pred_inv)
        r2_scores = [r2_score(y_val_inv[:, i], y_val_pred_inv[:, i]) for i in range(y_val_inv.shape[1])]
        avg_r2 = np.mean(r2_scores)
        print(f"\nTuned Model Validation MSE: {mse_val:.4f}")
        print(f"Tuned Model Validation MAE: {mae_val:.4f}")
        print(f"Tuned Model Average R²: {avg_r2:.4f}")
        print("\n=== Tuner Sonrası En İyi Model ile Maksimum Girdi Araması ===")
        best_in_tuner, best_out_tuner = self.find_optimal_inputs(search_size=2000)
        print("Optimal girdi ve çıktı değerleri yukarıda listelendi.")
        print("\n=== K-Katlı Çapraz Doğrulama METRİK RAPORU ===")
        best_trials = tuner.oracle.get_best_trials(num_trials=10)
        for trial in best_trials:
            print(f"Trial {trial.trial_id} summary")
            for hp_name, hp_value in trial.hyperparameters.values.items():
                print(f" - {hp_name}: {hp_value}")
            print(f"Score: {trial.score:.4f}\n")
        return tuner, best_hyperparameters, history

# =============================================================================
# Sinir Ağı Mimarisi Görselleştirme Fonksiyonu (Model Summary & Plot)
# =============================================================================
def visualize_model_architecture(analyzer, filename="model_architecture.png"):
    """
    analyzer nesnesinin model özetini konsola yazdırır ve mimariyi 'filename' adlı dosyaya kaydeder.
    """
    analyzer.visualize_model_architecture(filename)

# =============================================================================
# Grafiklerin PDF Dosyasına Kaydedilmesi
# =============================================================================
def save_plots_to_pdf(pdf_filename, analyzer):
    """
    Oluşturulan tüm grafiklerin tek bir PDF dosyasına kaydedilmesini sağlar.
    """
    with PdfPages(pdf_filename) as pdf:
        # 1. Korelasyon Isı Haritası
        plt.figure(figsize=(10, 8))
        sns.heatmap(analyzer.data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title("Korelasyon Matrisi")
        pdf.savefig()
        plt.close()
        # 2. Eğitim Grafikleri
        fig = plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(analyzer.history.history['loss'], label='Train')
        plt.plot(analyzer.history.history['val_loss'], label='Validation')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 3, 2)
        plt.plot(analyzer.history.history['mae'], label='Train')
        plt.plot(analyzer.history.history['val_mae'], label='Validation')
        plt.title('MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.subplot(1, 3, 3)
        plt.plot(analyzer.history.history['mse'], label='Train')
        plt.plot(analyzer.history.history['val_mse'], label='Validation')
        plt.title('MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        pdf.savefig(fig)
        plt.close(fig)
        # 3. Model Çıktıları: % hBN vs. Çıktılar (Line Plot)
        cure_temp = 80.0
        func_degree = 2.0
        hbn_range = np.linspace(0, 1, 50)
        predictions = []
        for h in hbn_range:
            x_in = np.array([h, cure_temp, func_degree, h**2]).reshape(1, -1)
            x_scaled = analyzer.X_scaler.transform(x_in)
            y_scaled = analyzer.model.predict(x_scaled)
            y_pred = analyzer.y_scaler.inverse_transform(y_scaled)
            predictions.append(y_pred[0])
        preds = np.array(predictions)
        fig = plt.figure(figsize=(8, 6))
        plt.plot(hbn_range, preds[:, 0], label='Elastic Modulus (GPa)')
        plt.plot(hbn_range, preds[:, 1], label='Tensile Strength (MPa)')
        plt.plot(hbn_range, preds[:, 2], label='Glass Transition Temp (°C)')
        plt.plot(hbn_range, preds[:, 3], label='High Temp Strength (MPa)')
        plt.plot(hbn_range, preds[:, 4], label='High Temp Modulus (GPa)')
        plt.xlabel('% hBN')
        plt.ylabel('Tahmin Değerleri')
        plt.title(f'Çıktıların % hBN ile Değişimi\n(CureT={cure_temp}, Func={func_degree})')
        plt.legend()
        pdf.savefig(fig)
        plt.close(fig)
        # 4. Random Search: 2D ve 3D Scatter Plot
        hbn_vals = np.random.uniform(0, 1, 1000)
        cure_vals = np.random.uniform(50, 120, 1000)
        func_vals = np.random.uniform(0, 5, 1000)
        hbn_sq_vals = hbn_vals ** 2
        X_cands = np.column_stack((hbn_vals, cure_vals, func_vals, hbn_sq_vals))
        X_cands_scaled = analyzer.X_scaler.transform(X_cands)
        y_pred_scaled = analyzer.model.predict(X_cands_scaled)
        y_preds = analyzer.y_scaler.inverse_transform(y_pred_scaled)
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(1, 2, 1)
        sc = ax1.scatter(hbn_vals, y_preds[:, 1], c=cure_vals, cmap='viridis', alpha=0.8)
        fig.colorbar(sc, ax=ax1, label='Cure Temperature (°C)')
        ax1.set_xlabel('% hBN')
        ax1.set_ylabel('Tensile Strength (Tahmin) [MPa]')
        ax1.set_title('2D Random Search - hBN vs. Tensile Strength')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        p = ax2.scatter(hbn_vals, cure_vals, y_preds[:, 1], c=cure_vals, cmap='viridis', alpha=0.8)
        fig.colorbar(p, ax=ax2, label='Cure Temperature (°C)')
        ax2.set_xlabel('% hBN')
        ax2.set_ylabel('Cure Temperature (°C)')
        ax2.set_zlabel('Tensile Strength (Tahmin) [MPa]')
        ax2.set_title('3D Random Search')
        pdf.savefig(fig)
        plt.close(fig)
        # 5. Tüm Çıktılar için 3B Surface Plot
        hbn_points = 20
        cure_points = 20
        hbn_grid = np.linspace(0, 1, hbn_points)
        cure_grid = np.linspace(50, 120, cure_points)
        HBN, CURE = np.meshgrid(hbn_grid, cure_grid)
        func_fixed = 2.0
        HBN_squared = HBN ** 2
        num_samples = HBN.size
        X_input = np.column_stack((
            HBN.flatten(),
            CURE.flatten(),
            np.full(num_samples, func_fixed),
            HBN_squared.flatten()
        ))
        X_input_scaled = analyzer.X_scaler.transform(X_input)
        y_pred_scaled = analyzer.model.predict(X_input_scaled)
        y_pred = analyzer.y_scaler.inverse_transform(y_pred_scaled)
        output_names = [
            "Elastic Modulus (GPa)",
            "Tensile Strength (MPa)",
            "Glass Transition Temp (°C)",
            "High Temp Strength (MPa)",
            "High Temp Modulus (GPa)"
        ]
        fig = plt.figure(figsize=(18, 12))
        for i, name in enumerate(output_names):
            ax = fig.add_subplot(3, 2, i + 1, projection='3d')
            Z = y_pred[:, i].reshape(CURE.shape)
            surf = ax.plot_surface(HBN, CURE, Z, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_xlabel('% hBN')
            ax.set_ylabel('Cure Temperature (°C)')
            ax.set_zlabel(name)
            ax.set_title(f"3B Surface Plot: {name}")
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        pdf.savefig(fig)
        plt.close(fig)
    print(f"All plots saved to {pdf_filename}")


# =============================================================================
# Örnek Kullanım
# =============================================================================
if __name__ == "__main__":
    analyzer = HBNAnalysisSystem(random_state=42)
    analyzer.load_data(remove_outliers=False)
    X, y = analyzer.prepare_data(augment_data=True, noise_level=0.02, num_samples=50)
    tuner, best_hps, tuning_history = analyzer.tune_with_keras_tuner(max_trials=5, executions_per_trial=1, epochs=100)
    print("\n=== K-Katlı Çapraz Doğrulama METRİK RAPORU ===")
    tuner.results_summary()
    print("\n=== Nihai Model METRİK RAPORU (Train/Validation Sonu) ===")
    for k, v in analyzer.final_metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nEğitim süreci grafikleri görüntüleniyor...")
    analyzer.plot_results()
    print("\nRandom search sonuçları (2D ve 3D) gösteriliyor...")
    analyzer.visualize_random_search()
    print("\nÇıktıların % hBN ve Cure Temperature ile 3B surface plot'ları gösteriliyor...")
    analyzer.visualize_all_outputs_3d()
    print("\nTahmin sonuçları ve model metriklerini içeren PDF raporu oluşturuluyor...")
    analyzer.plot_output_vs_inputs()
    #  Sinir ağı mimarisini görselleştir (özet ve grafik dosyası)
    analyzer.visualize_model_architecture()
    #  Kalan (Residual) grafikleri çiz
    # (Örneğin, test verisi olarak hazırlanan verinin ilk 20 örneğini kullanıyoruz)
    analyzer.plot_residuals(analyzer.X_scaled[:20], analyzer.y_scaled[:20])
    #  Özellik önem analizini çalıştır (SHAP)
    analyzer.analyze_feature_importance(num_samples=100)
    #  İnteraktif tahmin grafiği (Plotly) için
    analyzer.plot_interactive_predictions(output_index=0)
    #  Çapraz doğrulama güven aralıklarını raporla (fold sayısı örneğin 5 olarak alınmıştır)
    analyzer.report_cv_confidence_intervals(n_folds=5)
    # Tüm grafiklerin PDF dosyasına kaydedilmesi
    save_plots_to_pdf("all_graphics.pdf", analyzer)
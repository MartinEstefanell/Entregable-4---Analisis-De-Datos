import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Crear carpeta outputs
output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(r'c:\Users\marti\Downloads\Titanic.csv')

print("DATOS INICIALES")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

df_pca = df.copy()

df_pca = df_pca.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

# Rellenamos valores faltantes
df_pca['Age'] = df_pca['Age'].fillna(df_pca['Age'].median())
df_pca['Embarked'] = df_pca['Embarked'].fillna(df_pca['Embarked'].mode()[0])
df_pca['Fare'] = df_pca['Fare'].fillna(df_pca['Fare'].median())

print("\nCONVERSIÓN DE VARIABLES CATEGÓRICAS")

# Sex: 0=female, 1=male
df_pca = pd.get_dummies(df_pca, columns=['Sex'], drop_first=True, dtype=int)

# Embarked: creamos variables dummy
df_pca = pd.get_dummies(df_pca, columns=['Embarked'], drop_first=True, dtype=int)

print(f"Columnas finales: {df_pca.columns.tolist()}")

print("\nCARACTERÍSTICAS DE SUPERVIVIENTES")
print("\nPromedio por grupo:")
print(df_pca.groupby('Survived').mean())

X = df_pca.drop('Survived', axis=1)
y = df_pca['Survived']

feature_names = X.columns.tolist()

# Estandarizamos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicamos PCA
pca = PCA()  # Sin especificar n_components calcula todas (8)
X_pca = pca.fit_transform(X_scaled)

print("\nVARIANZA")
for i in range(len(pca.explained_variance_ratio_)):
    print(f"PC{i+1}: {pca.explained_variance_ratio_[i]*100:.2f}%")

var_acum_pc12 = pca.explained_variance_ratio_[:2].sum()
var_acum_pc123 = pca.explained_variance_ratio_[:3].sum()

print(f"\nPC1 + PC2: {var_acum_pc12*100:.2f}%")
print(f"PC1 + PC2 + PC3: {var_acum_pc123*100:.2f}%")

# Loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i}' for i in range(1, len(pca.components_)+1)],
    index=feature_names
)

print("\nIMPORTANCIA EN PC1:")
print(loadings['PC1'].abs().sort_values(ascending=False))

# 1. Varianza
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.xlabel('Componente')
plt.ylabel('Varianza')
plt.title('Varianza por componente')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), 'o-')
plt.xlabel('Número de componentes')
plt.ylabel('Varianza acumulada')
plt.title('Varianza acumulada')
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'varianza.png'))
plt.show()

# 2. Biplot PC1 vs PC2
plt.figure(figsize=(10, 8))

colors = ['red' if s == 0 else 'blue' for s in y]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.3, s=20)

# Vectores de variables
escala = 4
for i, var in enumerate(feature_names):
    arrow_x = loadings.iloc[i, 0] * escala
    arrow_y = loadings.iloc[i, 1] * escala
    
    plt.arrow(0, 0, arrow_x, arrow_y,
              head_width=0.15, head_length=0.15, fc='black', ec='black')
    
    # Calculamos ángulo de la flecha para posicionar texto
    angle = np.arctan2(arrow_y, arrow_x)
    
    # Distancia adicional desde la punta de la flecha (aprox 1cm = 0.35 en unidades del gráfico)
    offset_distance = 0.35
    
    text_x = arrow_x + offset_distance * np.cos(angle)
    text_y = arrow_y + offset_distance * np.sin(angle)
    
    plt.text(text_x, text_y, var, fontsize=9, ha='center', va='bottom')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.title('Biplot PC1 vs PC2\n(Rojo: murió, Azul: sobrevivió)')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'biplot_pc1_pc2.png'))
plt.show()

# 3. Biplot PC2 vs PC3
plt.figure(figsize=(10, 8))

plt.scatter(X_pca[:, 1], X_pca[:, 2], c=colors, alpha=0.3, s=20)

for i, var in enumerate(feature_names):
    arrow_x = loadings.iloc[i, 1] * escala
    arrow_y = loadings.iloc[i, 2] * escala
    
    plt.arrow(0, 0, arrow_x, arrow_y,
              head_width=0.15, head_length=0.15, fc='black', ec='black')
    
    # Calculamos ángulo y posición base
    angle = np.arctan2(arrow_y, arrow_x)
    offset_distance = 0.5
    
    text_x = arrow_x + offset_distance * np.cos(angle)
    text_y = arrow_y + offset_distance * np.sin(angle)
    
    # Ajustamos nombres de variables específicas que se superponen
    if var == 'Embarked_S':
        text_x = arrow_x * 1.15
        text_y = arrow_y * 1.10
    elif var == 'Pclass':
        text_x = arrow_x * 1.2
        text_y = arrow_y * 0.7
    elif var == 'SibSp':
        text_x = arrow_x * 1.0
        text_y = arrow_y * 3.3
    elif var == 'Parch':
        text_x = arrow_x * 1.3
        text_y = arrow_y * 1.2
    
    plt.text(text_x, text_y, var, fontsize=9, ha='center', va='center')

plt.xlabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.ylabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)')
plt.title('Biplot PC2 vs PC3\n(Rojo: murió, Azul: sobrevivió)')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'biplot_pc2_pc3.png'))
plt.show()

print("\n" + "="*50)
print("ANÁLISIS COMPLETADO")
print("="*50)
print(f"\nArchivos guardados en: {output_dir}")
print("- varianza.png")
print("- biplot_pc1_pc2.png")
print("- biplot_pc2_pc3.png")


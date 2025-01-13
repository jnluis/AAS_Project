import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Define the file path
file_path = '../Data_files/conn.log.labeled'

# Manually define the expected column names
column_names = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto",
    "service", "duration", "orig_bytes", "resp_bytes", "conn_state",
    "local_orig", "local_resp", "missed_bytes", "history", "orig_pkts",
    "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tunnel_parents"
]

# Read the file
df = pd.read_csv(
    file_path,
    sep="\t",  # Separator is tab
    comment="#",  # Skip lines starting with '#'
    header=None,  # No header in data section
    skip_blank_lines=True  # Avoid blank lines
)

# Add placeholder columns if fewer than expected
while df.shape[1] < len(column_names):
    df[df.shape[1]] = "-"

# Assign column names
df.columns = column_names

# Replace `-` with NaN for better handling of missing data
df.replace("-", pd.NA, inplace=True)

# Check which columns have only NaN values
all_na_columns = df.columns[df.isna().all()].tolist()
print(f"Columns with only missing values: {all_na_columns}")

# Optionally drop columns with only NaN
df.drop(columns=all_na_columns, inplace=True)

# Handle 'tunnel_parents' column to split space-separated values
df[['tunnel_parents', 'label', 'detailed-label']] = df['tunnel_parents'].str.split(r'\s+', n=2, expand=True)

# Contar a quantidade de cada tipo de 'detailed-label'
detailed_label_counts = df['detailed-label'].value_counts()

# Verificar a quantidade da classe majoritária (DDoS)
ddos_count = detailed_label_counts['DDoS']

# Verificar a quantidade de dados da classe minoritária
minority_class_size = len(df[df['detailed-label'] != 'DDoS'])

# Definir o percentual de redução da classe DDoS
percentage_reduction = 0.6 # Exemplo: reduzir 60% da classe DDoS

# Calcular o número de amostras a serem mantidas na classe DDoS
ddos_undersample_size = int(ddos_count * (1 - percentage_reduction))

# Realizar undersampling na classe DDoS para a quantidade calculada
df_ddos_undersampled = df[df['detailed-label'] == 'DDoS'].sample(ddos_undersample_size, random_state=42)

# Manter a parte dos dados que não são DDoS
df_other = df[df['detailed-label'] != 'DDoS']

# Concatenar as duas partes balanceadas
df_balanced = pd.concat([df_ddos_undersampled, df_other])

# Embaralhar o DataFrame final para aleatorizar os dados
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Exibir a nova distribuição
print(df_balanced['detailed-label'].value_counts())

# Contar a quantidade de cada tipo de 'label'
detailed_label_counts = df_balanced['label'].value_counts()

# Verificar a quantidade da classe majoritária (Malicious)
malicious_count = detailed_label_counts['Malicious']

# Verificar a quantidade de dados da classe minoritária (Benign)
benign_count = detailed_label_counts['Benign']

# Definir o percentual de redução da classe Malicious
percentage_reduction_malicious = 0.7  # Exemplo: reduzir 70% da classe Malicious

# Dividir os dados da classe Malicious em duas partes:
# 1. Parte com detailed_label = 'PartOfAHorizontalPortScan'
df_malicious_partscan = df_balanced[(df_balanced['label'] == 'Malicious') & 
                                    (df_balanced['detailed-label'] == 'PartOfAHorizontalPortScan')]

# 2. Parte restante da classe Malicious (sem 'PartOfAHorizontalPortScan')
df_malicious_others = df_balanced[(df_balanced['label'] == 'Malicious') & 
                                   (df_balanced['detailed-label'] != 'PartOfAHorizontalPortScan')]

# Calcular o número de amostras a serem mantidas na parte restante da classe Malicious
malicious_undersample_size = int(df_malicious_others.shape[0] * (1 - percentage_reduction_malicious))

# Realizar undersampling na parte restante da classe Malicious
df_malicious_undersampled = df_malicious_others.sample(malicious_undersample_size, random_state=42)

# Manter a parte dos dados que não são Malicious
df_other = df_balanced[df_balanced['label'] != 'Malicious']

# Concatenar a parte não alterada de 'PartOfAHorizontalPortScan', 
# a parte reduzida de Malicious e os dados de outras classes
df_balanced = pd.concat([df_malicious_partscan, df_malicious_undersampled, df_other])

# Embaralhar o DataFrame final para aleatorizar os dados
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Exibir a nova distribuição
print(df_balanced['label'].value_counts())

# Mostrar gráficos novamente para verificar a distribuição balanceada
detailed_label_counts_balanced = df_balanced['label'].value_counts()
print(df_balanced['detailed-label'].value_counts())

df_balanced.drop(columns=['ts','uid','detailed-label','tunnel_parents'], inplace=True)

df_balanced['label'] = df_balanced['label'].replace({'Malicious': 0, 'Benign': 1})






# Replace <NA> with np.nan in 'duration' column
df_balanced['duration'] = df_balanced['duration'].replace({pd.NA: np.nan})

# Step 1: Impute missing values in the 'duration' column (impute before transformation)
imputer = SimpleImputer(strategy='median')
df_balanced['duration'] = imputer.fit_transform(df_balanced[['duration']])

# Step 2: Apply the log transformation to the 'duration' column
df_balanced['duration'] = df_balanced['duration'].apply(lambda x: np.log(x + 1) if x > 0 else 0)

# Step 3: Standardize the 'duration' column
scaler = StandardScaler()
df_balanced['duration'] = scaler.fit_transform(df_balanced[['duration']])






columns_to_encode = ['id.orig_h', 'id.resp_h', 'proto', 'service', 'conn_state', 'history']

# Substitui valores NA por 'No_Value'
df_balanced[columns_to_encode] = df_balanced[columns_to_encode].replace({pd.NA: 'No_Value'})

# Inicializa um DataFrame para os dados codificados
encoded_columns_df = pd.DataFrame()

for column in columns_to_encode:
    if column == 'id.resp_h':
        # Usa OneHotEncoder com max_categories=3 para 'id.resp_h'
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=3)
    else:
        # Usa o OneHotEncoder padrão para outras colunas
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', max_categories=7)
    
    # Aplica o encoder na coluna
    encoded_column = encoder.fit_transform(df_balanced[[column]])
    
    # Converte a coluna codificada para DataFrame
    encoded_df = pd.DataFrame(
        encoded_column, 
        columns=encoder.get_feature_names_out([column])
    )
    
    # Concatena com o DataFrame de dados codificados
    encoded_columns_df = pd.concat([encoded_columns_df, encoded_df], axis=1)

# Concatena as colunas codificadas no DataFrame original
df_balanced_encoded = pd.concat([encoded_columns_df, df_balanced], axis=1)

# Remove as colunas originais
df_balanced_encoded.drop(columns=columns_to_encode, inplace=True)







# Ensure columns are numeric before filling NaNs
df_balanced_encoded['orig_bytes'] = pd.to_numeric(df_balanced_encoded['orig_bytes'], errors='coerce')
df_balanced_encoded['resp_bytes'] = pd.to_numeric(df_balanced_encoded['resp_bytes'], errors='coerce')

# Fill missing values with mean for numeric columns (assigning back to the column)
df_balanced_encoded['orig_bytes'] = df_balanced_encoded['orig_bytes'].fillna(df_balanced_encoded['orig_bytes'].mean())
df_balanced_encoded['resp_bytes'] = df_balanced_encoded['resp_bytes'].fillna(df_balanced_encoded['resp_bytes'].mean())

# For categorical columns, use the mode instead (assigning back to the column)
df_balanced_encoded['local_resp'] = df_balanced_encoded['local_resp'].fillna(df_balanced_encoded['local_resp'].mode()[0])
df_balanced_encoded['label'] = df_balanced_encoded['label'].fillna(df_balanced_encoded['label'].mode()[0])

# Calculate the mode of the 'missed_bytes' column
mode_value = df_balanced_encoded['missed_bytes'].mode()[0]

# Replace 'C' with the mode value
df_balanced_encoded['missed_bytes'] = df_balanced_encoded['missed_bytes'].replace('C', mode_value)


#Remove row with column "resp_ip_bytes" = "-   Malicious   DDoS"
df_balanced_encoded = df_balanced_encoded[df_balanced_encoded['resp_ip_bytes'] != '-   Malicious   DDoS']

#remove columns all zero columns
columns_to_remove = [col for col in df_balanced_encoded.columns if df_balanced_encoded[col].nunique() <= 1]
df_balanced_encoded = df_balanced_encoded.drop(columns=columns_to_remove)



#Grafico com a quantidade de valores benignos e maliciosos
#Coloca o número de amostras de cada classe em um gráfico de barras
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df_balanced_encoded)
plt.title("Distribution of Labels")
plt.show()



# Check the distribution of labels after undersampling
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df_balanced_encoded)
plt.title("Distribution of Labels")
plt.show()



# Now, calculate the correlation matrix again
correlation_matrix = df_balanced_encoded.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()


train_df, test_df = train_test_split(df_balanced_encoded, test_size=0.9, random_state=42)

# Check the distribution in train and test sets
print("Training set distribution:")
print(train_df['label'].value_counts())

print("\nTesting set distribution:")
print(test_df['label'].value_counts())

# Save the test DataFrame to a CSV file
train_df.to_csv('../Data_files/train_data.csv', index=False)
print("Train data saved to 'train_data.csv'.")

test_df.to_csv('../Data_files/test_data.csv', index=False)
print("Test data saved to 'test_data.csv'.")


#Print the name of the columns
print(train_df.columns)


#how many malicious samples are in the test set
malicious_samples = test_df[test_df['label'] == 0]
print(f"Number of malicious samples in the test set: {malicious_samples.shape[0]}")
benign_samples = test_df[test_df['label'] == 1]
print(f"Number of benign samples in the test set: {benign_samples.shape[0]}")

malicious_sampled = malicious_samples.sample(n=200, random_state=42)
benign_sampled = benign_samples.sample(n=150, random_state=42)

mcc_test_df = pd.concat([malicious_sampled, benign_sampled])

# Save the DataFrame to a CSV file
mcc_test_df.to_csv('../Data_files/mcc_test.csv', index=False)

print(f"mcc_test.csv created with {mcc_test_df.shape[0]} samples.")

# Remove the sampled rows from the original test_df
test_df = test_df.drop(malicious_sampled.index)
test_df = test_df.drop(benign_sampled.index)

# Save the modified test_df if you need to
test_df.to_csv('../Data_files/test_data.csv', index=False)

print(f"mcc_test.csv created with {mcc_test_df.shape[0]} samples. The test dataset has been updated.")



# Import necessary libraries
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

# Separate features (X) and labels (y) for training and testing sets
X_train = train_df.drop(columns=['label'])
y_train = train_df['label']

X_test = test_df.drop(columns=['label'])
y_test = test_df['label']

# Initialize Logistic Regression model
logreg_model = SGDClassifier(loss='log_loss', random_state=42, max_iter=1000, alpha=0.001)

# Train the model
logreg_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg_model.predict(X_test)

# Ensure the 'Model' folder exists
os.makedirs('../Model', exist_ok=True)
model_path = '../Model/logreg_model.pkl'
joblib.dump(logreg_model, model_path)
print(f"Model saved to '{model_path}'.")

# Evaluate the model
print("Model Performance on Test Data:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Specify the path to the text file
accuracy_file_path = '../Model/final_accuracy.txt'
# Write the final accuracy to the text file
with open(accuracy_file_path, 'a') as f:
    f.write(f"Final Accuracy: {accuracy_score(y_test, y_pred):.4f} - Inicial\n")

print(f"Final accuracy saved to {accuracy_file_path}")







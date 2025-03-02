import pandas as pd
import matplotlib.pyplot as plt
import math

# -------------------------------
# 1. Carregamento dos datasets
# -------------------------------
try:
    from ucimlrepo import fetch_ucirepo
    data_fetch_available = True
except ImportError:
    data_fetch_available = False

# -------------------------------
# 2. Função para divisão estratificada
# -------------------------------
def stratified_split(df, target_col, train_ratio=0.7):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for cls in df[target_col].unique():
        cls_data = df[df[target_col] == cls]
        train_size = int(len(cls_data) * train_ratio)
        cls_train = cls_data.sample(n=train_size, random_state=42)
        cls_test = cls_data.drop(cls_train.index)
        train_df = pd.concat([train_df, cls_train])
        test_df = pd.concat([test_df, cls_test])
    return train_df.sample(frac=1, random_state=42), test_df.sample(frac=1, random_state=42)

# -------------------------------
# 3. Implementação do Naive Bayes
# -------------------------------
class NaiveBayesClassifier:
    def fit(self, X, y):
        self.classes = y.unique()
        self.features = X.columns
        # Probabilidades a priori
        self.priors = {c: (y == c).sum() / len(y) for c in self.classes}
        # Probabilidades condicionais com suavização Laplace
        self.likelihoods = {}
        for feature in self.features:
            self.likelihoods[feature] = {}
            feature_values = X[feature].unique()
            for val in feature_values:
                self.likelihoods[feature][val] = {}
                for c in self.classes:
                    subset = X[(X[feature] == val) & (y == c)]
                    count = len(subset)
                    total = (y == c).sum()
                    n_values = len(X[feature].unique())
                    self.likelihoods[feature][val][c] = (count + 1) / (total + n_values)
    
    def predict_proba(self, X):
        probs = []
        for _, row in X.iterrows():
            class_probs = {}
            for c in self.classes:
                prob = self.priors[c]
                for feature in self.features:
                    val = row[feature]
                    # Se o valor não foi visto durante o treinamento, usa fator neutro 1
                    if val in self.likelihoods[feature]:
                        prob *= self.likelihoods[feature][val].get(c, 1)
                    else:
                        prob *= 1
                class_probs[c] = prob
            # Normalização das probabilidades
            total = sum(class_probs.values())
            for c in class_probs:
                class_probs[c] /= total
            probs.append(class_probs)
        return probs
    
    def predict(self, X):
        proba = self.predict_proba(X)
        predictions = [max(p, key=p.get) for p in proba]
        return predictions

# -------------------------------
# 4. Implementação do ID3
# -------------------------------
class ID3Classifier:
    def fit(self, X, y):
        data = X.copy()
        data['label'] = y
        features = data.columns.drop('label')
        self.tree = self._build_tree(data, features)
    
    def _entropy(self, labels):
        counts = labels.value_counts()
        entropy = 0
        for count in counts:
            p = count / len(labels)
            entropy -= p * math.log2(p)
        return entropy
    
    def _build_tree(self, data, features):
        # Se todos os exemplos possuem a mesma classe, cria nó folha
        if len(data['label'].unique()) == 1:
            return {'label': data['label'].iloc[0]}
        # Se não há mais atributos, retorna a classe majoritária
        if len(features) == 0:
            return {'label': data['label'].mode()[0]}
        
        current_entropy = self._entropy(data['label'])
        best_gain = 0
        best_feature = None
        best_splits = None
        
        # Escolhe o atributo com maior ganho de informação
        for feature in features:
            splits = {}
            for val in data[feature].unique():
                splits[val] = data[data[feature] == val]
            weighted_entropy = sum((len(subset)/len(data)) * self._entropy(subset['label']) 
                                   for subset in splits.values())
            gain = current_entropy - weighted_entropy
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_splits = splits
        
        if best_feature is None:
            return {'label': data['label'].mode()[0]}
        
        tree = {best_feature: {}}
        remaining_features = features.drop(best_feature)
        for val, subset in best_splits.items():
            if subset.empty:
                tree[best_feature][val] = {'label': data['label'].mode()[0]}
            else:
                tree[best_feature][val] = self._build_tree(subset, remaining_features)
        return tree
    
    def _predict_single(self, tree, instance):
        if 'label' in tree:
            return tree['label']
        # A árvore possui uma chave: o atributo usado na divisão
        feature = list(tree.keys())[0]
        feature_val = instance[feature]
        if feature_val in tree[feature]:
            return self._predict_single(tree[feature][feature_val], instance)
        else:
            # Se o valor não foi visto, utiliza um rótulo padrão (pode ser aprimorado)
            return self._majority_class(self.tree)
    
    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            pred = self._predict_single(self.tree, row)
            predictions.append(pred)
        return predictions
    
    def _majority_class(self, tree):
        # Função auxiliar para retornar a classe majoritária encontrada na árvore
        if 'label' in tree:
            return tree['label']
        counts = {}
        feature = list(tree.keys())[0]
        for subtree in tree[feature].values():
            label = self._majority_class(subtree)
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
        if counts:
            return max(counts, key=counts.get)
        return None

# -------------------------------
# 5. Função para avaliação e plotagem da matriz de confusão
# -------------------------------
def compute_confusion_matrix(y_true, y_pred, classes):
    cm = [[0 for _ in classes] for _ in classes]
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    for true, pred in zip(y_true, y_pred):
        i = class_to_index[true]
        j = class_to_index[pred]
        cm[i][j] += 1
    return cm

def plot_confusion_matrix(cm, classes, title="Matriz de Confusão"):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = range(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    plt.xlabel("Rótulo Predito", fontsize=14)
    plt.ylabel("Rótulo Real", fontsize=14)
    
    thresh = cm[0][0] if len(cm) == 1 else max(max(row) for row in cm) / 2.0
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, format(cm[i][j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i][j] > thresh else "black",
                     fontsize=14)
    plt.tight_layout()
    plt.show()

def evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = sum(pred == true for pred, true in zip(predictions, y_test)) / len(y_test)
    # Cálculo e plotagem da matriz de confusão
    classes = list(y_test.unique())
    cm = compute_confusion_matrix(y_test.tolist(), predictions, classes)
    plot_confusion_matrix(cm, classes, title="Matriz de Confusão")
    return accuracy

def run_experiment(dataset_name, X, y):
    print("Dataset:", dataset_name)
    data = X.copy()
    data['label'] = y
    train_data, test_data = stratified_split(data, 'label', train_ratio=0.7)
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']
    
    # Avaliação do Naive Bayes
    print("\n--- Naive Bayes ---")
    nb = NaiveBayesClassifier()
    acc_nb = evaluate_classifier(nb, X_train, y_train, X_test, y_test)
    print("Acurácia Naive Bayes:", acc_nb)
    
    # Avaliação do ID3
    print("\n--- ID3 ---")
    id3 = ID3Classifier()
    acc_id3 = evaluate_classifier(id3, X_train, y_train, X_test, y_test)
    print("Acurácia ID3:", acc_id3)
    
    return acc_nb, acc_id3

datasets_results = {}

if data_fetch_available:
    # ----- Congressional Voting Records -----
    congressional = fetch_ucirepo(id=105)
    X_cong = congressional.data.features
    y_cong = congressional.data.targets
    acc_nb, acc_id3 = run_experiment("Congressional Voting Records", X_cong, y_cong)
    datasets_results["Congressional Voting Records"] = (acc_nb, acc_id3)
    
    # ----- Breast Cancer -----
    breast = fetch_ucirepo(id=14)
    X_breast = breast.data.features
    y_breast = breast.data.targets
    acc_nb, acc_id3 = run_experiment("Breast Cancer", X_breast, y_breast)
    datasets_results["Breast Cancer"] = (acc_nb, acc_id3)
else:
    print("Módulo ucimlrepo não disponível. Carregue os datasets manualmente (ex.: pd.read_csv).")

# -------------------------------
# 6. Visualização dos resultados finais
# -------------------------------
if datasets_results:
    datasets = list(datasets_results.keys())
    nb_accuracies = [datasets_results[d][0] for d in datasets]
    id3_accuracies = [datasets_results[d][1] for d in datasets]

    x = range(len(datasets))
    width = 0.35

    plt.figure(figsize=(8,6))
    plt.bar(x, nb_accuracies, width, label='Naive Bayes')
    plt.bar([p + width for p in x], id3_accuracies, width, label='ID3')
    plt.xlabel('Datasets', fontsize=14)
    plt.ylabel('Acurácia', fontsize=14)
    plt.title('Comparação de Acurácia: Naive Bayes vs ID3', fontsize=16)
    plt.xticks([p + width/2 for p in x], datasets, fontsize=12)
    plt.ylim(0,1)
    plt.legend(fontsize=12)
    plt.show()

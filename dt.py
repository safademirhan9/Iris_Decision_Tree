class DecisionTree:
    def __init__(self, max_depth=2, min_size=1):
        self.max_depth = max_depth
        self.model = None
        self.min_size = min_size

    def fit(self, X_train_list):
        root = DecisionTreeClassifier.get_split(X_train_list)
        DecisionTreeClassifier.split(root, self.max_depth, self.min_size, 1)
        self.model = DecisionTreeClassifier.build_tree(X_train_list, self.max_depth, self.min_size)
    
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            prediction = DecisionTreeClassifier.predict(self.model, row)
            prediction = round(prediction)
            predictions.append(prediction)
        return predictions

class DecisionTreeClassifier:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth

    # Bir oznitelige gore veri setini boler
    def test_split(index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Gini burada hesaplaniyor
    def gini_index(groups, classes):
        n_instances = float(sum([len(group) for group in groups]))
        # Her grup icin Gini indis toplami 
        gini = 0.0
        for group in groups:
            size = float(len(group))
            if size == 0:
                continue
            score = 0.0
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            gini += (1.0 - score) * (size / n_instances)
        return gini

    # En iyi bolme seklinin secilmesini saglar
    def get_split(dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = DecisionTreeClassifier.test_split(index, row[index], dataset)
                gini = DecisionTreeClassifier.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    def to_terminal(group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Child node'lar olusturur veya agaci tamamlar
    # Max depth, bolme olmamasi ve cocuk node durumlarina bakar
    def split(node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = DecisionTreeClassifier.to_terminal(left + right)
            return
        if depth >= max_depth:
            node['left'], node['right'] = DecisionTreeClassifier.to_terminal(left), DecisionTreeClassifier.to_terminal(right)
            return
        if len(left) <= min_size:
            node['left'] = DecisionTreeClassifier.to_terminal(left)
        else:
            node['left'] = DecisionTreeClassifier.get_split(left)
            DecisionTreeClassifier.split(node['left'], max_depth, min_size, depth+1)
        if len(right) <= min_size:
            node['right'] = DecisionTreeClassifier.to_terminal(right)
        else:
            node['right'] = DecisionTreeClassifier.get_split(right)
            DecisionTreeClassifier.split(node['right'], max_depth, min_size, depth+1)

    # Karar agacini olusturur
    def build_tree(train, max_depth, min_size):
        root = DecisionTreeClassifier.get_split(train)
        DecisionTreeClassifier.split(root, max_depth, min_size, 1)
        return root

    # Karar agaci ile tahmin yapar
    def predict(node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return DecisionTreeClassifier.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return DecisionTreeClassifier.predict(node['right'], row)
            else:
                return node['right']

if __name__ == '__main__':

    pass
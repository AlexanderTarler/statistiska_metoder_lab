import numpy as np
import scipy.stats as stats


class LinearRegression:

    def __init__(self, confidence_level=0.95):
        self.coefficients = None

        self.intercept = None

        self._confidence_level = confidence_level

    @property
    def number_of_features(self):
        """
        Returnerar antalet features (kolumner/dimensioner) i modellen.
        """
        return self._dimensions

    @property
    def sample_size(self):
        """
        Returnerar antalet datapunkter (rader) i modellen.
        """
        return self._sample_size

    @property
    def confidence_level(self):
        """
        Returnerar confidence level för modellen.
        """
        return self._confidence_level

    @confidence_level.setter
    def confidence_level(self, new_level):
        """
        Uppdaterar confidence level och ser till att den är mellan 0 och 1.
        """
        if 0 < new_level < 1:
            self._confidence_level = new_level
        else:
            raise ValueError("Konfidensnivån måste vara mellan 0 och 1.")

    def fit(self, X, y):

        # Om X är en en-dimensionell array så ser koden till att göra den till en kolumn (2D-array) för att kunna använda den i matrisberäkningar.
        if X.ndim == 1:
            X = X[:, np.newaxis]

        # Här lägger jag till en kolumn av ettor för intercept (β0), dvs vad y är när x = 0.
        X = np.c_[np.ones(X.shape[0]), X]

        # Nedanför använder jag OLS (Least Ordinary Squares) för att hitta den bästa passningen för en linje genom datapunkterna.
        self.coefficients = np.linalg.pinv(X.T @ X) @ X.T @ y

        # För att hitta interceptet så tar jag första koefficienten.
        self.intercept = self.coefficients[0]
        # Jag sorterar sedan bort interceptet från koefficienterna så att bara lutningarna är kvar.
        self.coefficients = self.coefficients[1:]

        # Här ger jag variablerna _dimensions och _sample_size värden.
        self._dimensions = X.shape[1] - 1
        self._sample_size = X.shape[0]

    def predict(self, X):
        """
        Gör en förutsägelse för varje datapunkt i X.
        """

        if X.ndim == 1:
            X = X[:, np.newaxis]

        X = np.c_[np.ones(X.shape[0]), X]

        # I koden nedan så multiplicerar jag X med koefficienterna för att få en förutsägelse för y.
        # Jag returnerar sedan en matris där varje rad är en förutsägelse för varje datapunkt i X.

        prediction_results = X @ np.r_[self.intercept, self.coefficients]
        return prediction_results

    def r_squared(self, X, y):
        """
        Beräknar R² (förklaringsgraden) för modellen. 
        """

        predicted_values = self.predict(X)

        # Beräknar SSE (Sum of Squared Errors)
        SSE = np.sum((y - predicted_values) ** 2)
        # Beräknar SST (Total Sum of Squares)
        SST = np.sum((y - np.mean(y)) ** 2)

        # Beräknar R²
        R2 = 1 - SSE / SST
        return R2

    def variance(self, X, y):
        """
        Beräknar variansen (σ²) för modellen. 
        """

        predicted_values = self.predict(X)

        SSE = np.sum((y - predicted_values) ** 2)

        n = self.sample_size  # Stickprovsstorleken
        d = self.number_of_features  # Antal dimensioner

        variance = SSE / (n - d - 1)
        return variance

    def standard_deviation(self, X, y):
        """
        Beräknar standardavvikelsen (σ) för modellen. 
        """

        # Beräknar variansen (σ²) för modellen
        variance = self.variance(X, y)

        # Beräknar standardavvikelsen (σ) genom att ta kvadratroten av variansen (σ²)
        standard_deviation = np.sqrt(variance)
        return standard_deviation

    def significance(self, X, y):
        """
        Beräknar F-statistik och p-värde för regressionens signifikans.
        """
        predicted_values = self.predict(X)

        sse = np.sum((y - predicted_values) ** 2)

        sst = np.sum((y - np.mean(y)) ** 2)

        ssr = sst - sse

        n = self.sample_size
        d = self.number_of_features

        variance = self.variance(X, y)

        f_stat = (ssr / d) / variance

        p_value = stats.f.sf(f_stat, d, n - d - 1)

        return f_stat, p_value

    def individual_significance(self, X, y):
        """
        Beräknar T-statistik och p-värde för varje individuell koefficient.
        Om X endast innehåller en binär variabel (Observer) används Welch's t-test.
        """
        if X.ndim == 1:
            X = X[:, np.newaxis]

        c = np.linalg.pinv(X.T @ X) * self.variance(X, y)

        # Standardberäkning för flera variabler
        t_values = [self.coefficients[i] /
                    (np.sqrt(c[i, i])) for i in range(len(self.coefficients))]

        dof = self.sample_size - self.number_of_features - 1
        p_values = [2 * min(stats.t.cdf(t, dof), stats.t.sf(t, dof))
                    for t in t_values]

        return t_values, p_values

    def pearson_correlation_matrix(self, X, feature_names):
        """
        Beräknar Pearsons korrelationskoefficient mellan alla par av features i X.
        """
        if X.ndim == 1:
            X = X[:, np.newaxis]

        num_features = X.shape[1]
        correlation_matrix = {}

        for i in range(num_features):
            for j in range(i + 1, num_features):
                feature_1 = feature_names[i]
                feature_2 = feature_names[j]
                correlation, p_value = stats.pearsonr(X[:, i], X[:, j])
                correlation_matrix[(feature_1, feature_2)] = (
                    correlation, p_value)

        return correlation_matrix

    def confidence_intervals(self, X, y):
        """
        Beräknar konfidensintervall för varje regressionskoefficient i modellen.
        """

        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        if X.ndim == 1:
            X = X[:, np.newaxis]

        _, features = X.shape

        standard_deviation = self.standard_deviation(X, y)

        # Jag använder np.linalg.pinv för att invertera matrisen och då få "cii​" (diagonalen i inverterade matrisen).
        covariance_matrix = np.linalg.pinv(X.T @ X) * standard_deviation**2

        # Signifikansnivån (alpha = 1 - konfidensnivån) används för att beräkna marginalen av fel i intervallet
        alpha = 1 - self.confidence_level

        dof = self.sample_size - self.number_of_features - 1

        # det kritiska t-värdet används för att beräkna  marginalen av fel i intervallet.
        # ppf står för "percent point function" och används för att beräkna det kritiska t-värdet för en viss konfidensnivå.
        t_critical = stats.t.ppf(1 - alpha/2, dof)  # Här får vi fram tα/2​

        confidence_intervals = {}

        for i in range(features):
            coefficient_value = self.coefficients[i]
            standard_error = np.sqrt(covariance_matrix[i, i])
            margin_of_error = t_critical * standard_deviation * standard_error

            lower_bound = coefficient_value - margin_of_error
            upper_bound = coefficient_value + margin_of_error

            confidence_intervals[f"Coefficient {i}"] = (
                coefficient_value, margin_of_error, lower_bound, upper_bound)

        return confidence_intervals

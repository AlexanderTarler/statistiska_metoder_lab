import numpy as np
import scipy.stats as stats


data_path = "./data/Small-diameter-flow.csv"

# Testdata
X = np.array([1, 2, 3, 4, 5])  # Oberoende variabel
y = np.array([2.2, 2.8, 4.5, 3.7, 5.5])  # Beroende variabel


class LinearRegression:
    def __init__(self):
        """
        Initialiserar klassen med tomma attribut.
        """
        # Här sparas lutningen (Hur mycket y ökar per enhet x)
        self.coefficients = None
        # Här sparas intercept (skärningspunkten med y-axeln, dvs vad y är när x = 0)
        self.intercept = None

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

    def fit(self, X, y):
        """
        Denna metod ska räkna ut interceptet och koefficienterna för linjär regression.
        Den beräknar koefficienterna med Least Squares Estimation (LSE).

        Formeln för LSE är: β = (X^T * X)^-1 * X^T * y

        X: insatsdata (features/oberoende variabler), en matris. 
            X är inputmatrisen som innehåller alla oberoende variabler (features).
            Varje kolumn i X representerar en oberoende variabel.
            Varje rad i X representerar en datapunkt (en observation).

        y: målvärden (beroende variabel).

        """

        # Om X är en en-dimensionell array, gör [:, np.newaxis] det till en kolumn (2D-array)
        # ":" representerar här att vi vill ha alla rader, och "np.newaxis" lägger till en ny axel (kolumn) i arrayen.
        # Om raderna t.ex är [1, 2, 3] blir arrayen till [[1], [2], [3]].

        if X.ndim == 1:
            X = X[:, np.newaxis]

        # Lägg till en kolumn av ettor för intercept (β0), dvs vad y är när x = 0
        # np.c_ är en funktion som används för att sammanfoga arrayer längs kolumnaxeln, där np.ones(X.shape[0]) skapar en array av ettor med samma antal rader som X.
        # np.c_[np.ones(X.shape[0]), X] lägger sedan in arrayern av ettor som första kolumn i X. X.shape[0] betyder att vi vill att antalet rader av ettor ska vara lika med antalet rader i X.
        # PS: ifall vi skulle byta plats på np.ones(X.shape[0]) och X så skulle vi få en array där kolumnen med ettor är sist.

        X = np.c_[np.ones(X.shape[0]), X]

        # Beräkna koefficienterna med LSE (Least Squares Estimation)

        # koefficienterna är en vektor som innehåller interceptet och lutningen för varje oberoende variabel.
        # Förenklat så kan man se β som "k" i en linjär ekvation y = kx + m, där k är lutningen och m är interceptet.

        # Formeln för LSE är β = (X^T * X)^-1 * X^T * y
        # "np.linalg.inv" används för att invertera en matris.
        # "@"" används för att multiplicera matriser.
        # "".T" används för att transponera en matris. Exempel: om X är en matris så är X.T transponeringen av X.

        # Nedanför så tar vi alltså klassens attribut "coefficients" och sätter det till att vara lika med (X^T * X)^-1 * X^T * y.
        # Uträkningen görs i tre steg:
        # 1. X^T * X (X transponerat multiplicerat med X)
        # 2. (X^T * X)^-1 (invertering av X^T * X)
        # 3. (X^T * X)^-1 * X^T * y (multiplikation av inverteringen av X^T * X med X^T och y).

        self.coefficients = np.linalg.inv(X.T @ X) @ X.T @ y
        print("Koefficienter (inklusive intercept):", self.coefficients)

        # Spara interceptet och koefficienterna i separata attribut
        # Interceptet är den första koefficienten, och resten av koefficienterna är lutningarna för varje oberoende variabel.
        self.intercept = self.coefficients[0]
        # Ta bort interceptet från koefficienterna så att vi bara har lutningarna kvar.
        self.coefficients = self.coefficients[1:]

        # Antal kolumner i X exklusive intercept
        self._dimensions = X.shape[1] - 1

        # Antalet rader i X
        self._sample_size = X.shape[0]

    def predict(self, X):
        """
        Här ska vi skriva predict-metoden som ska ta nya X-värden och använda koefficienterna för att förutsäga y-värden.

        Formel: y^ = β0 + β1*x1 + β2*x2 + ... + βn*xn
        I matrisform: y^ = X * β
        Där β=[β0, β1, β2, ... , βn] och XX är matrisen med inputvärden (inklusive kolumnen med ettor för interceptet).
        """

        #  Återigen så kollar vi om X är en en-dimensionell array och om den är det så använder vi [:, np.newaxis] för att göra den till en kolumn (2D-array).
        if X.ndim == 1:
            X = X[:, np.newaxis]

        # Likt ovan så lägger vi till en kolumn av ettor för intercept (β0), dvs vad y är när x = 0
        X = np.c_[np.ones(X.shape[0]), X]

        # I koden nedan så multiplicerar vi X med koefficienterna för att få en förutsägelse för y.
        # Vi returnerar sedan en matris där varje rad är en förutsägelse för varje datapunkt i X.

        prediction_results = X @ np.r_[self.intercept, self.coefficients]
        return prediction_results

    def r_squared(self, X, y):
        """
        Beräknar R² (förklaringsgraden) för modellen. Den visar hur bra modellen passar data genom att visa en procentsats.
        X: Oberoende variabler.
        y: Beroende variabler.
        Formel: R2 = 1 - SSE / SST där SSE = Σ(y - y^)² och SST = Σ(y - y_mean)² och SST = Σ(y - y_mean)²

        """

        predicted_values = self.predict(X)

        # Beräkna SSE (Sum of Squared Errors)
        SSE = np.sum((y - predicted_values) ** 2)
        # Beräkna SST (Total Sum of Squares)
        SST = np.sum((y - np.mean(y)) ** 2)

        # Beräkna R²
        R2 = 1 - SSE / SST
        return R2

    def variance(self, X, y):
        """
        Beräknar variansen (σ²) för modellen. Variansen mäter spridningen av felet mellan de observerade värdena (y) och de förutsagda värdena (y^).
        Formel: σ² = SSE / (n - d - 1) där SSE = Σ(y - y^)², n är stickprovsstorleken och d är antalet dimensioner (features).
        """
        predicted_values = self.predict(X)
        SSE = np.sum((y - predicted_values) ** 2)

        n = self.sample_size  # Stickprovsstorleken
        d = self.number_of_features  # Antal dimensioner

        variance = SSE / (n - d - 1)
        return variance

    def standard_deviation(self, X, y):
        """
        Beräknar standardavvikelsen (σ) för modellen. Den visar den genomsnittliga spridningen av felet mellan observerade och förutsagda värden.
        Formel: σ = √(σ²)
        """

        # Beräknar variansen (σ²) för modellen
        variance = self.variance(X, y)

        # Beräknar standardavvikelsen (σ) genom att ta kvadratroten av variansen (σ²)
        standard_deviation = np.sqrt(variance)
        return standard_deviation

    def significance(self, X, y):
        """
        Beräknar F-statistik och signifikansen för regressionen. F-statistiken används för att testa om minst en av de oberoende variablerna har en signifikant effekt på den beroende variabeln.
        Formel: F = (SSR / d) / σ²
        """
        predicted_values = self.predict(X)

        # Beräknar SSR (Sum of Squared Errors)
        sse = np.sum((y - predicted_values) ** 2)
        # Beräknar SST (Total Sum of Squares)
        sst = np.sum((y - np.mean(y)) ** 2)
        # Beräknar SSR (Sum of Squared Regression)
        ssr = sst - sse

        dimensions = self.number_of_features

        variance = self.variance(X, y)

        f_stat = (ssr / dimensions) / variance

        # Beräknar stickprovsstorleken
        n = self.sample_size

        # Beräknar p-värdet, vilket är sannolikheten att få ett F-värde som är lika extremt som det observerade F-värdet. Om p-värdet är mindre än 0.05 så är det signifikant.
        p_value = stats.f.sf(f_stat, dimensions, n - dimensions - 1)

        return f_stat, p_value


model = LinearRegression()

# Tränar modellen:
model.fit(X, y)
print("Intercept (B0):", model.intercept)
print("Koefficienter (B1, ...):", model.coefficients)

# Nya testvärden för X
X_new = np.array([6, 7, 8])

# Testar för att förutsäga med modellen:
y_pred = model.predict(X_new)
print("Förutsagda värden:", y_pred)

# Testar för R2-metoden med samma data:
r2 = model.r_squared(X, y)
print("R²:", r2)

# Kontrollerar antal dimensioner:
print("Antal features (dimensioner):", model.number_of_features)

# Kontrollerar stickprovsstorleken:
print("Antal datapunkter (n):", model.sample_size)

# Beräknar variansen:
variance = model.variance(X, y)
print("Varians (σ²):", variance)

# Beräknar standardavvikelsen:
standard_deviation = model.standard_deviation(X, y)
print("Standardavvikelse (σ):", standard_deviation)

# Beräknar signifikans för regressionen:
f_stat, p_value = model.significance(X, y)
print("F-statistik:", f_stat)
print("p-värde:", p_value)

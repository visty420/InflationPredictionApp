document.addEventListener('DOMContentLoaded', function() {
    updateInputFields();  // Initial call to set up form fields based on default selected model

    document.getElementById('predictionForm').addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(this);
        const model = document.getElementById('modelSelect').value;
        const data = { model: model, features: [], steps: 1 };

        // Collect all input values
        formData.forEach((value, key) => {
            if (key !== 'model') {
                if (model === 'ARIMA') {
                    data.steps = parseInt(value);
                } else {
                    data.features.push(parseFloat(value));
                }
            }
        });

        // Make AJAX call to server
        fetch('/predict/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer yourAuthToken' 
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.innerText = 'Predicted Inflation: ' + data.predicted_inflation;
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});

function updateInputFields() {
    const model = document.getElementById('modelSelect').value;
    const inputFields = document.getElementById('inputFields');
    inputFields.innerHTML = '';

    if (model === 'ARIMA') {
        inputFields.innerHTML = '<input type="number" name="steps" placeholder="Months to predict" required>';
    } else {
        const placeholders = [
            'Consumer Price Index', 
            'Producer Price Index', 
            'Personal Consumption Expenditures',
            'Federal Funds Rate', 
            'Unemployment Rate', 
            'Gross Domestic Product', 
            'Money Supply M2', 
            'Consumer Sentiment', 
            'Wage Growth' 
        ];

        const inputCount = model === 'NN_3' ? 3 : 9;
        for (let i = 0; i < inputCount; i++) {
            inputFields.innerHTML += `<input type="text" name="feature${i + 1}" placeholder="${placeholders[i]}" required>`;
        }
    }
}


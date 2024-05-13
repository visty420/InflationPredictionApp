document.addEventListener('DOMContentLoaded', function() {
    updateInputFields();  // Call initial to set up form fields based on default selected model

    document.getElementById('predictionForm').addEventListener('submit', function(event) {
        event.preventDefault();

        const modelSelect = document.getElementById('modelSelect');
        const model = modelSelect.value;

        let data, endpoint;
        const inputFields = document.getElementById('inputFields').getElementsByTagName('input');

        if (model === "ARIMA") {
            const months = parseInt(inputFields[0].value);
            data = {
                months: months  
            };
            endpoint = '/predict/arima/';  
        } else {
            const features = Array.from(inputFields).map(input => parseFloat(input.value));
            data = {
                model_name: model,
                features: features
            };
            endpoint = '/predict/';  // Endpoint general pentru alte modele
        }

        fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = '';  // Clear previous contents

            if (Array.isArray(data.predicted_inflation)) {
                // Special treatment for ARIMA, where predicted_inflation is an array
                data.predicted_inflation.forEach((inflation, index) => {
                    const monthResult = document.createElement('div');
                    monthResult.innerText = `Month ${index + 1}: Predicted Inflation: ${inflation}`;
                    resultDiv.appendChild(monthResult);
                });
            } else {
                // Treatment for other models
                resultDiv.innerText = 'Predicted Inflation: ' + data.predicted_inflation;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to predict inflation. Check the console for more information.');
        });
    });
});

function updateInputFields() {
    const modelSelect = document.getElementById('modelSelect');
    const model = modelSelect.value;
    const inputFields = document.getElementById('inputFields');
    inputFields.innerHTML = '';

    const placeholders = {
        'ARIMA': ['Months to predict'],
        'LSTM': ['Consumer Price Index', 'Producer Price Index', 'Personal Consumption Expenditures', 'Federal Funds Rate', 'Unemployment Rate', 'Gross Domestic Product', 'Money Supply M2', 'Consumer Sentiment', 'Wage Growth'],
        'NN_3': ['Consumer Price Index', 'Producer Price Index', 'Personal Consumption Expenditures'],
        'NN_9': ['Consumer Price Index', 'Producer Price Index', 'Personal Consumption Expenditures', 'Federal Funds Rate', 'Unemployment Rate', 'Gross Domestic Product', 'Money Supply M2', 'Consumer Sentiment', 'Wage Growth'],
        'RNN': ['Consumer Price Index', 'Producer Price Index', 'Personal Consumption Expenditures', 'Federal Funds Rate', 'Unemployment Rate', 'Gross Domestic Product', 'Money Supply M2', 'Consumer Sentiment', 'Wage Growth']
    };

    const inputCount = placeholders[model].length;
    for (let i = 0; i < inputCount; i++) {
        inputFields.innerHTML += `<input type="text" name="feature${i + 1}" placeholder="${placeholders[model][i]}" required>`;
    }
}
document.addEventListener('DOMContentLoaded', function() {
    updateInputFields();  
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
            endpoint = '/predict/'; 
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
            resultDiv.innerHTML = '';  

            if (Array.isArray(data.predicted_inflation)) {
                
                data.predicted_inflation.forEach((inflation, index) => {
                    const monthResult = document.createElement('div');
                    monthResult.innerText = `Month ${index + 1}: Predicted Inflation: ${inflation}`;
                    resultDiv.appendChild(monthResult);
                });
            } else {
               
                resultDiv.innerText = 'Predicted Inflation: ' + data.predicted_inflation;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to predict inflation. Check the console for more information.');
        });
    });

    document.getElementById('getModelAndScalerButton').addEventListener('click', function() {
        const modelName = document.getElementById('modelSelect').value;
        const token = localStorage.getItem("token"); 
        if (!token) {
            alert("User not authenticated!");
            return;
        }

        fetch('/send_model_scaler', {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer ' + token,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ model_name: modelName })
        })
        .then(response => response.json().then(data => ({ status: response.status, body: data })))
        .then(({ status, body }) => {
            if (status === 200) {
                alert('Model and scaler sent successfully!');
            } else {
                alert(`Failed to send model and scaler: ${body.detail}`);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while sending the model and scaler.');
        });
    });
});

function updateInputFields() {
    const modelSelect = document.getElementById('modelSelect');
    const inputFieldsDiv = document.getElementById('inputFields');

    inputFieldsDiv.innerHTML = ''; 

    const commonFields = [
        { id: 'CPIAUCSL', label: 'Consumer Price Index', placeholder: 'Consumer Price Index' },
        { id: 'PPIACO', label: 'Producer Price Index', placeholder: 'Producer Price Index' },
        { id: 'PCE', label: 'Personal Consumption Expenditures', placeholder: 'Personal Consumption Expenditures' },
        { id: 'FEDFUNDS', label: 'Federal Funds Rate', placeholder: 'Federal Funds Rate' },
        { id: 'UNRATE', label: 'Unemployment Rate', placeholder: 'Unemployment Rate' },
        { id: 'GDP', label: 'Gross Domestic Product', placeholder: 'Gross Domestic Product' },
        { id: 'M2SL', label: 'Money Supply M2', placeholder: 'Money Supply M2' },
        { id: 'UMCSENT', label: 'Consumer Sentiment', placeholder: 'Consumer Sentiment' },
        { id: 'WageGrowth', label: 'Overall Wage Growth', placeholder: 'Overall Wage Growth' }
    ];

    let fieldsToAdd = [];

    switch (modelSelect.value) {
        case 'ARIMA':
            fieldsToAdd = [
                { id: 'ARIMAValue', label: 'ARIMA Input', placeholder: 'ARIMA Input' }
            ];
            break;
        case 'LSTM':
        case 'NN_9':
        case 'RNN':
            fieldsToAdd = commonFields;  
            break;
        case 'NN_3':
            fieldsToAdd = commonFields.slice(0, 3);  
            break;
        default:
            fieldsToAdd = [];  
    }

    fieldsToAdd.forEach(field => {
        const div = document.createElement('div');
        div.className = 'form-group';

        const label = document.createElement('label');
        label.htmlFor = field.id;
        label.textContent = field.label;

        const input = document.createElement('input');
        input.type = 'text';
        input.id = field.id;
        input.name = field.id;
        input.placeholder = field.placeholder;

        div.appendChild(label);
        div.appendChild(input);
        inputFieldsDiv.appendChild(div);
    });
}

function updateModelImage() {
    const modelSelect = document.getElementById('modelSelect');
    const modelImage = document.getElementById('modelImage');
    let imageUrl = '';

    switch(modelSelect.value) {
        case 'LSTM':
            imageUrl = '/auxiliaries/lstm_arhitecture.png';
            break;
        case 'NN_3':
            imageUrl = '/auxiliaries/3inmlp_arhitecture.png';
            break;
        case 'NN_9':
            imageUrl = '/auxiliaries/9inmlp_arhitecture.png';
            break;
        case 'RNN':
            imageUrl = '/auxiliaries/rnn_arhitecture.png';
            break;
        default:
            imageUrl = '';
    }

    if (imageUrl) {
        modelImage.src = imageUrl;
        modelImage.style.display = 'block';
    } else {
        modelImage.style.display = 'none';
    }
}

document.getElementById('modelSelect').addEventListener('change', updateModelImage);
document.addEventListener('DOMContentLoaded', updateInputFields);

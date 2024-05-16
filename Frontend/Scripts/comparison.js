document.addEventListener('DOMContentLoaded', function() {
    const inputFields = document.getElementById('inputFields');
    const inputLabels = [
        { id: 'CPIAUCSL', label: 'Consumer Price Index', placeholder: 'Consumer Price Index' },
        { id: 'PPIACO', label: 'Producer Price Index', placeholder: 'Producer Price Index' },
        { id: 'PCE', label: 'Personal Consumption Expenditures', placeholder: 'Personal Consumption Expenditures' },
        { id: 'FEDFUNDS', label: 'Federal Funds Rate', placeholder: 'Federal Funds Rate' },
        { id: 'UNRATE', label: 'Unemployment Rate', placeholder: 'Unemployment Rate' },
        { id: 'GDP', label: 'Gross Domestic Product', placeholder: 'Gross Domestic Product' },
        { id: 'M2SL', label: 'Money Supply M2', placeholder: 'Money Supply M2' },
        { id: 'UMCSENT', label: 'Consumer Sentiment', placeholder: 'Consumer Sentiment' },
        { id: 'WageGrowth', label: 'Wage Growth', placeholder: 'Wage Growth' }
    ];

    inputLabels.forEach(field => {
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
        input.required = true;

        div.appendChild(label);
        div.appendChild(input);
        inputFields.appendChild(div);
    });

    let chartInstance = null;

    document.getElementById('comparisonForm').addEventListener('submit', function(event) {
        event.preventDefault();

        const inputs = Array.from(inputFields.getElementsByTagName('input')).map(input => parseFloat(input.value));
        
        fetch('/compare_models/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ features: inputs })
        })
        .then(response => response.text())
        .then(text => {
            console.log("Server response:", text);
            const data = JSON.parse(text);
            document.getElementById('nn3Result').innerText = 'NN (3 inputs): ' + data.nn3_prediction;
            document.getElementById('nn9Result').innerText = 'NN (9 inputs): ' + data.nn9_prediction;
            document.getElementById('lstmResult').innerText = 'LSTM: ' + data.lstm_prediction;

            plotResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to compare models. Check the console for more information.');
        });
    });

    function plotResults(data) {
        const ctx = document.getElementById('plot').getContext('2d');

        if (chartInstance) {
            chartInstance.destroy();
        }

        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['NN (3 inputs)', 'NN (9 inputs)', 'LSTM'],
                datasets: [{
                    label: 'Predicted Inflation',
                    data: [parseFloat(data.nn3_prediction), parseFloat(data.nn9_prediction), parseFloat(data.lstm_prediction)],
                    backgroundColor: ['#ff6384', '#36a2eb', '#cc65fe']
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
});

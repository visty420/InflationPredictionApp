document.addEventListener('DOMContentLoaded', function() {
    const inputFields = document.getElementById('inputFields');
    const inputLabels = ['Consumer Price Index', 'Producer Price Index', 'Personal Consumption Expenditures', 'Federal Funds Rate', 'Unemployment Rate', 'Gross Domestic Product', 'Money Supply M2', 'Consumer Sentiment', 'Wage Growth'];
    
    inputLabels.forEach(label => {
        const input = document.createElement('input');
        input.type = 'text';
        input.placeholder = label;
        input.required = true;
        inputFields.appendChild(input);
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

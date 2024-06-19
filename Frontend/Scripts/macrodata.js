document.addEventListener('DOMContentLoaded', function() {
    const inputs = document.querySelectorAll('.form-group input');

    inputs.forEach(input => {
        input.addEventListener('input', function() {
            const value = parseFloat(this.value);
            const min = parseFloat(this.getAttribute('data-min'));
            const max = parseFloat(this.getAttribute('data-max'));
            const warning = this.nextElementSibling;
            if (value < min || value > max) {
                warning.textContent = `Warning: Value might not be real (${min} - ${max})`;
            } else {
                warning.textContent = '';
            }
        });
    });

    document.getElementById('viewDataButton').addEventListener('click', function() {
        fetch('/recent_data')
            .then(response => response.json())
            .then(data => {
                const tableBody = document.querySelector('#data-table tbody');
                tableBody.innerHTML = '';  
                data.forEach(row => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td>${row.date}</td>
                        <td>${row.cpi}</td>
                        <td>${row.ppi}</td>
                        <td>${row.pce}</td>
                        <td>${row.fedfunds}</td>
                        <td>${row.unrate}</td>
                        <td>${row.gdp}</td>
                        <td>${row.m2sl}</td>
                        <td>${row.umcsent}</td>
                        <td>${row.wagegrowth}</td>
                        <td>${row.inflrate}</td>
                    `;
                    tableBody.appendChild(tr);
                });
            });
    });
});

document.getElementById('macroDataForm').addEventListener('submit', function(event) {
    const inputs = document.querySelectorAll('.form-group input');
    inputs.forEach(input => {
        const value = parseFloat(input.value);
        const min = parseFloat(input.getAttribute('data-min'));
        const max = parseFloat(input.getAttribute('data-max'));
        const warning = input.nextElementSibling;
        if (value < min || value > max) {
            warning.textContent = `Warning: Value might not be real (${min} - ${max})`;
        } else {
            warning.textContent = '';
        }
    });
});

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
                        <td>${row.DATE}</td>
                        <td>${row.CPIAUCSL}</td>
                        <td>${row.PPIACO}</td>
                        <td>${row.PCE}</td>
                        <td>${row.FEDFUNDS}</td>
                        <td>${row.UNRATE}</td>
                        <td>${row.GDP}</td>
                        <td>${row.M2SL}</td>
                        <td>${row.UMCSENT}</td>
                        <td>${row['Overall Wage Growth']}</td>
                        <td>${row.INFLRATE}</td>
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

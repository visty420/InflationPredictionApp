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

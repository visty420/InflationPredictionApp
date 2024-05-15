async function login(event) {
    event.preventDefault(); // Previne reîncărcarea paginii la submit

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    const response = await fetch('/token', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded' // Asigură-te că tipul conținutului este corect
        },
        body: new URLSearchParams({
            username: username,
            password: password
        })
    });

    if (response.ok) {
        const data = await response.json();
        localStorage.setItem('token', data.access_token); // Stocăm token-ul în localStorage
        window.location.href = '/factors'; // Redirecționăm utilizatorul
    } else {
        console.error('Login failed');
        alert('Login failed. Please check your credentials and try again.');
    }
}

document.getElementById('loginForm').addEventListener('submit', login);

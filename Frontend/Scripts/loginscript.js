async function login(event) {
    event.preventDefault(); 
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    const response = await fetch('/token', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded' 
        },
        body: new URLSearchParams({
            username: username,
            password: password
        })
    });

    if (response.ok) {
        const data = await response.json();
        localStorage.setItem('token', data.access_token); 
        window.location.href = '/factors'; 
    } else {
        console.error('Login failed');
        alert('Login failed. Please check your credentials and try again.');
    }
}

document.getElementById('loginForm').addEventListener('submit', login);

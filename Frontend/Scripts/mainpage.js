function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) {
        return parts.pop().split(';').shift();
    }
    return null;
}

document.addEventListener("DOMContentLoaded", async function() {
    const welcomeMessage = document.getElementById("welcome-message");
    const logoutButton = document.getElementById("logout-button");
    const loginButton = document.getElementById("login-button");
    const registerButton = document.getElementById("register-button");

    try {
        let token = getCookie("token");
        if (!token) {
            token = localStorage.getItem("token"); 
        }
        console.log("Token:", token);  

        if (!token) {
            welcomeMessage.textContent = "Hello, Guest";
            welcomeMessage.classList.add("welcome-guest");
            welcomeMessage.classList.remove("welcome-user");
            logoutButton.style.display = "none";
            loginButton.style.backgroundColor = "";
            registerButton.style.backgroundColor = "";
            return;
        }

        const response = await fetch("/current_user", {
            headers: {
                "Authorization": `Bearer ${token}`
            }
        });
        const data = await response.json();
        console.log("Response data:", data);  

        if (data.username) {
            welcomeMessage.textContent = `Hello, ${data.username}`;
            welcomeMessage.classList.add("welcome-user");
            welcomeMessage.classList.remove("welcome-guest");
            welcomeMessage.style.fontWeight = "bold";
            welcomeMessage.style.fontSize = "1.2em";
            logoutButton.style.display = "block";
            loginButton.style.backgroundColor = "red";
            registerButton.style.backgroundColor = "red";
            logoutButton.style.backgroundColor = "blue";

            
            loginButton.addEventListener("click", function(event) {
                event.preventDefault();
                alert("You are already logged in!");
            });

            registerButton.addEventListener("click", function(event) {
                event.preventDefault();
                alert("You already have an account!");
            });
        } else {
            welcomeMessage.textContent = "Hello, Guest";
            welcomeMessage.classList.add("welcome-guest");
            welcomeMessage.classList.remove("welcome-user");
            logoutButton.style.display = "none";
            loginButton.style.backgroundColor = "";
            registerButton.style.backgroundColor = "";
        }
    } catch (error) {
        console.error("Error fetching current user:", error);
        welcomeMessage.textContent = "Hello, Guest";
        welcomeMessage.classList.add("welcome-guest");
        welcomeMessage.classList.remove("welcome-user");
        logoutButton.style.display = "none";
        loginButton.style.backgroundColor = "";
        registerButton.style.backgroundColor = "";
    }

    
    logoutButton.addEventListener("click", function() {
        localStorage.removeItem("token");
        document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
        window.location.reload();
    });
});

document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    form.addEventListener("submit", async function(event) {
        event.preventDefault(); 
        const email = document.querySelector("#email").value;
        const emailRegex = /^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$/;

        if (!emailRegex.test(email)) {
            alert("Invalid email format");
            return;
        }

        try {
            const response = await fetch(`/check-email?email=${encodeURIComponent(email)}`);
            const result = await response.json();

            if (response.status === 400) {
                alert(result.detail); 
            } else {
                form.submit(); 
            }
        } catch (error) {
            console.error("Error checking email:", error);
            alert("An error occurred while checking the email. Please try again.");
        }
    });
});
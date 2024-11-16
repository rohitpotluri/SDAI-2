// Add an event listener to the form
document.addEventListener("DOMContentLoaded", () => {
    const form = document.querySelector("form");
    const button = document.querySelector("button");
    const textarea = document.querySelector("textarea");

    // Add a click animation to the button
    button.addEventListener("click", () => {
        button.classList.add("clicked");
        setTimeout(() => {
            button.classList.remove("clicked");
        }, 200);
    });

    // Clear the form after submission
    form.addEventListener("submit", (event) => {
        setTimeout(() => {
            textarea.value = ""; // Clear the textarea after submission
        }, 500);
    });
});

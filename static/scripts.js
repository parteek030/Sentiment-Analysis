document.getElementById("sentimentForm").addEventListener("submit", async function(event) {
    event.preventDefault();
    const formData = new FormData(this);
    const text = formData.get("textInput");

    const response = await fetch("/predict/", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text })
    });

    const data = await response.json();
    document.getElementById("result").innerText = `Sentiment: ${data.sentiment}`;
});

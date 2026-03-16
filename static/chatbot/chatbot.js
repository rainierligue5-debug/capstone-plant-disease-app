function toggleChat() {
    let chat = document.getElementById("chatWindow");
    if(chat.style.display === "flex") {
        chat.style.display = "none";
    } else {
        chat.style.display = "flex";
    }
}

async function sendMessage() {
    let input = document.getElementById("userInput");
    let message = input.value;
    if(message === "") return;

    let chat = document.getElementById("chatbox");
    chat.innerHTML += "<div class='user'>" + message + "</div>";
    input.value = "";

    let response = await fetch("/ai-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
    });

    let data = await response.json();
    chat.innerHTML += "<div class='bot'>" + data.reply + "</div>";
    chat.scrollTop = chat.scrollHeight;
}
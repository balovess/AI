const form = document.getElementById('chat-form');
const input = document.getElementById('chat-input');
const messages = document.getElementById('chat-messages');

form.addEventListener('submit', event => {
  event.preventDefault();
  const message = input.value; 
  addMessage('user', message);// 添加用户输入
  fetch('/api/input', { 
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ message })
  })
  .then(response => response.json())
  .then(data => {
    // 发送用户输入到后端，并从后端接收AI回答
    fetch('/api/output', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ message: data.response })
    })
    .then(response => response.json())
    .then(data => {
      // 处理从后端接收到的AI回答
      const aiResponse = data.response;

      // 在聊天框中添加AI回答
      addMessage('ai', aiResponse);
    });
  });
  input.value = '';
});

function addMessage(sender, message) {
  const messageElement = document.createElement('div');
  messageElement.classList.add('chat-message', sender);
  messageElement.innerText = message;
  messages.appendChild(messageElement);
}




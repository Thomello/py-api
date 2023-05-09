const chatForm = document.querySelector('#chat-form');
const chatInput = document.querySelector('#chat-input');
const chatHistory = document.querySelector('#chat-history');

chatForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = chatInput.value;
  if (!question) return;
  chatInput.value = '';
  const response = await fetch('http://localhost:8000/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question }),
  });
  const { chat_history } = await response.json();
  chatHistory.innerHTML = '';
  chat_history.forEach(([question, answer]) => {
    const questionEl = document.createElement('li');
    questionEl.classList.add('text-gray-700', 'text-sm', 'mb-2');
    questionEl.textContent = question;
    const answerEl = document.createElement('li');
    answerEl.classList.add('text-blue-700', 'text-lg', 'mb-4');
    answerEl.textContent = answer;
    chatHistory.appendChild(questionEl);
    chatHistory.appendChild(answerEl);
  });
});

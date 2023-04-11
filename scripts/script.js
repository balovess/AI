$(document).ready(function() {
	var $messages = $('.messages');

	// Send message when the send button is clicked or enter key is pressed
	$('.send-button, .input-message').click(function() {
		sendMessage();
	});
	$('.input-message').keypress(function(event) {
		if (event.which == 13) {
			sendMessage();
			event.preventDefault();
		}
	});

	function sendMessage() {
		var message = $('.input-message').val();
		if (message.trim() == '') {
			return;
		}
		appendMessage('user', message);
		$('.input-message').val('');

		// Send message to the backend API and get the response
		$.ajax({
			url: '/api/chatbot',
			type: 'POST',
			dataType: 'json',
			data: { message: message },
			success: function(response) {
				appendMessage('bot', response.message);
			},
			error: function() {
				appendMessage('bot', 'Sorry, something went wrong.');
			}
		});
	}

	function appendMessage(sender, message) {
		var $message = $('<div>').addClass('message').addClass(sender + '-message').text(message);
		$messages.append($message);
		$messages.scrollTop($messages[0].scrollHeight);
	}
});